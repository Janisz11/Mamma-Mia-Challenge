import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression

TARGET_SHAPE = (128, 128, 128)
MAX_PHASES = 5   # max DCE-MRI phases
MARGIN = 60      # bounding box margin in voxels
EPS = 1e-8


def _normalize_stack(phase_list_np):
    """Z-score normalise all phases using mean/std computed from the pre-contrast phase."""
    baseline = phase_list_np[0].astype(np.float32)
    mu = baseline.mean()
    std = baseline.std() + EPS
    return [(ph.astype(np.float32) - mu) / std for ph in phase_list_np]


def _augment_volumes(phase_vols_np, mask_np):
    """Apply consistent spatial and intensity augmentations to all phases and the mask."""
    for ax in (0, 1, 2):
        if random.random() < 0.5:
            phase_vols_np = [np.flip(v, axis=ax).copy() for v in phase_vols_np]
            mask_np = np.flip(mask_np, axis=ax).copy()

    k = random.randint(0, 3)
    if k:
        phase_vols_np = [np.rot90(v, k=k, axes=(1, 2)).copy() for v in phase_vols_np]
        mask_np = np.rot90(mask_np, k=k, axes=(1, 2)).copy()

    if random.random() < 0.5:
        sigma = 0.02
        phase_vols_np = [
            v + np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
            for v in phase_vols_np
        ]

    scale = np.random.normal(loc=1.0, scale=0.05)
    shift = np.random.normal(loc=0.0, scale=0.05)
    phase_vols_np = [(v * scale + shift).astype(np.float32) for v in phase_vols_np]

    return phase_vols_np, mask_np


def _generate_tic_curves(image_stack: np.ndarray, mask: np.ndarray):
    """Return {voxel_index: [I/I0 per phase]} for all tumour voxels."""
    tic_curves = {}
    I0 = image_stack[0].astype(np.float32) + EPS
    for idx in zip(*np.where(mask > 0)):
        tic_curves[idx] = (image_stack[:, idx[0], idx[1], idx[2]] / I0[idx]).tolist()
    return tic_curves


def _compute_voxel_tic_features(tic):
    tic = np.array(tic, dtype=np.float32)
    n_phases = len(tic)
    baseline = tic[0]
    peak_idx = int(np.argmax(tic))
    peak_val = tic[peak_idx]

    wash_in_rate = (peak_val - baseline) / (peak_idx + EPS)
    wash_out_enh = (tic[-1] - peak_val) / (peak_val + EPS)

    if peak_idx < n_phases - 1:
        x = np.arange(peak_idx, n_phases).reshape(-1, 1)
        y = tic[peak_idx:]
        y_pred = LinearRegression().fit(x, y).predict(x)
        rss = np.sum((y - y_pred) ** 2)
        wash_out_stab = rss / ((baseline + EPS) * (n_phases - peak_idx))
    else:
        wash_out_stab = 0.0

    return wash_in_rate, wash_out_enh, wash_out_stab


def _aggregate_tic_features(image_stack: np.ndarray, mask: np.ndarray):
    """Return tensor [log_voxel_count, avg_wash_in, avg_wash_out_enh, avg_wash_out_stab]."""
    tic_curves = _generate_tic_curves(image_stack, mask)
    n = len(tic_curves)
    if n == 0:
        return torch.zeros(4)

    total_in = total_out = total_stab = 0.0
    for tic in tic_curves.values():
        wi, wo_e, wo_s = _compute_voxel_tic_features(tic)
        total_in += wi
        total_out += wo_e
        total_stab += wo_s

    log_n = torch.log10(torch.tensor([n + EPS]))
    avgs = torch.tensor([total_in / n, total_out / n, total_stab / n], dtype=torch.float32)
    return torch.cat([log_n, avgs], dim=0)


class MammaMiaCompetitionDataset(Dataset):
    def __init__(self, patient_ids, images_root, clinical_xlsx, segmentation_root):
        self.patient_ids = patient_ids
        self.images_root = images_root
        self.segmentation_root = segmentation_root
        clin_df = pd.read_excel(clinical_xlsx).dropna(subset=["pcr"])
        self.labels = dict(zip(clin_df["patient_id"], clin_df["pcr"].astype(int)))

    def _to_tensor(self, vol_np):
        return torch.tensor(vol_np[None, ...], dtype=torch.float32)  # [1, D, H, W]

    def _resize(self, vol_t, target_shape):
        return F.interpolate(
            vol_t.unsqueeze(0), size=target_shape, mode="trilinear", align_corners=False
        ).squeeze(0)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pid_dir = os.path.join(self.images_root, pid)

        phase_paths = sorted(
            [os.path.join(pid_dir, f) for f in os.listdir(pid_dir) if f.endswith(".nii.gz")]
        )[:MAX_PHASES]

        phase_vols_np = [nib.load(p).get_fdata().astype(np.float32) for p in phase_paths]
        while len(phase_vols_np) < MAX_PHASES:
            phase_vols_np.append(np.zeros_like(phase_vols_np[0]))

        phase_vols_np = _normalize_stack(phase_vols_np)

        mask_path = os.path.join(self.segmentation_root, f"{pid}.nii.gz")
        if os.path.exists(mask_path):
            mask_np = (nib.load(mask_path).get_fdata() > 0).astype(np.uint8)
        else:
            mask_np = np.zeros_like(phase_vols_np[0])

        nz = np.where(mask_np > 0)
        if len(nz[0]) > 0:
            zmin, ymin, xmin = np.min(nz[0]), np.min(nz[1]), np.min(nz[2])
            zmax, ymax, xmax = np.max(nz[0]), np.max(nz[1]), np.max(nz[2])
            zmin = max(zmin - MARGIN, 0)
            ymin = max(ymin - MARGIN, 0)
            xmin = max(xmin - MARGIN, 0)
            zmax = min(zmax + MARGIN + 1, mask_np.shape[0])
            ymax = min(ymax + MARGIN + 1, mask_np.shape[1])
            xmax = min(xmax + MARGIN + 1, mask_np.shape[2])
            phase_vols_np = [v[zmin:zmax, ymin:ymax, xmin:xmax] for v in phase_vols_np]
            mask_np = mask_np[zmin:zmax, ymin:ymax, xmin:xmax]

        # TIC features computed before intensity augmentation to avoid corrupting the curves
        image_stack_np = np.stack(phase_vols_np, axis=0)  # [T, D, H, W]
        tic_features = _aggregate_tic_features(image_stack_np, mask_np)

        phase_vols_np, mask_np = _augment_volumes(phase_vols_np, mask_np)

        phase_vols_t = [self._resize(self._to_tensor(v), TARGET_SHAPE) for v in phase_vols_np]
        mask_t = self._resize(self._to_tensor(mask_np.astype(np.float32)), TARGET_SHAPE)

        while len(phase_vols_t) < MAX_PHASES:
            phase_vols_t.append(torch.zeros((1, *TARGET_SHAPE)))

        x_img = torch.cat(phase_vols_t, dim=0)           # [MAX_PHASES, D, H, W]
        x = torch.cat([x_img, mask_t], dim=0)            # +1 mask channel

        y = torch.tensor(self.labels[pid], dtype=torch.long)
        return x, y, tic_features
