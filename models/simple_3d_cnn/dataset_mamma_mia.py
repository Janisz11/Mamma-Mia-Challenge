import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import pandas as pd

class MammaMiaDataset(Dataset):
    def __init__(self, image_root, seg_root, clinical_xlsx, margin=10, crop=True, use_diffs=True, skip_missing_labels=False):
        self.image_root = image_root
        self.seg_root = seg_root
        self.margin = margin
        self.crop = crop
        self.use_diffs = use_diffs
        self.skip_missing_labels = skip_missing_labels

        self.pids = [pid for pid in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, pid))]
        df = pd.read_excel(clinical_xlsx)
        self.pcr_map = dict(zip(df['patient_id'], df['pcr']))

        if self.skip_missing_labels:
            self.pids = [pid for pid in self.pids if self.pcr_map.get(pid) in [0, 1]]

    def _crop_to_bbox(self, image, mask):
        coords = np.array(np.where(mask))
        if coords.size == 0:
            return image, mask
        minc = np.maximum(coords.min(axis=1) - self.margin, 0)
        maxc = np.minimum(coords.max(axis=1) + self.margin, mask.shape)
        slices = tuple(slice(minc[i], maxc[i]) for i in range(3))
        return image[:, slices[0], slices[1], slices[2]], mask[slices[0], slices[1], slices[2]]

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        pid_lower = pid.lower()
        img_dir = os.path.join(self.image_root, pid)
        seg_path = os.path.join(self.seg_root, f"{pid_lower}.nii.gz")

        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing segmentation for {pid}")

        seg = nib.load(seg_path).get_fdata()
        seg = (seg > 0).astype(np.uint8)

        phases = []
        i = 0
        while True:
            img_path = os.path.join(img_dir, f"{pid_lower}_000{i}.nii.gz")
            if not os.path.exists(img_path):
                break
            phase = nib.load(img_path).get_fdata()
            phases.append(phase)
            i += 1

        if len(phases) == 0:
            raise RuntimeError(f"No MRI phases found for {pid}")

        image = np.stack(phases, axis=0)  # [C, D, H, W]

        if self.crop:
            image, seg = self._crop_to_bbox(image, seg)

        if self.use_diffs and image.shape[0] > 1:
            diffs = [image[i+1] - image[i] for i in range(image.shape[0] - 1)]
            image = np.concatenate((image, np.stack(diffs, axis=0)), axis=0)

        label = self.pcr_map.get(pid)
        if label not in [0, 1]:
            raise ValueError(f"Invalid label for {pid}")

        return torch.from_numpy(image).float(), torch.tensor(label).float()