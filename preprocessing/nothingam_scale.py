import argparse
import os
import csv
import numpy as np
import nibabel as nib
from sklearn.linear_model import LinearRegression


def parse_args():
    parser = argparse.ArgumentParser(description="Compute per-patient Nottingham TIC features from DCE-MRI.")
    parser.add_argument("--data-root", default="data",
                        help="Root directory containing images/ and segmentations/expert/")
    parser.add_argument("--output-csv", default="data/nottingham_summary.csv",
                        help="Path to the output CSV file")
    return parser.parse_args()


def generate_voxelwise_tic_curves(image_stack, mask):
    tic_curves = {}
    I0 = image_stack[0].astype(np.float32) + 1e-8
    for idx in zip(*np.where(mask > 0)):
        curve = [image_stack[t][idx] / I0[idx] for t in range(image_stack.shape[0])]
        tic_curves[idx] = curve
    return tic_curves


def compute_tic_features(tic):
    tic = np.array(tic, dtype=np.float32)
    eps = 1e-8
    n_phases = len(tic)

    baseline = tic[0]
    peak_idx = np.argmax(tic)
    peak_val = tic[peak_idx]
    last_val = tic[-1]

    wash_in_rate = (peak_val - baseline) / (peak_idx + eps)
    wash_out_enhancement = (last_val - peak_val) / (peak_val + eps)

    if peak_idx < n_phases - 1:
        x = np.arange(peak_idx, n_phases).reshape(-1, 1)
        y = tic[peak_idx:]
        y_pred = LinearRegression().fit(x, y).predict(x)
        rss = np.sum((y - y_pred) ** 2)
        wash_out_stability = rss / ((baseline + eps) * (n_phases - peak_idx))
    else:
        wash_out_stability = 0.0

    return {
        "wash_in_rate": wash_in_rate,
        "wash_out_enhancement": wash_out_enhancement,
        "wash_out_stability": wash_out_stability,
    }


def classify_nottingham_type(features):
    """Assign Nottingham Type I/II/III based on wash-out enhancement."""
    wash_out = features["wash_out_enhancement"]
    if wash_out > 0.05:
        return "Type I"    # persistent enhancement
    elif wash_out >= -0.05:
        return "Type II"   # plateau
    else:
        return "Type III"  # wash-out (associated with malignancy)


def classify_all_voxels(tic_curves):
    type_counts = {"Type I": 0, "Type II": 0, "Type III": 0}
    totals = {"wash_in_rate": 0.0, "wash_out_enhancement": 0.0, "wash_out_stability": 0.0}
    voxel_count = 0

    for tic in tic_curves.values():
        features = compute_tic_features(tic)
        type_counts[classify_nottingham_type(features)] += 1
        for k in totals:
            totals[k] += features[k]
        voxel_count += 1

    avg = {k: v / voxel_count for k, v in totals.items()} if voxel_count > 0 else {k: 0.0 for k in totals}
    return type_counts, avg, voxel_count


def main():
    args = parse_args()
    IMG_DIR = os.path.join(args.data_root, "images")
    SEG_DIR = os.path.join(args.data_root, "segmentations", "expert")

    header = [
        "patient_id", "type_I_count", "type_II_count", "type_III_count", "dominant_type",
        "avg_wash_in_rate", "avg_wash_out_enhancement", "avg_wash_out_stability", "voxel_count",
    ]

    with open(args.output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for pid in os.listdir(IMG_DIR):
            patient_path = os.path.join(IMG_DIR, pid)
            if not os.path.isdir(patient_path):
                continue

            pid_lower = pid.lower()

            phases = []
            phase_index = 0
            while True:
                full_path = os.path.join(patient_path, f"{pid_lower}_000{phase_index}.nii.gz")
                if not os.path.exists(full_path):
                    break
                phases.append(nib.load(full_path).get_fdata())
                phase_index += 1

            if not phases:
                continue

            seg_path = os.path.join(SEG_DIR, f"{pid_lower}.nii.gz")
            if not os.path.exists(seg_path):
                continue

            seg_np = (nib.load(seg_path).get_fdata() > 0).astype(np.uint8)
            if seg_np.sum() == 0:
                continue

            image_stack = np.stack(phases, axis=0)
            tic_curves = generate_voxelwise_tic_curves(image_stack, seg_np)
            type_counts, avg, voxel_count = classify_all_voxels(tic_curves)
            dominant_type = max(type_counts, key=type_counts.get)

            writer.writerow([
                pid,
                type_counts["Type I"], type_counts["Type II"], type_counts["Type III"],
                dominant_type,
                avg["wash_in_rate"], avg["wash_out_enhancement"], avg["wash_out_stability"],
                voxel_count,
            ])

    print(f"Saved Nottingham features → {args.output_csv}")


if __name__ == "__main__":
    main()
