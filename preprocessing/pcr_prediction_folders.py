import argparse
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess DCE-MRI volumes for pCR classification.")
    parser.add_argument("--data-root", default="data",
                        help="Root directory containing images/, segmentations/ and clinical_and_imaging_info.xlsx")
    parser.add_argument("--margin", type=int, default=10,
                        help="Bounding box margin in voxels when cropping to tumour region")
    parser.add_argument("--no-diffs", action="store_true",
                        help="Disable inter-phase difference channels")
    parser.add_argument("--no-full-breast", action="store_true",
                        help="Skip saving full-breast crops")
    return parser.parse_args()


def crop_to_bbox(image, mask, margin=10):
    coords = np.array(np.where(mask))
    minc = np.maximum(coords.min(axis=1) - margin, 0)
    maxc = np.minimum(coords.max(axis=1) + margin, mask.shape)
    slices = tuple(slice(minc[i], maxc[i]) for i in range(3))
    return image[:, slices[0], slices[1], slices[2]], mask[slices[0], slices[1], slices[2]]


def crop_breast_containing_mask(image_stack, mask):
    """Isolate the breast that contains the tumour (used for DUKE patients)."""
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return image_stack, mask

    center_x = coords[1].mean()
    full_x_dim = image_stack.shape[2]
    mid_x = full_x_dim / 2

    slices_x = slice(0, int(mid_x)) if center_x < mid_x else slice(int(mid_x), full_x_dim)
    return image_stack[:, :, :, slices_x], mask[:, :, slices_x]


def main():
    args = parse_args()

    SRC_ROOT = args.data_root
    IMG_DIR = os.path.join(SRC_ROOT, "images")
    SEG_EXPERT_DIR = os.path.join(SRC_ROOT, "segmentations", "expert")
    CLINICAL_XLSX = os.path.join(SRC_ROOT, "clinical_and_imaging_info.xlsx")
    DEST_ROOT = os.path.join(SRC_ROOT, "processed_dataset")
    DEST_JSON = os.path.join(SRC_ROOT, "train.json")

    USE_DIFFS = not args.no_diffs
    PROCESS_FULL_BREAST = not args.no_full_breast
    MARGIN_SIZE = args.margin

    clin_df = pd.read_excel(CLINICAL_XLSX)
    pcr_map = dict(zip(clin_df["patient_id"], clin_df["pcr"]))

    datalist = []

    for pid in os.listdir(IMG_DIR):
        patient_path = os.path.join(IMG_DIR, pid)
        if not os.path.isdir(patient_path):
            continue

        pid_lower = pid.lower()
        is_duke_patient = "duke" in pid_lower

        image_array_list = []
        phase_index = 0
        while True:
            full_path = os.path.join(patient_path, f"{pid_lower}_000{phase_index}.nii.gz")
            if not os.path.exists(full_path):
                break
            image_array_list.append(nib.load(full_path).get_fdata())
            phase_index += 1

        if not image_array_list:
            continue

        seg_path = os.path.join(SEG_EXPERT_DIR, f"{pid_lower}.nii.gz")
        if not os.path.exists(seg_path):
            continue

        seg_np = (nib.load(seg_path).get_fdata() > 0).astype(np.uint8)
        if seg_np.sum() == 0:
            continue

        label = pcr_map.get(pid)
        if label not in [0, 1]:
            continue

        image_stack = np.stack(image_array_list, axis=0)

        out_dir_patient = os.path.join(DEST_ROOT, pid)
        os.makedirs(out_dir_patient, exist_ok=True)
        with open(os.path.join(out_dir_patient, "label.txt"), "w") as f:
            f.write(str(int(label)))

        data_entry = {"patient_id": pid, "label": int(label)}

        if PROCESS_FULL_BREAST:
            full_img = image_stack.copy()
            full_mask = seg_np.copy()
            if is_duke_patient:
                full_img, full_mask = crop_breast_containing_mask(full_img, full_mask)

            out_dir_full = os.path.join(out_dir_patient, "full_breast")
            os.makedirs(out_dir_full, exist_ok=True)
            np.save(os.path.join(out_dir_full, "images.npy"), full_img)
            np.save(os.path.join(out_dir_full, "mask.npy"), full_mask)
            data_entry.update({
                "full_breast_path": os.path.join(pid, "full_breast"),
                "is_duke": is_duke_patient,
                "full_breast_num_phases": full_img.shape[0],
                "full_breast_shape": full_img.shape[1:],
                "full_breast_mask_shape": full_mask.shape,
            })

        cropped_img, cropped_seg = crop_to_bbox(image_stack.copy(), seg_np.copy(), margin=MARGIN_SIZE)
        if USE_DIFFS and cropped_img.shape[0] > 1:
            diffs = [cropped_img[i + 1] - cropped_img[i] for i in range(cropped_img.shape[0] - 1)]
            cropped_img = np.concatenate((cropped_img, np.stack(diffs, axis=0)), axis=0)

        out_dir_cropped = os.path.join(out_dir_patient, "cropped")
        os.makedirs(out_dir_cropped, exist_ok=True)
        np.save(os.path.join(out_dir_cropped, "images.npy"), cropped_img)
        np.save(os.path.join(out_dir_cropped, "mask.npy"), cropped_seg)
        data_entry.update({
            "cropped_path": os.path.join(pid, "cropped"),
            "cropped_num_phases": cropped_img.shape[0],
            "cropped_shape": cropped_img.shape[1:],
            "cropped_mask_shape": cropped_seg.shape,
            "cropped_with_diffs": USE_DIFFS,
        })

        datalist.append(data_entry)

    with open(DEST_JSON, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"Processed {len(datalist)} patients → {DEST_JSON}")


if __name__ == "__main__":
    main()
