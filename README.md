# MAMMA MIA Challenge — Breast Tumour pCR Prediction

<div align="center">

This project was developed as part of the **[MAMMA MIA Challenge](https://mamma-mia-challenge.grand-challenge.org/)** — an international medical imaging competition for predicting pathological complete response (pCR) to neoadjuvant chemotherapy in breast cancer patients using DCE-MRI and clinical data.

**Kacper Janiszewski · Krzysztof Nosek · Róża Mazurek**

</div>

---

## Clinical Background

Neoadjuvant chemotherapy (NAC) is administered before surgery to shrink breast tumours. The key outcome metric is **pathological Complete Response (pCR)**: whether any invasive cancer remains after treatment.

| Label | Meaning |
|-------|---------|
| `pCR = 1` | No invasive cancer after treatment — complete response |
| `pCR = 0` | Residual cancer still present |

Predicting pCR before or early in treatment allows clinicians to personalise therapy and avoid ineffective regimens. This project explores both imaging-based (3D deep learning on DCE-MRI) and tabular (clinical data) approaches.

---

## Dataset

The challenge dataset comprises:

- **1,506 breast cancer patients** from the ISPY2 and DUKE clinical cohorts
- **Multi-phase DCE-MRI** volumes — dynamic contrast-enhanced scans acquired at sequential time points after contrast injection
- **Expert tumour segmentations** in NIfTI format for each patient
- **Clinical metadata** — demographics, tumour characteristics (ER/PR/HER2, grade), NAC agent information

---

## Data Preparation

### 1. Image Preprocessing

**Script:** `preprocessing/pcr_prediction_folders.py`

For each patient:
1. Load all DCE-MRI phases and the expert segmentation mask
2. For DUKE cohort: crop to the breast containing the tumour (axial midline split)
3. Crop the full volume to a tight bounding box around the tumour + configurable margin
4. Compute inter-phase difference channels to capture temporal contrast dynamics
5. Save cropped volumes as `.npy` arrays and write a `train.json` manifest

<br>

<div align="center">

![Segmentation mask applied to tumour region](assets/Mask.png)

*Expert segmentation mask applied to the DCE-MRI volume*

</div>

<br>

Slices along each axis after cropping:

<div align="center">

![Axis X](assets/Mask_X.png)

*Axis X*

![Axis Y](assets/Mask_Y.png)

*Axis Y*

![Axis Z](assets/Mask_Z.png)

*Axis Z*

</div>

---

### 2. Nottingham TIC Features

**Script:** `preprocessing/nothingam_scale.py`

For each patient we compute a **Time-Intensity Curve (TIC)** per tumour voxel — how contrast signal intensity changes across MRI phases.

<div align="center">

![TIC curves for 6 random tumour voxels](assets/tics.png)

*TIC curves for 6 randomly selected voxels within one patient's tumour*

</div>

<br>

From each voxel's TIC we extract three features:

| Feature | Definition |
|---------|------------|
| **Wash-in rate** | Speed of contrast uptake after injection |
| **Wash-out enhancement** | Change in intensity from peak to final phase |
| **Wash-out stability** | Linearity of the post-peak signal (RSS of linear fit) |

Wash-out enhancement determines the **Nottingham TIC type**, which correlates with tumour aggressiveness:

| Type | Wash-out | Interpretation |
|------|----------|----------------|
| Type I | > 0.05 | Persistent — low-grade |
| Type II | −0.05 to 0.05 | Plateau — intermediate |
| Type III | < −0.05 | Wash-out — high-grade, more likely pCR |

<div align="center">

![TIC type classification diagram](assets/TIC_Type.png)

*Wash-out curve types and their clinical interpretation*

![Wash-out stability calculation](assets/Stability.png)

*How wash-out stability is computed from the post-peak linear fit*

</div>

<br>

Per-patient aggregates (dominant type, average TIC features, voxel counts per type) are saved to `data/nottingham_summary.csv`.

---

### 3. Clinical Feature Engineering

**Module:** `models/tabular/preprocessing_pipeline.py`

The clinical dataset had many missing Nottingham Grade values. We imputed them using image-derived dominant TIC types. All features pass through a scikit-learn pipeline that:

- Drops acquisition metadata and patient identifiers
- Imputes missing values (median, constant, "unknown")
- Corrects NAC agent name typos and binary-encodes each drug substance
- One-hot encodes remaining categorical columns

---

## Models

### FusionPCRNet — Main Model

**Location:** `models/fusion_pcr_net/`

A multimodal architecture that fuses 3D imaging features with per-patient TIC statistics.

```
Input: 6-channel 3D volume (5 DCE-MRI phases + 1 mask), resized to 128×128×128

CNN Branch
  └─ Pre-trained R3D-18 (Kinetics400)
       └─ First conv adapted: 3 → 6 input channels
            (pretrained weights for ch 0-2, mean-initialised for ch 3-5)
       └─ SpatialAttention3D (CBAM-style) after layers 2, 3, 4
       └─ Global average pooling → Linear projection 512 → 128
                                                            ┐
TIC Branch (4 aggregated features per patient)              ├─ concat → 192-dim
  └─ FC 4 → 32 → 64  (LayerNorm, batch-size independent)   ┘

Fusion Head
  └─ Linear 192 → 128 → 2   (CrossEntropyLoss)
```

**Key design decisions:**

- **CBAM spatial attention** on deeper ResNet layers to focus on tumour-relevant regions
- **LayerNorm** in the TIC branch — avoids BatchNorm dependency on batch size during medical imaging training with small batches
- **GroupNorm** in the CNN projection head for the same reason
- Early ResNet layers frozen (`freeze_until="layer1"`) to preserve low-level spatiotemporal features while adapting to medical data
- TIC features computed **before** intensity augmentation to avoid corrupting the physiological signal

---

### Simple3DCNN — Baseline

**Location:** `models/simple_3d_cnn/`

Lightweight 3D CNN without pretrained weights, used as a baseline.

```
Input: 11-channel volume (5 phases + 5 phase differences + 1 mask)

Conv3D(11 → 64)  + BN + ReLU + MaxPool3D
Conv3D(64 → 128) + BN + ReLU + MaxPool3D
Conv3D(128 → 256) + BN + ReLU + AdaptiveAvgPool3D(1)

Flatten → Linear(256 → 1024) → ReLU → Linear(1024 → 1) → Sigmoid   (BCELoss)
```

Variable-size volumes are padded to a common shape via a custom `pad_collate` function.

---

### Hard Gating Mixture of Experts — Tabular

**Location:** `models/tabular/hard_gating_moe_model.py`

An ensemble of two tree-based experts with a learned routing model, trained on preprocessed clinical features.

```
Expert 1 — XGBoost  (hyperparameters tuned with Optuna)
  n_estimators=124 · max_depth=3 · lr=0.041 · subsample=0.978

Expert 2 — RandomForest
  n_estimators=100

Gating model — LightGBM  (also Optuna-tuned)
  Labels: 1 = RF correct & XGB wrong  →  route to RF
          0 = otherwise               →  route to XGB
```

---

### PCRNet — Tabular MLP

**Location:** `models/tabular/pytorch_model_nottingham.py`

Simple MLP baseline on the merged Nottingham TIC + clinical features.

```
StandardScaler → Linear(d → 64) → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1) → Sigmoid
BCELoss
```

---

## Statistics

<div align="center">

![Distribution of Nottingham grades and TIC types](assets/Statistics.png)

*Distribution of Nottingham grades and TIC types across patients*

</div>

<br>

<div align="center">

![Wash-in rate vs wash-out enhancement](assets/WashInOut.png)

*Scatter plot: wash-in rate vs wash-out enhancement, coloured by pCR label*

</div>

<br>

<div align="center">

![Distribution plot](assets/Plot.png)

</div>

---

## Project Structure

```
├── assets/                         # Figures and plots
├── data/
│   └── nottingham_summary.csv      # Pre-computed TIC features (434 patients)
│   # clinical xlsx files excluded — obtain from the MAMMA MIA challenge
├── hpc_scripts/                    # SLURM job scripts used on HPC cluster
├── models/
│   ├── fusion_pcr_net/             # FusionPCRNet — primary model
│   │   ├── model.py
│   │   ├── dataset.py
│   │   └── train.py
│   ├── simple_3d_cnn/              # Baseline 3D CNN
│   │   ├── dataset_mamma_mia.py
│   │   └── 3d_train_cnn.py
│   └── tabular/                    # Tabular models
│       ├── preprocessing_pipeline.py
│       ├── hard_gating_moe_model.py
│       └── pytorch_model_nottingham.py
├── notebooks/
│   └── viz.ipynb                   # Visualisation of pipeline and TIC curves
├── preprocessing/
│   ├── pcr_prediction_folders.py   # Crop and organise raw NIfTI volumes
│   └── nothingam_scale.py          # Compute Nottingham TIC features
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

All scripts are run from the **project root**:

```bash
# Step 1 — preprocess raw DCE-MRI images
python preprocessing/pcr_prediction_folders.py --data-root data

# Step 2 — compute Nottingham TIC features
python preprocessing/nothingam_scale.py --data-root data --output-csv data/nottingham_summary.csv

# Step 3a — train FusionPCRNet (main model)
python models/fusion_pcr_net/train.py --data-root data --epochs 50

# Step 3b — train baseline 3D CNN
python models/simple_3d_cnn/3d_train_cnn.py --data-root data

# Step 3c — train tabular MoE
python models/tabular/hard_gating_moe_model.py

# Step 3d — train tabular MLP on TIC features
python models/tabular/pytorch_model_nottingham.py
```

---

## Technologies

| Library | Purpose |
|---------|---------|
| PyTorch + torchvision | Deep learning models, R3D-18 backbone |
| nibabel | Medical image I/O (`.nii.gz`) |
| scikit-learn | Preprocessing pipeline, metrics |
| XGBoost / LightGBM | Tabular expert models |
| pandas / numpy | Data manipulation |
| Optuna | Hyperparameter optimisation for MoE gating |
