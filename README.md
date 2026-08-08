# Handwriting-Based Biometric Identification System (HB-BIS)

A complete, end-to-end biometric identification system that uses handwriting
analysis for person identification. It implements the full biometric pipeline —
image acquisition, preprocessing, feature extraction, encrypted storage,
similarity decision-making and a Tkinter GUI — using **two fundamentally
different approaches**:

- **Classical ML (SVM)** with 42 interpretable handcrafted features
- **Deep Learning (SqueezeNet)** with 512-dimensional learned embeddings,
  optionally fine-tuned with triplet loss

A fully **reproducible evaluation suite** (`experiments/`) benchmarks both
models under a rigorous protocol: closed-set identification, verification
(FAR/FRR/EER), robustness to acquisition degradations, preprocessing ablations,
and transfer-learning (fine-tuning) analysis.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)

---

## Features

### Dual-Model Feature Extraction
- **SVM**: 42-dimensional handcrafted features — ink density (4), stroke
  orientation histogram (8), curvature statistics (6), geometry (8), spacing /
  gaps (6), statistical moments (10). Fast (~30 ms/img) and interpretable.
- **SqueezeNet**: 512-dimensional deep embeddings from an ImageNet-pretrained
  SqueezeNet 1.1 (auto-downloaded on first use, cached by torchvision).
  Optional **triplet-loss fine-tuning / retraining** to adapt the embedding to
  the enrolled population (`models/saved/squeezenet_finetuned.pth`).

### Complete Biometric Pipeline (6 Layers)
1. **Data Acquisition** — image loading, format/dimension/contrast validation
2. **Preprocessing** — Gaussian denoising, CLAHE lighting normalization, Otsu
   binarization, size normalization (224x224), morphological stroke
   normalization
3. **Feature Engineering** — SVM and SqueezeNet extraction
4. **Database** — encrypted storage of raw images + feature templates per user
5. **Decision** — cosine-distance matching with accept / uncertain / reject
   thresholds and duplicate-enrollment warnings
6. **GUI** — Tkinter interface with Identification, Enrollment and Management
   (stats + retraining) modes

### Encryption
- XOR + Base64 encryption of images and feature templates.

### Evaluation Suite
- Fully reproducible, seeded experiments (identification, verification,
  robustness, ablations, fine-tuning, efficiency) with CSV tables, Markdown
  summary and PNG figures.

---

## System Architecture

```
HB-BIS/
├── main.py                     # Entry point (dependency check + GUI launch)
├── config.py                   # All tunable system parameters & thresholds
├── requirements.txt            # Dependencies
├── run_experiments.bat         # One-command evaluation (Windows)
├── conftest.py                 # Pytest path setup
│
├── layers/                     # The 6 modular layers
│   ├── data_acquisition.py    # Layer 1: image I/O & validation
│   ├── preprocessing.py       # Layer 2: enhancement & normalization
│   ├── feature_engineering.py # Layer 3: SVM + SqueezeNet features
│   ├── database.py            # Layer 4: encrypted storage
│   ├── decision.py            # Layer 5: similarity & accept/reject logic
│   └── gui.py                 # Layer 6: user interface
│
├── models/
│   ├── squeezenet_model.py    # Triplet-loss retraining logic
│   └── saved/                 # Created at runtime: squeezeNet_finetuned.pth
│
├── utils/
│   ├── encryption.py          # XOR/Base64 encryption
│   └── validators.py          # Input validation helpers
│
├── experiments/               # Rigorous evaluation framework
│   ├── config.py             # Experiment hyper-parameters (seed, sizes, ...)
│   ├── synthetic_data.py     # Seeded synthetic "writers" generator
│   ├── dataset.py            # Benchmark construction + enrollment/probe splits
│   ├── features.py           # Batched feature-extraction wrapper
│   ├── metrics.py            # rank-N, FAR/FRR/EER, confusion matrices
│   ├── robustness.py         # Acquisition perturbations (blur, noise, ...)
│   ├── ablations.py          # One-stage-removed preprocessing variants
│   ├── finetune.py           # Triplet-loss training loop
│   ├── plots.py              # Evaluation figures
│   ├── report.py             # CSV/JSON/Markdown result serialization
│   └── run_experiments.py    # Entry point for the whole suite
│
├── tests/                     # Unit tests (preprocessing, features,
│                              # decision, database, encryption, metrics,
│                              # synthetic data)
├── database/                  # Created at runtime
│   ├── metadata.json          # User registry
│   └── users/{user_id}/       # raw_images/, svm_features/, squeezenet_embeddings/
│
├── results/                   # Evaluation outputs (generated)
│   ├── SUMMARY.md             # Human-readable summary of all experiments
│   ├── *.csv                  # Raw result tables
│   ├── *.png                  # Figures (rank curve, ROC/DET, confusion, ...)
│   └── environment.json       # Hardware/software snapshot for reproducibility
└── assets/                    # Screenshots etc.
```

---

## Installation

Requires **Python 3.9+** (verified on Python 3.12). Tkinter ships with Python.

### Step 1 — Install dependencies

```bash
cd HB-BIS
pip install -r requirements.txt
```

> `requirements.txt` targets CPU-only PyTorch. If you want GPU support, install
> the appropriate `torch`/`torchvision` build first (see pytorch.org).

### Step 2 — Run the application

```bash
python main.py
```

The first launch downloads the ImageNet-pretrained SqueezeNet weights
(automatically cached by torchvision).

---

## Usage

The GUI provides three modes:

1. **Identification** — load a handwriting sample; the system compares it
   against all enrolled users and returns **Match** / **Uncertain** /
   **Unknown**, with the confidence score.
2. **Enrollment** — register a new user (ID + handwriting samples). Enrollment
   is validated for quality, minimum sample count, and suspicious similarity to
   existing users.
3. **Management** — view statistics and retrain the SqueezeNet model using
   triplet loss (requires at least 3 users with 2+ samples each).

Both SVM and SqueezeNet pipelines run in parallel, so you can compare the
classical and deep approaches side by side

