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
- [Evaluation Suite](#evaluation-suite)
- [Current Results](#current-results)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Configuration](#configuration)
- [Limitations](#limitations)

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
classical and deep approaches side by side.

---

## Evaluation Suite

`experiments/` adds a rigorous, reproducible protocol around the existing
algorithms — **no core algorithm is modified**. It uses a **seeded synthetic
benchmark** of 24 writers x 8 samples (192 images) generated procedurally, so
ground-truth identity is known by construction.

### Run everything

```bash
python -m experiments.run_experiments --all      # or: run_experiments.bat
```

### Run individual experiments

```bash
python -m experiments.run_experiments --identification
python -m experiments.run_experiments --verification
python -m experiments.run_experiments --robustness
python -m experiments.run_experiments --ablation
python -m experiments.run_experiments --finetune
python -m experiments.run_experiments --summary-only   # rebuild SUMMARY.md from CSVs
```

### What is measured

| Experiment | Protocol |
|---|---|
| **1. Closed-set identification** | Writer-disjoint enrollment/probe splits, 3 seeded repetitions; rank-1..5 accuracy |
| **2. Verification** | Genuine vs impostor score distributions; FAR / FRR curves, EER, operating-point rates |
| **3. Robustness** | Rank-1 under 5 perturbations (blur, noise, rotation, brightness, resolution) x 3 severities, applied *before* preprocessing |
| **4. Preprocessing ablations** | Rank-1 with one pipeline stage removed at a time (denoise, CLAHE, binarize, morph) |
| **5. Transfer learning** | Pretrained vs triplet fine-tuned embeddings vs from-scratch baseline, on held-out and seen writers; embedding-drift analysis |
| **6. Efficiency** | Extraction time/img, feature dimension, template size (raw + encrypted) |

All results are written to `results/` (`SUMMARY.md`, `*.csv`, `plots/*.png`,
`environment.json`).

---

## Current Results

Reference run (seed 42, Python 3.12, CPU-only torch). Figures rounded.

### Closed-Set Identification

| Model | Rank-1 | Rank-5 |
|---|---|---|
| SVM (42-d handcrafted) | 0.9250 | 0.9972 |
| SqueezeNet (512-d learned) | 0.9944 | 1.0000 |

### Verification

| Model | EER | FRR @ FAR=1% |
|---|---|---|
| SVM | 5.10% | 16.67% |
| SqueezeNet | 1.55% | 1.94% |

### Robustness Highlights (rank-1 under severe degradation)

| Perturbation (severe) | SVM | SqueezeNet |
|---|---|---|
| Gaussian noise | 0.85 | 1.00 |
| Brightness gain | 0.94 | 1.00 |
| Rotation 15° | 0.19 | 0.86 |
| Resolution downscale | 0.44 | 0.53 |
| Gaussian blur | 0.23 | 0.35 |

### Preprocessing Ablation (rank-1, SVM only)

| Variant | Rank-1 |
|---|---|
| Full pipeline | 0.9250 |
| No denoise | 0.9250 |
| No CLAHE | 0.9333 |
| No binarize | 0.7250 |
| No morph | 0.9250 |

### Transfer Learning (triplet fine-tuning)

| Protocol | Pretrained | Fine-tuned | From-scratch |
|---|---|---|---|
| Held-out writers (generalization) | 1.0000 | 1.0000 | 1.0000 |
| Fine-tuned writers (sanity) | 0.9750 | 1.0000 | 1.0000 |

Embedding drift (1 - mean cosine similarity, pretrained vs fine-tuned):
`0.370`. Fine-tuning substantially rearranges the embedding space but stays at
the accuracy ceiling — the synthetic benchmark saturates deep embeddings, so
fine-tuning is performance-neutral here.

### Efficiency (CPU)

| Model | Extraction time | Feature dim | Template size |
|---|---|---|---|
| SVM | ~0.031 s/img | 42 | 168 B |
| SqueezeNet | ~0.025 s/img | 512 | 2048 B |

> Full tables and figures: `results/SUMMARY.md` and `results/plots/`.

---

## Reproducibility

- A single master seed (`SEED = 42`) derives all downstream randomness
  (dataset, splits, perturbations, fine-tuning).
- The synthetic writer generator is fully seeded — identical images are
  regenerated on every run.
- `results/environment.json` records the exact Python / library versions and
  hardware at run time.
- Re-running `python -m experiments.run_experiments --all` reproduces the
  reported tables.

---

## Testing

```bash
pytest
```

The unit tests cover preprocessing, feature extraction, decision logic,
database storage, encryption, evaluation metrics and the synthetic generator
(including determinism and writer distinctness).

---

## Configuration

All tunable parameters live in **`config.py`** (system) and
**`experiments/config.py`** (evaluation). Key knobs:

| Parameter | Default | Meaning |
|---|---|---|
| `SVM_THRESHOLD_ACCEPT` / `SVM_THRESHOLD_REJECT` | 0.15 / 0.30 | Cosine-distance accept / reject bands (SVM) |
| `SQUEEZENET_THRESHOLD_ACCEPT` / `..._REJECT` | 0.20 / 0.40 | Same, for deep embeddings |
| `TRIPLET_MARGIN`, `LEARNING_RATE`, `RETRAIN_EPOCHS` | 0.2 / 1e-4 / 10 | Triplet-loss fine-tuning |
| `N_WRITERS`, `SAMPLES_PER_WRITER`, `ENROLLMENT_SAMPLES` | 24 / 8 / 3 | Synthetic benchmark size |
| `N_REPEATS` | 3 | Repeated split repetitions |
| `PERTURBATIONS` | — | Robustness degradation levels |
| `ENCRYPTION_ENABLED` | `True` | Toggle encryption |

---

## Limitations

- **Encryption**: XOR with a fixed key is a simple demonstration-level scheme.
- **No liveness detection**: a photo of handwriting could impersonate a user.
- **Synthetic benchmark**: results measure the *pipeline and methodology*, not
  real-world handwriting. Accuracy on real handwriting will differ.
- **Small scale**: not optimized for thousands of users; thresholds are not
  professionally calibrated.

---

## License

No warranty, express or implied.
