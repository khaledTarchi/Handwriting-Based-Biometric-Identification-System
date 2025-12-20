"""
Experimental configuration for the HB-BIS evaluation framework.

All experiment hyper-parameters (seeds, dataset size, split ratio, number of
repetitions, perturbation levels, ablation variants and fine-tuning settings)
live here so the entire evaluation is reproducible from a single location.
"""

import os

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42                # Master seed; all downstream seeds derive from it
N_REPEATS = 3            # Repeated random enrollment/probe splits

# ---------------------------------------------------------------------------
# Synthetic benchmark dataset
# ---------------------------------------------------------------------------
N_WRITERS = 24                # Number of synthetic writers (identities)
SAMPLES_PER_WRITER = 8        # Handwriting samples per writer
ENROLLMENT_SAMPLES = 3        # Samples reserved for the gallery (enrollment)

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
RANK_MAX = 5                  # Report rank-1 .. rank-N accuracy
N_THRESHOLDS = 300            # Threshold sweep granularity for FAR/FRR curves

# ---------------------------------------------------------------------------
# Robustness perturbations (applied at acquisition, BEFORE preprocessing)
# ---------------------------------------------------------------------------
PERTURBATIONS = {
    "gaussian_blur":       {"levels": {"mild": 1.0, "moderate": 2.0, "severe": 3.0}},
    "gaussian_noise":      {"levels": {"mild": 15.0, "moderate": 30.0, "severe": 50.0}},
    "rotation":            {"levels": {"mild": 5.0, "moderate": 10.0, "severe": 15.0}},
    "brightness_gain":     {"levels": {"mild": 0.8, "moderate": 0.6, "severe": 0.4}},
    "resolution_scale":    {"levels": {"mild": 0.50, "moderate": 0.35, "severe": 0.25}},
}

# ---------------------------------------------------------------------------
# Preprocessing ablation variants (one pipeline stage removed at a time)
# ---------------------------------------------------------------------------
PIPELINE_VARIANTS = [
    "full",          # baseline: denoise + CLAHE + binarize + resize + morph
    "no_denoise",    # remove Gaussian denoising
    "no_clahe",      # remove lighting normalization (CLAHE)
    "no_binarize",   # remove Otsu binarization (keeps grayscale)
    "no_morph",      # remove stroke-width normalization (morphological opening)
]

# ---------------------------------------------------------------------------
# Triplet-loss fine-tuning experiment
# ---------------------------------------------------------------------------
FT_TRAIN_WRITERS = 8          # Writers used to fine-tune (must be < N_WRITERS)
FT_HELDOUT_WRITERS = N_WRITERS - FT_TRAIN_WRITERS
FT_EPOCHS = 5
FT_BATCH_SIZE = 8
FT_LEARNING_RATE = 1e-4
FT_TRIPLET_MARGIN = 0.2
