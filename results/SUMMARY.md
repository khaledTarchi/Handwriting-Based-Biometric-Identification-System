# HB-BIS Experimental Results Summary

*Generated automatically by `python -m experiments.run_experiments --all`.*

Master seed: `42` | Python 3.12.10 | Windows-10-10.0.19045-SP0 | torch 2.8.0+cpu

## 1. Closed-Set Identification

Repeated random enrollment/probe splits (writer-disjoint samples); mean ± std over repetitions.

| Model | Rank | Accuracy (mean) | Std |
|---|---|---|---|
| svm | 1 | 0.9250 | 0.0236 |
| svm | 2 | 0.9722 | 0.0039 |
| svm | 3 | 0.9917 | 0.0000 |
| svm | 4 | 0.9944 | 0.0039 |
| svm | 5 | 0.9972 | 0.0039 |
| squeezenet | 1 | 0.9944 | 0.0079 |
| squeezenet | 2 | 1.0000 | 0.0000 |
| squeezenet | 3 | 1.0000 | 0.0000 |
| squeezenet | 4 | 1.0000 | 0.0000 |
| squeezenet | 5 | 1.0000 | 0.0000 |

## 2. Verification (FAR / FRR / EER)

| Model | EER | EER threshold | FRR @ FAR=0.1% | FRR @ FAR=1% | FAR @ cfg threshold | FRR @ cfg threshold |
|---|---|---|---|---|---|---|
| svm | 0.0510 | 0.0045 | 0.51667 | 0.16667 | 0.9614 | 0.0000 |
| squeezenet | 0.0155 | 0.0279 | 0.07222 | 0.01944 | 0.9274 | 0.0000 |

EER per repetition (mean ± std): svm: 0.0509 ± 0.0029; squeezenet: 0.0141 ± 0.0039

## 3. Robustness to Acquisition Degradations

Rank-1 accuracy under perturbations applied before preprocessing.

| Model | Perturbation | Severity | Rank-1 |
|---|---|---|---|
| squeezenet | brightness_gain | mild | 1.0000 |
| squeezenet | brightness_gain | moderate | 1.0000 |
| squeezenet | brightness_gain | severe | 1.0000 |
| squeezenet | gaussian_blur | mild | 0.9917 |
| squeezenet | gaussian_blur | moderate | 0.5000 |
| squeezenet | gaussian_blur | severe | 0.3500 |
| squeezenet | gaussian_noise | mild | 1.0000 |
| squeezenet | gaussian_noise | moderate | 1.0000 |
| squeezenet | gaussian_noise | severe | 1.0000 |
| squeezenet | resolution_scale | mild | 1.0000 |
| squeezenet | resolution_scale | moderate | 0.8500 |
| squeezenet | resolution_scale | severe | 0.5250 |
| squeezenet | rotation | mild | 0.9667 |
| squeezenet | rotation | moderate | 0.9833 |
| squeezenet | rotation | severe | 0.8583 |
| svm | brightness_gain | mild | 0.9583 |
| svm | brightness_gain | moderate | 0.9583 |
| svm | brightness_gain | severe | 0.9417 |
| svm | gaussian_blur | mild | 0.8667 |
| svm | gaussian_blur | moderate | 0.4000 |
| svm | gaussian_blur | severe | 0.2250 |
| svm | gaussian_noise | mild | 0.9333 |
| svm | gaussian_noise | moderate | 0.9083 |
| svm | gaussian_noise | severe | 0.8500 |
| svm | resolution_scale | mild | 0.8417 |
| svm | resolution_scale | moderate | 0.6583 |
| svm | resolution_scale | severe | 0.4417 |
| svm | rotation | mild | 0.3500 |
| svm | rotation | moderate | 0.2333 |
| svm | rotation | severe | 0.1917 |

## 4. Preprocessing Ablations

One preprocessing stage removed at a time (rank-1 accuracy).

| Variant | Model | Rank-1 |
|---|---|---|
| full | squeezenet | 1.0000 |
| full | svm | 0.9250 |
| no_binarize | squeezenet | 1.0000 |
| no_binarize | svm | 0.7250 |
| no_clahe | squeezenet | 1.0000 |
| no_clahe | svm | 0.9333 |
| no_denoise | squeezenet | 0.9917 |
| no_denoise | svm | 0.9250 |
| no_morph | squeezenet | 1.0000 |
| no_morph | svm | 0.9250 |

## 5. Triplet-Loss Fine-Tuning (Before/After Embeddings)

Fine-tuned on 8 writers, evaluated on 16 never-seen writers (generalization) and on probe samples of the fine-tuned writers themselves (sanity check). A from-scratch triplet baseline (no pretrained init, same data and budget) quantifies the value of the pretrained initialization.

| Protocol | Pretrained rank-1 | Fine-tuned rank-1 | From-scratch rank-1 |
|---|---|---|---|
| Held-out writers (generalization) | 1.0000 | 1.0000 | 1.0000 |
| Fine-tuned writers (sanity) | 0.9750 | 1.0000 | 1.0000 |

Embedding drift (1 - mean cosine similarity between pretrained and fine-tuned embeddings of the same held-out images): 0.370054.

## 6. Efficiency

| Model | Extraction time (s/img) | Feature dim | Template size (bytes) | Encrypted size (bytes) |
|---|---|---|---|---|
| svm | 0.031 | 42 | 168 | 224 |
| squeezenet | 0.025 | 512 | 2048 | 2732 |

---

Figures are in `results/plots/`. Raw data in `results/*.csv`.
Environment: `results/environment.json`.
