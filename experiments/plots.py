"""
Automatic figure generation for the HB-BIS evaluation framework.
All figures are saved as PNG into results/plots/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


def _save(fig, name):
    path = os.path.join(config.PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rank_curve(rank_means, rank_stds, models, name="rank_curve.png"):
    ranks = np.arange(1, len(rank_means[models[0]]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model in models:
        ax.errorbar(
            ranks, rank_means[model], yerr=rank_stds[model],
            marker="o", capsize=4, label=model,
        )
    ax.set_xlabel("Rank (N)")
    ax.set_ylabel("Identification Accuracy")
    ax.set_title("Closed-Set Identification: Rank-N Accuracy")
    ax.set_xticks(ranks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_roc(curves, name="roc.png"):
    """curves: {model: (far, frr)} where TPR = 1 - FRR."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, (far, frr) in curves.items():
        ax.plot(far, 1.0 - frr, label=model)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 - FRR)")
    ax.set_title("Verification ROC (log-FAR)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_det(curves, name="det.png"):
    """DET curve: FRR vs FAR on log-log axes."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for model, (far, frr) in curves.items():
        ax.plot(far + 1e-6, frr + 1e-6, label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    ax.set_title("Detection Error Tradeoff (DET)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion(matrices, labels, name="confusion_matrix.png"):
    n = len(labels)
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5),
                             squeeze=False)
    for ax, (model, mat) in zip(axes[0], matrices.items()):
        row_norm = mat / (mat.sum(axis=1, keepdims=True) + 1e-9)
        im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(model)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Average Confusion Matrix (Probes)", y=1.02)
    fig.tight_layout()
    return _save(fig, name)


def plot_robustness(rows, name="robustness.png"):
    """
    rows: list of (model, perturbation, level, rank1).
    One subplot per perturbation, bars per severity level, series per model.
    """
    pert_names = sorted({r[1] for r in rows})
    levels = ["mild", "moderate", "severe"]
    models = sorted({r[0] for r in rows})

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    axes = axes.ravel()
    for ax, pname in zip(axes, pert_names):
        width = 0.35
        x = np.arange(len(levels))
        for i, model in enumerate(models):
            vals = []
            for level in levels:
                match = [r[3] for r in rows if r[0] == model and r[1] == pname and r[2] == level]
                vals.append(match[0] if match else np.nan)
            ax.bar(x + i * width, vals, width, label=model)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 1.0)
        ax.set_title(pname)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Robustness to Acquisition Degradations (Rank-1 Accuracy)")
    fig.tight_layout()
    return _save(fig, name)


def plot_ablation(rows, name="ablation.png"):
    """rows: list of (variant, model, rank1). Grouped bars by variant."""
    variants = [r[0] for r in rows if r[1] == rows[0][1]]
    models = sorted({r[1] for r in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(variants))
    for i, model in enumerate(models):
        vals = [r[2] for r in rows if r[1] == model]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variants, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rank-1 Accuracy")
    ax.set_title("Preprocessing Ablation Study")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_finetune(loss_history, bars, name="finetune.png"):
    """
    bars: {stage: rank1} e.g. {"pretrained": .., "fine-tuned": ..}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Triplet Loss")
    ax.set_title("Triplet-Loss Fine-Tuning")
    ax.grid(alpha=0.3)
    ax = axes[1]
    stages = list(bars.keys())
    vals = list(bars.values())
    ax.bar(stages, vals, color=["#3498db", "#9b59b6"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rank-1 Accuracy")
    ax.set_title("Held-Out Writers: Before vs After Fine-Tuning")
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    return _save(fig, name)
