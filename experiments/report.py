"""
Result serialization: CSVs, JSON environment capture and the Markdown summary.
"""

import csv
import json
import os
import platform
import subprocess

from . import config


def write_csv(path, header, rows):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_text(path, text):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def capture_environment():
    """Capture hardware + software versions for full reproducibility."""
    import numpy, scipy, cv2, torch, torchvision, sklearn, skimage, PIL

    info = {
        "seed": config.SEED,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_device": str(torch.device("cpu")),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "scikit_learn": sklearn.__version__,
        "scikit_image": skimage.__version__,
        "pillow": PIL.__version__,
    }
    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        pass
    return info


def write_environment(info):
    path = os.path.join(config.RESULTS_DIR, "environment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def _fmt(value, precision=4):
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_summary(identification, verification, robustness_rows, ablation_rows,
                  finetune, efficiency, env):
    lines = []
    lines.append("# HB-BIS Experimental Results Summary")
    lines.append("")
    lines.append(f"*Generated automatically by `python -m experiments.run_experiments --all`.*")
    lines.append("")
    lines.append(f"Master seed: `{env['seed']}` | Python {env['python']} | "
                 f"{env['platform']} | torch {env['torch']}")
    lines.append("")

    # ---- Identification
    lines.append("## 1. Closed-Set Identification")
    lines.append("")
    lines.append("Repeated random enrollment/probe splits (writer-disjoint samples); "
                 "mean ± std over repetitions.")
    lines.append("")
    ident_rows = []
    for model, data in identification.items():
        for rank, (mean, std) in enumerate(zip(data["mean"], data["std"]), start=1):
            ident_rows.append([model, rank, _fmt(mean), _fmt(std)])
    lines.append(md_table(["Model", "Rank", "Accuracy (mean)", "Std"], ident_rows))
    lines.append("")

    # ---- Verification
    lines.append("## 2. Verification (FAR / FRR / EER)")
    lines.append("")
    veri_rows = []
    for model, v in verification.items():
        veri_rows.append([
            model,
            _fmt(v["eer"]),
            _fmt(v["eer_threshold"]),
            _fmt(v["frr_at_far_0.1%"], 5),
            _fmt(v["frr_at_far_1%"], 5),
            _fmt(v["operating_far"]),
            _fmt(v["operating_frr"]),
        ])
    lines.append(md_table(
        ["Model", "EER", "EER threshold", "FRR @ FAR=0.1%", "FRR @ FAR=1%",
         "FAR @ cfg threshold", "FRR @ cfg threshold"],
        veri_rows))
    lines.append("")
    lines.append("EER per repetition (mean ± std): "
                 + "; ".join(f"{m}: {_fmt(v['eer_mean'])} ± {_fmt(v['eer_std'])}"
                             for m, v in verification.items()))
    lines.append("")

    # ---- Robustness
    lines.append("## 3. Robustness to Acquisition Degradations")
    lines.append("")
    lines.append("Rank-1 accuracy under perturbations applied before preprocessing.")
    lines.append("")
    rob_rows = []
    for model, pname, level, r1 in sorted(robustness_rows):
        rob_rows.append([model, pname, level, _fmt(r1)])
    lines.append(md_table(["Model", "Perturbation", "Severity", "Rank-1"], rob_rows))
    lines.append("")

    # ---- Ablation
    lines.append("## 4. Preprocessing Ablations")
    lines.append("")
    lines.append("One preprocessing stage removed at a time (rank-1 accuracy).")
    lines.append("")
    ab_rows = []
    for variant, model, r1 in sorted(ablation_rows):
        ab_rows.append([variant, model, _fmt(r1)])
    lines.append(md_table(["Variant", "Model", "Rank-1"], ab_rows))
    lines.append("")

    # ---- Fine-tuning
    if finetune:
        lines.append("## 5. Triplet-Loss Fine-Tuning (Before/After Embeddings)")
        lines.append("")
        lines.append("Fine-tuned on %d writers, evaluated on %d never-seen writers "
                     "(generalization) and on probe samples of the fine-tuned "
                     "writers themselves (sanity check). A from-scratch triplet "
                     "baseline (no pretrained init, same data and budget) quantifies "
                     "the value of the pretrained initialization."
                     % (finetune["train_writers"], finetune["heldout_writers"]))
        lines.append("")
        r1 = finetune["rank1"]
        lines.append(md_table(
            ["Protocol", "Pretrained rank-1", "Fine-tuned rank-1", "From-scratch rank-1"],
            [["Held-out writers (generalization)", _fmt(r1["heldout_pretrained"]),
              _fmt(r1["heldout_finetuned"]), _fmt(r1["heldout_scratch"])],
             ["Fine-tuned writers (sanity)", _fmt(r1["seen_pretrained"]),
              _fmt(r1["seen_finetuned"]), _fmt(r1["seen_scratch"])]]))
        lines.append("")
        lines.append("Embedding drift (1 - mean cosine similarity between "
                     "pretrained and fine-tuned embeddings of the same held-out "
                     "images): %s." % _fmt(finetune["embedding_drift"], 6))
        lines.append("")

    # ---- Efficiency
    lines.append("## 6. Efficiency")
    lines.append("")
    eff_rows = []
    for row in efficiency:
        eff_rows.append([row[0], _fmt(row[1], 3), row[2], row[3], row[4]])
    lines.append(md_table(["Model", "Extraction time (s/img)", "Feature dim",
                           "Template size (bytes)", "Encrypted size (bytes)"], eff_rows))
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("Figures are in `results/plots/`. Raw data in `results/*.csv`.")
    lines.append("Environment: `results/environment.json`.")
    lines.append("")
    return "\n".join(lines)
