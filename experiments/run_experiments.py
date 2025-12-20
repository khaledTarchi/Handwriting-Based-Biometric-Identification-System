"""
HB-BIS experiment runner
========================
One command to reproduce the entire evaluation:

    python -m experiments.run_experiments --all
    python -m experiments.run_experiments --identification
    python -m experiments.run_experiments --robustness
    python -m experiments.run_experiments --ablation
    python -m experiments.run_experiments --finetune

All results (CSV tables, Markdown summary, PNG figures, environment) are
written to results/. The experiments only add a rigorous protocol around the
existing HB-BIS algorithms; no core algorithm is modified.
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from experiments import config
from experiments.dataset import build_benchmark, sample_splits
from experiments.features import FeaturePipeline
from experiments import metrics, robustness, ablations, plots, report
from experiments.finetune import train_triplet


MODELS = ["svm", "squeezenet"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Shared split machinery
# ---------------------------------------------------------------------------
def identity_codes(images):
    writer_ids = [wid for wid, _ in images]
    code_map = {wid: i for i, wid in enumerate(sorted(set(writer_ids)))}
    return np.array([code_map[wid] for wid in writer_ids]), code_map


def build_templates(gallery_feats, gallery_ids):
    """Average enrollment features per writer into one template per identity.

    Args:
        gallery_feats: (E, D) features of the enrollment samples only
        gallery_ids:   (E,) writer codes aligned with gallery_feats
    """
    unique = np.unique(gallery_ids)
    templates = np.stack([gallery_feats[gallery_ids == c].mean(axis=0) for c in unique])
    return templates, unique


# ---------------------------------------------------------------------------
# Experiment 1: closed-set identification
# ---------------------------------------------------------------------------
def run_identification(pipelines, images, clean_feats, cfg):
    print("\n[1/6] Closed-set identification (rank-N, %d repeats)..." % cfg.N_REPEATS)
    ids_codes, code_map = identity_codes(images)
    n_writers = len(code_map)
    writer_labels = [code_map[wid] for wid in sorted(code_map, key=lambda w: code_map[w])]

    results = {}
    for model, pipe in pipelines.items():
        feats = clean_feats[model]
        per_repeat = []
        cms = []
        for r in range(cfg.N_REPEATS):
            enroll, probe = sample_splits(images, cfg.ENROLLMENT_SAMPLES,
                                          cfg.SEED * 7 + r)
            gallery_ids = ids_codes[enroll]
            query_ids = ids_codes[probe]
            templates, unique = build_templates(feats[enroll], gallery_ids)
            dists = metrics.cosine_distances_matrix(feats[probe], templates)
            per_repeat.append(metrics.rank_accuracy(query_ids, unique, dists, cfg.RANK_MAX))
            cms.append(metrics.confusion_matrix(query_ids, unique, dists, n_writers))
        arr = np.asarray(per_repeat)
        results[model] = {
            "mean": arr.mean(axis=0),
            "std": arr.std(axis=0),
            "confusion": np.mean(cms, axis=0),
        }
        print(f"   {model}: rank-1 = {results[model]['mean'][0]:.4f} ± {results[model]['std'][0]:.4f}")

    # CSV
    rows = []
    for model, data in results.items():
        for rank in range(1, cfg.RANK_MAX + 1):
            rows.append([model, rank, round(data["mean"][rank - 1], 4),
                         round(data["std"][rank - 1], 4)])
    report.write_csv(os.path.join(config.RESULTS_DIR, "identification.csv"),
                     ["model", "rank", "accuracy_mean", "accuracy_std"], rows)

    plots.plot_rank_curve(
        {m: d["mean"] for m, d in results.items()},
        {m: d["std"] for m, d in results.items()},
        list(results.keys()),
    )
    plots.plot_confusion(
        {m: d["confusion"] for m, d in results.items()},
        [f"w{c:02d}" for c in writer_labels],
    )
    return results


# ---------------------------------------------------------------------------
# Experiment 2: verification (FAR / FRR / EER)
# ---------------------------------------------------------------------------
def run_verification(pipelines, images, clean_feats, cfg):
    print("\n[2/6] Verification (FAR/FRR/EER)...")
    ids_codes, _ = identity_codes(images)

    results = {}
    curves = {}
    for model, pipe in pipelines.items():
        feats = clean_feats[model]
        pooled_g, pooled_i = [], []
        per_repeat_eer = []
        for r in range(cfg.N_REPEATS):
            enroll, probe = sample_splits(images, cfg.ENROLLMENT_SAMPLES,
                                          cfg.SEED * 7 + r)
            gallery_ids = ids_codes[enroll]
            query_ids = ids_codes[probe]
            templates, unique = build_templates(feats[enroll], gallery_ids)
            dists = metrics.cosine_distances_matrix(feats[probe], templates)
            g, i = metrics.gather_genuine_impostor(query_ids, unique, dists)
            pooled_g.append(g)
            pooled_i.append(i)
            th, far, frr = metrics.verification_curves(g, i, cfg.N_THRESHOLDS)
            per_repeat_eer.append(metrics.eer_from_curves(th, far, frr)[0])

        g_all = np.concatenate(pooled_g)
        i_all = np.concatenate(pooled_i)
        th, far, frr = metrics.verification_curves(g_all, i_all, cfg.N_THRESHOLDS)
        eer, eer_th = metrics.eer_from_curves(th, far, frr)

        # Operating point at the thresholds hard-coded in config.py
        accept_threshold = 0.15 if model == "svm" else 0.20
        operating_far = float(np.mean(i_all <= accept_threshold))
        operating_frr = float(np.mean(g_all > accept_threshold))

        results[model] = {
            "eer": eer,
            "eer_threshold": eer_th,
            "eer_mean": float(np.mean(per_repeat_eer)),
            "eer_std": float(np.std(per_repeat_eer)),
            "frr_at_far_0.1%": metrics.rate_at(0.001, far, frr),
            "frr_at_far_1%": metrics.rate_at(0.01, far, frr),
            "operating_far": operating_far,
            "operating_frr": operating_frr,
            "n_genuine": int(len(g_all)),
            "n_impostor": int(len(i_all)),
        }
        curves[model] = (far, frr)
        print(f"   {model}: EER = {eer:.4f} (genuine={len(g_all)}, impostor={len(i_all)})")

    rows = []
    for model, v in results.items():
        rows.append([model, v["eer"], v["eer_threshold"], v["frr_at_far_0.1%"],
                     v["frr_at_far_1%"], v["operating_far"], v["operating_frr"],
                     v["eer_mean"], v["eer_std"], v["n_genuine"], v["n_impostor"]])
    report.write_csv(os.path.join(config.RESULTS_DIR, "verification.csv"),
                     ["model", "eer", "eer_threshold", "frr@far0.1%", "frr@far1%",
                      "far@cfg_threshold", "frr@cfg_threshold",
                      "eer_mean", "eer_std", "n_genuine", "n_impostor"], rows)

    plots.plot_roc(curves)
    plots.plot_det(curves)
    return results, curves


# ---------------------------------------------------------------------------
# Experiment 3: robustness to acquisition degradations
# ---------------------------------------------------------------------------
def run_robustness(pipelines, images, clean_feats, cfg):
    print("\n[3/6] Robustness to acquisition degradations...")
    ids_codes, _ = identity_codes(images)
    enroll, probe = sample_splits(images, cfg.ENROLLMENT_SAMPLES, cfg.SEED * 13)
    gallery_ids = ids_codes[enroll]
    query_ids = ids_codes[probe]
    unique = np.unique(gallery_ids)

    rows = []
    for model, pipe in pipelines.items():
        templates, _ = build_templates(clean_feats[model][enroll], gallery_ids)
        for pname, spec in cfg.PERTURBATIONS.items():
            for level, parameter in spec["levels"].items():
                perturbed = [
                    robustness.apply(pname, images[idx][1], parameter, seed=cfg.SEED + idx)
                    for idx in probe
                ]
                feats = pipe.extract(perturbed)
                dists = metrics.cosine_distances_matrix(feats, templates)
                rank1 = metrics.rank_accuracy(query_ids, unique, dists, 1)[0]
                rows.append([model, pname, level, round(rank1, 4)])
        print(f"   {model}: done ({len(cfg.PERTURBATIONS)} perturbations x 3 levels)")

    report.write_csv(os.path.join(config.RESULTS_DIR, "robustness.csv"),
                     ["model", "perturbation", "level", "rank1"], rows)
    plots.plot_robustness(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment 4: preprocessing ablations
# ---------------------------------------------------------------------------
def run_ablation(images, cfg):
    print("\n[4/6] Preprocessing ablations...")
    ids_codes, _ = identity_codes(images)
    enroll, probe = sample_splits(images, cfg.ENROLLMENT_SAMPLES, cfg.SEED * 17)
    gallery_ids = ids_codes[enroll]
    query_ids = ids_codes[probe]
    unique = np.unique(gallery_ids)

    rows = []
    for variant in cfg.PIPELINE_VARIANTS:
        for model in MODELS:
            feats = np.stack([
                ablations.extract_variant_features(images[idx][1], variant, model)
                for idx in range(len(images))
            ])
            templates, _ = build_templates(feats[enroll], gallery_ids)
            dists = metrics.cosine_distances_matrix(feats[probe], templates)
            rank1 = metrics.rank_accuracy(query_ids, unique, dists, 1)[0]
            rows.append([variant, model, round(rank1, 4)])
        print(f"   variant '{variant}' done")

    report.write_csv(os.path.join(config.RESULTS_DIR, "ablation.csv"),
                     ["variant", "model", "rank1"], rows)
    plots.plot_ablation(rows)
    return rows


# ---------------------------------------------------------------------------
# Experiment 5: triplet-loss fine-tuning (before/after embeddings)
# ---------------------------------------------------------------------------
def evaluate_heldout(pipe, heldout_images, cfg):
    ids_codes, code_map = identity_codes(heldout_images)
    enroll, probe = sample_splits(heldout_images, cfg.ENROLLMENT_SAMPLES, cfg.SEED * 23)
    gallery_ids = ids_codes[enroll]
    query_ids = ids_codes[probe]
    feats = pipe.extract([im for _, im in heldout_images])
    templates, unique = build_templates(feats[enroll], gallery_ids)
    dists = metrics.cosine_distances_matrix(feats[probe], templates)
    return metrics.rank_accuracy(query_ids, unique, dists, 1)[0]


def run_finetune(images, cfg):
    print("\n[5/6] Triplet-loss fine-tuning (before/after embeddings)...")
    writer_ids = sorted(set(w for w, _ in images))
    train_ids = writer_ids[:cfg.FT_TRAIN_WRITERS]
    heldout_ids = writer_ids[cfg.FT_TRAIN_WRITERS:]
    train_images = [(wid, im) for wid, im in images if wid in set(train_ids)]
    heldout_images = [(wid, im) for wid, im in images if wid in set(heldout_ids)]

    print(f"   fine-tuning writers: {train_ids}")
    print(f"   held-out writers: {heldout_ids}")

    # --- Stage 1: pretrained embeddings (before fine-tuning)
    base_pipe = FeaturePipeline("squeezenet", weights="pretrained")
    heldout_rank1_base = evaluate_heldout(base_pipe, heldout_images, cfg)
    train_rank1_base = evaluate_heldout(base_pipe, train_images, cfg)
    print(f"   pretrained embeddings: held-out rank-1 = {heldout_rank1_base:.4f}, "
          f"seen-writers rank-1 = {train_rank1_base:.4f}")

    # --- Stage 2: fine-tune on enrollment samples of the train writers.
    #     train_triplet mutates the passed model in place, so give it a deep
    #     copy; base_pipe keeps the untouched pretrained weights for a fair
    #     before/after comparison.
    train_writers = [(wid, [im for w, im in images if w == wid]) for wid in train_ids]
    set_seed(cfg.SEED + 1)
    model_ft, loss_history = train_triplet(
        train_writers,
        epochs=cfg.FT_EPOCHS,
        batch_size=cfg.FT_BATCH_SIZE,
        learning_rate=cfg.FT_LEARNING_RATE,
        margin=cfg.FT_TRIPLET_MARGIN,
        base_model=copy.deepcopy(base_pipe._model),
        verbose=True,
    )

    # --- Stage 3: fine-tuned embeddings (after fine-tuning)
    ft_pipe = FeaturePipeline("squeezenet", model=model_ft)
    heldout_rank1_ft = evaluate_heldout(ft_pipe, heldout_images, cfg)
    train_rank1_ft = evaluate_heldout(ft_pipe, train_images, cfg)
    print(f"   fine-tuned embeddings: held-out rank-1 = {heldout_rank1_ft:.4f}, "
          f"seen-writers rank-1 = {train_rank1_ft:.4f}")

    # --- Embedding drift: verify the fine-tuning actually changed the space.
    #     (1 - mean cosine similarity) between pretrained and fine-tuned
    #     embeddings of the SAME held-out images. A small but non-zero drift
    #     confirms fine-tuning had an effect while preserving the pretrained
    #     representation.
    feats_base = base_pipe.extract([im for _, im in heldout_images])
    feats_ft = ft_pipe.extract([im for _, im in heldout_images])
    mean_cosine = float(np.mean((feats_base * feats_ft).sum(axis=1)))
    print(f"   embedding mean cosine similarity: {mean_cosine:.6f} "
          f"(drift = {1.0 - mean_cosine:.6f})")

    # --- Baseline: triplet training from a random initialization (no
    #     pretrained weights) on the same data and budget. This quantifies how
    #     much of the discriminative power comes from the pretrained init.
    print("   from-scratch triplet baseline (no pretrained init)...")
    set_seed(cfg.SEED + 2)
    model_scratch, _ = train_triplet(
        train_writers,
        epochs=cfg.FT_EPOCHS,
        batch_size=cfg.FT_BATCH_SIZE,
        learning_rate=cfg.FT_LEARNING_RATE,
        margin=cfg.FT_TRIPLET_MARGIN,
        base_model=None,
        verbose=False,
    )
    scratch_pipe = FeaturePipeline("squeezenet", model=model_scratch)
    heldout_rank1_scratch = evaluate_heldout(scratch_pipe, heldout_images, cfg)
    train_rank1_scratch = evaluate_heldout(scratch_pipe, train_images, cfg)
    print(f"   from-scratch: held-out rank-1 = {heldout_rank1_scratch:.4f}, "
          f"seen-writers rank-1 = {train_rank1_scratch:.4f}")

    rows = [
        ["pretrained", "heldout", round(heldout_rank1_base, 4)],
        ["finetuned", "heldout", round(heldout_rank1_ft, 4)],
        ["scratch", "heldout", round(heldout_rank1_scratch, 4)],
        ["pretrained", "seen_writers", round(train_rank1_base, 4)],
        ["finetuned", "seen_writers", round(train_rank1_ft, 4)],
        ["scratch", "seen_writers", round(train_rank1_scratch, 4)],
        ["meta", "embedding_drift", round(1.0 - mean_cosine, 6)],
    ]
    report.write_csv(os.path.join(config.RESULTS_DIR, "finetune.csv"),
                     ["stage", "protocol", "rank1"], rows)
    plots.plot_finetune(loss_history, {
        "pretrained": heldout_rank1_base,
        "fine-tuned": heldout_rank1_ft,
        "from-scratch": heldout_rank1_scratch,
    })

    return {
        "train_writers": len(train_ids),
        "heldout_writers": len(heldout_ids),
        "rank1": {
            "heldout_pretrained": heldout_rank1_base,
            "heldout_finetuned": heldout_rank1_ft,
            "heldout_scratch": heldout_rank1_scratch,
            "seen_pretrained": train_rank1_base,
            "seen_finetuned": train_rank1_ft,
            "seen_scratch": train_rank1_scratch,
        },
        "embedding_mean_cosine": mean_cosine,
        "embedding_drift": 1.0 - mean_cosine,
        "loss_history": loss_history,
    }


# ---------------------------------------------------------------------------
# Experiment 6: efficiency
# ---------------------------------------------------------------------------
def run_efficiency(pipelines, images, cfg):
    print("\n[6/6] Efficiency measurements...")
    rows = []
    sample = [im for _, im in images[:8]]
    for model, pipe in pipelines.items():
        t0 = time.perf_counter()
        feats = pipe.extract(sample)
        elapsed = (time.perf_counter() - t0) / len(sample)
        feat_dim = int(feats.shape[1])
        template_bytes = feat_dim * 4  # float32
        encrypted_bytes = int(np.ceil(template_bytes / 3) * 4)  # base64 overhead
        params = pipe.parameter_count()
        rows.append([model, round(elapsed, 4), feat_dim, template_bytes,
                     encrypted_bytes, params if params else "-"])
        print(f"   {model}: {elapsed:.3f}s/img, dim={feat_dim}, template={template_bytes}B")

    report.write_csv(os.path.join(config.RESULTS_DIR, "efficiency.csv"),
                     ["model", "seconds_per_image", "feature_dim", "template_bytes",
                      "encrypted_bytes", "parameters"], rows)
    return rows


# ---------------------------------------------------------------------------
# Summary rebuild from existing CSVs (no experiment re-run)
# ---------------------------------------------------------------------------
def read_csv(path):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def rebuild_summary_from_csvs():
    """Rebuild SUMMARY.md purely from results/*.csv (e.g. after a partial run)."""
    ident = {}
    for r in read_csv(os.path.join(config.RESULTS_DIR, "identification.csv")):
        ident.setdefault(r["model"], {"mean": [], "std": []})
        ident[r["model"]]["mean"].append(float(r["accuracy_mean"]))
        ident[r["model"]]["std"].append(float(r["accuracy_std"]))

    verification = {}
    for r in read_csv(os.path.join(config.RESULTS_DIR, "verification.csv")):
        verification[r["model"]] = {
            "eer": float(r["eer"]),
            "eer_threshold": float(r["eer_threshold"]),
            "frr_at_far_0.1%": float(r["frr@far0.1%"]),
            "frr_at_far_1%": float(r["frr@far1%"]),
            "operating_far": float(r["far@cfg_threshold"]),
            "operating_frr": float(r["frr@cfg_threshold"]),
            "eer_mean": float(r["eer_mean"]),
            "eer_std": float(r["eer_std"]),
        }

    robustness_rows = [(r["model"], r["perturbation"], r["level"], float(r["rank1"]))
                       for r in read_csv(os.path.join(config.RESULTS_DIR, "robustness.csv"))]
    ablation_rows = [(r["variant"], r["model"], float(r["rank1"]))
                     for r in read_csv(os.path.join(config.RESULTS_DIR, "ablation.csv"))]

    ft_rows = list(read_csv(os.path.join(config.RESULTS_DIR, "finetune.csv")))
    finetune = None
    if ft_rows:
        rank1 = {}
        drift = 0.0
        for r in ft_rows:
            if r["protocol"] == "embedding_drift":
                drift = float(r["rank1"])
            else:
                rank1[f"{r['protocol']}_{r['stage']}"] = float(r["rank1"])
        finetune = {
            "train_writers": config.FT_TRAIN_WRITERS,
            "heldout_writers": config.N_WRITERS - config.FT_TRAIN_WRITERS,
            "rank1": {
                "heldout_pretrained": rank1.get("heldout_pretrained", 0.0),
                "heldout_finetuned": rank1.get("heldout_finetuned", 0.0),
                "heldout_scratch": rank1.get("heldout_scratch", 0.0),
                "seen_pretrained": rank1.get("seen_writers_pretrained", 0.0),
                "seen_finetuned": rank1.get("seen_writers_finetuned", 0.0),
                "seen_scratch": rank1.get("seen_writers_scratch", 0.0),
            },
            "embedding_drift": drift,
        }

    efficiency = []
    for r in read_csv(os.path.join(config.RESULTS_DIR, "efficiency.csv")):
        efficiency.append([r["model"], float(r["seconds_per_image"]),
                           int(r["feature_dim"]), int(r["template_bytes"]),
                           int(r["encrypted_bytes"])])

    env = json.load(open(os.path.join(config.RESULTS_DIR, "environment.json"), encoding="utf-8"))
    summary = report.build_summary(ident, verification, robustness_rows,
                                   ablation_rows, finetune, efficiency, env)
    report.write_text(os.path.join(config.RESULTS_DIR, "SUMMARY.md"), summary)
    print("SUMMARY.md rebuilt from existing results/*.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HB-BIS evaluation suite")
    parser.add_argument("--all", action="store_true", help="run all experiments")
    parser.add_argument("--identification", action="store_true")
    parser.add_argument("--verification", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild SUMMARY.md from existing results/*.csv")
    parser.add_argument("--no-pretrain", action="store_true",
                        help="skip the clean-feature precomputation step")
    args = parser.parse_args()

    if args.summary_only:
        rebuild_summary_from_csvs()
        return

    if not (args.all or args.identification or args.verification or
            args.robustness or args.ablation or args.finetune):
        parser.print_help()
        return

    cfg = config
    set_seed(cfg.SEED)

    print("=" * 70)
    print("HB-BIS Experimental Evaluation")
    print(f"seed={cfg.SEED}  writers={cfg.N_WRITERS}  samples/writer={cfg.SAMPLES_PER_WRITER}"
          f"  enrollment={cfg.ENROLLMENT_SAMPLES}  repeats={cfg.N_REPEATS}")
    print("=" * 70)

    env = report.capture_environment()
    report.write_environment(env)
    print(f"Environment captured -> results/environment.json")

    writers, images = build_benchmark()
    print(f"Synthetic benchmark: {len(images)} images, {len(writers)} writers")

    pipelines = {
        "svm": FeaturePipeline("svm"),
        "squeezenet": FeaturePipeline("squeezenet", weights="pretrained"),
    }

    # Precompute clean features once (reused by identification & verification)
    clean_feats = {}
    if not args.no_pretrain:
        for model, pipe in pipelines.items():
            t0 = time.perf_counter()
            clean_feats[model] = pipe.extract([im for _, im in images])
            print(f"Precomputed {model} features for {len(images)} images "
                  f"in {time.perf_counter() - t0:.1f}s")

    identification = verification = curves = robustness_rows = ablation_rows = None
    finetune = efficiency = None

    if args.all or args.identification:
        identification = run_identification(pipelines, images, clean_feats, cfg)
    if args.all or args.verification:
        verification, curves = run_verification(pipelines, images, clean_feats, cfg)
    if args.all or args.robustness:
        robustness_rows = run_robustness(pipelines, images, clean_feats, cfg)
    if args.all or args.ablation:
        ablation_rows = run_ablation(images, cfg)
    if args.all or args.finetune:
        finetune = run_finetune(images, cfg)
    if args.all:
        efficiency = run_efficiency(pipelines, images, cfg)

    # Summary (only written when the experiments it references have run)
    if args.all:
        summary = report.build_summary(
            identification, verification, robustness_rows, ablation_rows,
            finetune, efficiency, env)
        report.write_text(os.path.join(config.RESULTS_DIR, "SUMMARY.md"), summary)
        print("\nResults written to results/ (SUMMARY.md, *.csv, plots/, environment.json)")


if __name__ == "__main__":
    main()
