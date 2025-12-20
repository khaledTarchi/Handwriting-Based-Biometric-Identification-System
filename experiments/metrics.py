"""
Evaluation metrics for biometric identification and verification.

Identification (closed-set, 1:N):
    rank-N accuracy - fraction of probes whose true identity is among the
    top-N closest gallery templates.

Verification (1:1 threshold decision):
    genuine/impostor score distributions, FAR, FRR, EER, ROC and DET curves.

All scores use cosine distance (lower = more similar), identical to the
distance used by the HB-BIS decision layer.
"""

import numpy as np


def cosine_distances_matrix(query_feats, gallery_feats):
    """
    Pairwise cosine distances between two feature matrices.

    Args:
        query_feats:  (M, D)
        gallery_feats:(N, D)

    Returns:
        (M, N) distance matrix (lower = more similar)
    """
    query_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-9)
    gallery_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-9)
    similarities = query_norm @ gallery_norm.T
    return 1.0 - np.clip(similarities, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------
def rank_accuracy(query_ids, gallery_ids, dists, max_rank=5):
    """
    Closed-set identification accuracy at ranks 1..max_rank.

    Args:
        query_ids:   (M,) integer identity codes of probes
        gallery_ids: (N,) integer identity codes of gallery templates
        dists:       (M, N) cosine-distance matrix (lower = better)
        max_rank:    maximum rank to evaluate

    Returns:
        list of accuracy values for ranks 1..max_rank
    """
    order = np.argsort(dists, axis=1)
    ranked = np.asarray(gallery_ids)[order[:, :max_rank]]
    hit = ranked == np.asarray(query_ids)[:, None]
    return [float(hit[:, :k].max(axis=1).mean()) for k in range(1, max_rank + 1)]


def confusion_matrix(query_ids, gallery_ids, dists, n_classes):
    """Un-normalized confusion matrix over probes (rows=true, cols=predicted)."""
    order = np.argsort(dists, axis=1)
    predicted = np.asarray(gallery_ids)[order[:, 0]]
    mat = np.zeros((n_classes, n_classes), dtype=np.float64)
    for true, pred in zip(query_ids, predicted):
        mat[int(true), int(pred)] += 1.0
    return mat


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def gather_genuine_impostor(query_ids, gallery_ids, dists):
    """
    Collect genuine and impostor score distributions from an identification
    probe set.

    Genuine  : distance from a probe to the template of its OWN writer.
    Impostor : distance from a probe to every OTHER writer's template
               (all cross-pairs are counted, giving a large impostor pool).

    Args:
        query_ids:   (M,)
        gallery_ids: (N,)
        dists:       (M, N)

    Returns:
        (genuine, impostor) numpy arrays of cosine distances
    """
    index_of = {gid: i for i, gid in enumerate(gallery_ids)}
    genuine, impostor = [], []
    for qi, row in zip(query_ids, dists):
        own = index_of[int(qi)]
        genuine.append(row[own])
        for gi, dist in enumerate(row):
            if gi != own:
                impostor.append(dist)
    return np.asarray(genuine), np.asarray(impostor)


def verification_curves(genuine, impostor, n_thresholds=300):
    """
    Sweep the accept threshold over cosine distance and compute FAR/FRR.

    Decision rule (matches HB-BIS): accept if distance <= threshold.

    Returns:
        (thresholds, far, frr) numpy arrays
    """
    all_scores = np.concatenate([genuine, impostor])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    far = np.array([np.mean(impostor <= t) for t in thresholds])
    frr = np.array([np.mean(genuine > t) for t in thresholds])
    return thresholds, far, frr


def eer_from_curves(thresholds, far, frr):
    """
    Equal Error Rate: the operating point where FAR == FRR.

    Returns (eer, threshold_at_eer).
    """
    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))
    eer = 0.5 * (far[idx] + frr[idx])
    return float(eer), float(thresholds[idx])


def rate_at(target, x_axis, y_axis):
    """Value of y_axis at the point where x_axis is closest to ``target``."""
    idx = int(np.argmin(np.abs(x_axis - target)))
    return float(y_axis[idx])
