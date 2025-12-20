"""Tests for the evaluation metrics (identification + verification)."""
import numpy as np
import pytest

from experiments.metrics import (
    cosine_distances_matrix, rank_accuracy, confusion_matrix,
    gather_genuine_impostor, verification_curves, eer_from_curves, rate_at,
)
from experiments.dataset import sample_splits


def _toy_data():
    """4 writers x 3 samples, perfectly separable unit vectors."""
    rng = np.random.RandomState(0)
    feats, labels = [], []
    for writer in range(4):
        base = rng.randn(16)
        base /= np.linalg.norm(base)
        for s in range(3):
            noise = 0.05 * rng.randn(16)
            v = base + noise
            v /= np.linalg.norm(v)
            feats.append(v)
            labels.append(writer)
    return np.stack(feats), np.array(labels)


def test_cosine_distances_matrix_symmetric_range():
    feats, _ = _toy_data()
    dists = cosine_distances_matrix(feats, feats)
    assert dists.shape == (len(feats), len(feats))
    assert dists.min() >= 0.0 and dists.max() <= 2.0
    assert np.allclose(dists, dists.T)


def test_rank_accuracy_perfect():
    feats, labels = _toy_data()
    enroll = [0, 3, 6, 9]
    probe = [i for i in range(12) if i not in enroll]
    templates = np.stack([feats[labels == c].mean(axis=0) for c in sorted(set(labels))])
    dists = cosine_distances_matrix(feats[probe], templates)
    acc = rank_accuracy(labels[probe], np.array(sorted(set(labels))), dists, max_rank=3)
    assert acc[0] == pytest.approx(1.0)  # perfectly separable
    assert len(acc) == 3


def test_confusion_matrix_shape():
    feats, labels = _toy_data()
    templates = np.stack([feats[labels == c].mean(axis=0) for c in sorted(set(labels))])
    dists = cosine_distances_matrix(feats, templates)
    mat = confusion_matrix(labels, np.array(sorted(set(labels))), dists, n_classes=4)
    assert mat.shape == (4, 4)
    assert np.sum(mat) == len(feats)


def test_genuine_impostor():
    feats, labels = _toy_data()
    templates = np.stack([feats[labels == c].mean(axis=0) for c in sorted(set(labels))])
    dists = cosine_distances_matrix(feats, templates)
    g, i = gather_genuine_impostor(labels, np.array(sorted(set(labels))), dists)
    assert len(g) == len(feats)
    assert len(i) == len(feats) * 3
    assert g.mean() < i.mean()


def test_verification_curves_and_eer():
    genuine = np.random.RandomState(0).normal(0.1, 0.05, 500)
    impostor = np.random.RandomState(1).normal(0.5, 0.1, 1000)
    th, far, frr = verification_curves(genuine, impostor, n_thresholds=200)
    assert len(th) == 200
    assert 0 <= far.min() and far.max() <= 1
    assert 0 <= frr.min() and frr.max() <= 1
    eer, _ = eer_from_curves(th, far, frr)
    assert 0 < eer < 0.5


def test_rate_at():
    x = np.array([0.001, 0.01, 0.1])
    y = np.array([0.2, 0.4, 0.8])
    assert rate_at(0.01, x, y) == pytest.approx(0.4)


def test_sample_splits_disjoint():
    images = [(f"w{i % 3:02d}", None) for i in range(9)]
    enroll, probe = sample_splits(images, n_enroll=2, repeat_seed=1)
    assert set(enroll).isdisjoint(probe)
    assert len(enroll) == 6
    assert len(probe) == 3
