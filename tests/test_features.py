"""Tests for feature engineering (dimensionality + normalization contracts)."""
import numpy as np
import pytest

from layers.feature_engineering import (
    extract_svm_features, extract_squeezenet_features,
    extract_ink_density, extract_orientation_histogram,
)
from layers.preprocessing import preprocess_for_svm, preprocess_for_squeezenet
from experiments.synthetic_data import SyntheticWriter


def _binary_sample():
    writer = SyntheticWriter("w00", seed=42)
    return preprocess_for_svm(writer.render(sample_seed=3))


@pytest.fixture(scope="module")
def writer_image():
    return SyntheticWriter("w00", seed=42).render(sample_seed=3)


def test_svm_feature_contract(writer_image):
    binary = preprocess_for_svm(writer_image)
    feats = extract_svm_features(binary)
    assert len(feats) == 42
    assert np.all(np.isfinite(feats))
    norm = np.linalg.norm(feats)
    assert np.isclose(norm, 1.0, atol=1e-4) or norm == 0.0


def test_squeezenet_feature_contract(writer_image):
    pre = preprocess_for_squeezenet(writer_image)
    feats = extract_squeezenet_features(pre)
    assert len(feats) == 512
    assert np.all(np.isfinite(feats))
    assert np.isclose(np.linalg.norm(feats), 1.0, atol=1e-4)


def test_blank_image_features():
    blank = np.zeros((224, 224), dtype=np.uint8)
    feats = extract_svm_features(blank)
    assert len(feats) == 42
    assert np.all(np.isfinite(feats))


def test_feature_group_counts(writer_image):
    binary = preprocess_for_svm(writer_image)
    assert len(extract_ink_density(binary)) == 4
    assert len(extract_orientation_histogram(binary)) == 8
