"""Tests for the decision & similarity layer."""
import numpy as np
import pytest

from layers.decision import (
    cosine_distance, euclidean_distance, normalized_euclidean_distance,
    make_decision,
)


def test_cosine_distance_properties():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_distance(v, v) == pytest.approx(0.0)
    assert cosine_distance(v, -v) == pytest.approx(2.0)
    assert cosine_distance(v, np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    # scaled vector has same direction -> distance 0
    assert cosine_distance(v, 5.0 * v) == pytest.approx(0.0)


def test_cosine_distance_zero_vector():
    v = np.zeros(3)
    assert cosine_distance(v, np.ones(3)) == pytest.approx(2.0)


def test_euclidean_distances():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)
    assert normalized_euclidean_distance(a, b) == pytest.approx(5.0 / np.sqrt(2))


def test_make_decision_zones():
    # accept zone (distance below accept threshold)
    decision, confidence = make_decision(0.05, 0.15, 0.30)
    assert decision == "MATCH"
    assert 0 < confidence <= 100

    # uncertain zone
    decision, confidence = make_decision(0.20, 0.15, 0.30)
    assert decision == "UNCERTAIN"

    # reject zone
    decision, confidence = make_decision(0.50, 0.15, 0.30)
    assert decision == "UNKNOWN"


def test_make_decision_boundaries():
    assert make_decision(0.15, 0.15, 0.30)[0] == "UNCERTAIN"
    assert make_decision(0.30, 0.15, 0.30)[0] == "UNKNOWN"
    assert make_decision(0.0, 0.15, 0.30)[0] == "MATCH"
