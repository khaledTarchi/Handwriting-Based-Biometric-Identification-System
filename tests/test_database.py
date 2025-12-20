"""Tests for the biometric database layer using an isolated temporary directory."""
import numpy as np
from PIL import Image

import layers.database as database


def _sample_image():
    return Image.new("RGB", (200, 200), (255, 255, 255))


def _enroll_helper(tmp_path, monkeypatch, n_samples=2):
    monkeypatch.setattr(database, "DATABASE_ROOT", str(tmp_path))
    images = [_sample_image() for _ in range(n_samples)]
    svm_feats = [np.random.randn(42).astype(np.float32) for _ in range(n_samples)]
    sqz_feats = [np.random.randn(512).astype(np.float32) for _ in range(n_samples)]
    success, user_id = database.enroll_user("Test_User", images, svm_feats, sqz_feats)
    assert success
    return user_id


def test_enroll_and_metadata(tmp_path, monkeypatch):
    user_id = _enroll_helper(tmp_path, monkeypatch)
    users = database.get_all_users()
    assert len(users) == 1
    assert users[0]["user_id"] == user_id
    assert users[0]["name"] == "Test_User"
    assert users[0]["num_samples"] == 2


def test_feature_roundtrip_through_storage(tmp_path, monkeypatch):
    user_id = _enroll_helper(tmp_path, monkeypatch, n_samples=2)
    svm = database.get_user_features(user_id, "svm")
    sqz = database.get_user_features(user_id, "squeezenet")
    assert len(svm) == 2
    assert len(sqz) == 2
    assert all(len(f) == 42 for f in svm)
    assert all(len(f) == 512 for f in sqz)


def test_get_user_sample_ids(tmp_path, monkeypatch):
    from layers.data_acquisition import get_user_sample_ids
    user_id = _enroll_helper(tmp_path, monkeypatch, n_samples=3)
    sample_ids = get_user_sample_ids(user_id, str(tmp_path))
    assert sample_ids == ["sample_000", "sample_001", "sample_002"]


def test_delete_user(tmp_path, monkeypatch):
    user_id = _enroll_helper(tmp_path, monkeypatch)
    assert database.delete_user(user_id)
    assert database.get_all_users() == []
    assert database.get_user_features(user_id, "svm") == []


def test_database_stats(tmp_path, monkeypatch):
    _enroll_helper(tmp_path, monkeypatch, n_samples=2)
    stats = database.get_database_stats()
    assert stats["total_users"] == 1
    assert stats["total_samples"] == 2
