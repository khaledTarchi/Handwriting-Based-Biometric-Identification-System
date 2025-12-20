"""Tests for the encryption module (round-trips + dimension checks)."""
import numpy as np
import pytest
from PIL import Image

from utils.encryption import (
    encrypt_data, decrypt_data, encrypt_image, decrypt_image,
    encrypt_features, decrypt_features,
)
from config import SVM_FEATURE_DIM, SQUEEZENET_FEATURE_DIM


def test_data_roundtrip():
    original = b"test biometric payload \x00\xff"
    assert decrypt_data(encrypt_data(original)) == original


def test_image_roundtrip():
    img = Image.new("RGB", (64, 48), (200, 100, 50))
    decrypted = decrypt_image(encrypt_image(img))
    assert decrypted.size == img.size
    assert decrypted.mode == "RGB"
    assert np.array_equal(np.array(decrypted), np.array(img))


def test_svm_features_roundtrip():
    feats = np.random.randn(SVM_FEATURE_DIM).astype(np.float32)
    restored = decrypt_features(encrypt_features(feats), SVM_FEATURE_DIM)
    assert restored.shape == (SVM_FEATURE_DIM,)
    assert np.allclose(restored, feats, atol=1e-4)


def test_squeezenet_features_roundtrip():
    feats = np.random.randn(SQUEEZENET_FEATURE_DIM).astype(np.float32)
    restored = decrypt_features(encrypt_features(feats), SQUEEZENET_FEATURE_DIM)
    assert restored.shape == (SQUEEZENET_FEATURE_DIM,)
    assert np.allclose(restored, feats, atol=1e-4)


def test_dimension_mismatch_raises():
    feats = np.random.randn(42).astype(np.float32)
    encrypted = encrypt_features(feats)
    with pytest.raises(ValueError):
        decrypt_features(encrypted, 41)


def test_wrong_key_fails_to_roundtrip(monkeypatch):
    import utils.encryption as enc_mod
    feats = np.random.randn(10).astype(np.float32)
    encrypted = encrypt_features(feats)
    monkeypatch.setattr(enc_mod, "ENCRYPTION_KEY", b"different-key")
    restored = decrypt_features(encrypted, 10)
    assert not np.allclose(restored, feats)
