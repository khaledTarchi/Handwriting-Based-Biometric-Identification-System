"""Tests for the preprocessing layer (shapes, dtypes, value ranges)."""
import numpy as np
import cv2
from PIL import Image, ImageDraw

from layers.preprocessing import (
    pil_to_cv2, cv2_to_pil, denoise_image, normalize_lighting,
    binarize_image, normalize_size, normalize_strokes,
    preprocess_for_svm, preprocess_for_squeezenet,
)
from config import TARGET_IMAGE_SIZE


def _sample_image():
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(5):
        d.arc([50 + i * 60, 40, 110 + i * 60, 120], 0, 300, fill=(0, 0, 0), width=5)
    return img


def test_conversions_roundtrip():
    img = _sample_image()
    cv = pil_to_cv2(img)
    back = cv2_to_pil(cv)
    assert np.array_equal(np.array(back), np.array(img))


def test_stage_outputs():
    img = _sample_image()
    cv = pil_to_cv2(img)

    denoised = denoise_image(cv)
    assert denoised.shape == cv.shape
    assert denoised.dtype == cv.dtype

    lit = normalize_lighting(denoised)
    assert lit.ndim == 2
    assert lit.dtype == np.uint8

    binary = binarize_image(lit)
    assert set(np.unique(binary)).issubset({0, 255})

    resized = normalize_size(binary, TARGET_IMAGE_SIZE)
    assert resized.shape[:2] == TARGET_IMAGE_SIZE

    opened = normalize_strokes(resized)
    assert opened.shape == resized.shape


def test_preprocess_for_svm_contract():
    img = _sample_image()
    out = preprocess_for_svm(img)
    assert out.shape == TARGET_IMAGE_SIZE
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert (out > 127).mean() > 0.001  # ink survived the pipeline


def test_preprocess_for_squeezenet_contract():
    img = _sample_image()
    out = preprocess_for_squeezenet(img)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
