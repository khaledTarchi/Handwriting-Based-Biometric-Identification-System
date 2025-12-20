"""
Acquisition-robustness perturbations.

These model realistic acquisition/sensor degradations and are applied to the
RAW image BEFORE the HB-BIS preprocessing pipeline, exactly like a noisy scan
or photograph would arrive at the system input. Each perturbation is applied
deterministically (seeded) for reproducibility.
"""

import numpy as np
import cv2
from PIL import Image


def apply_gaussian_blur(image, sigma, seed=0):
    """Sensor/optics blur. sigma = Gaussian kernel standard deviation."""
    return Image.fromarray(cv2.GaussianBlur(np.array(image), (0, 0), sigma))


def apply_gaussian_noise(image, sigma, seed=0):
    """Additive Gaussian sensor noise (intensity standard deviation sigma)."""
    rng = np.random.RandomState(int(seed))
    arr = np.array(image).astype(np.float32)
    noise = rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def apply_rotation(image, degrees, seed=0):
    """In-plane rotation (e.g., skewed scan). Rotates and fills with white."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degrees, 1.0)
    rotated = cv2.warpAffine(arr, matrix, (w, h), borderValue=255)
    return Image.fromarray(rotated)


def apply_brightness_gain(image, gain, seed=0):
    """Global illumination change (multiply intensity by gain < 1 dims the scan)."""
    arr = np.clip(np.array(image).astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_resolution_scale(image, factor, seed=0):
    """Reduced scan resolution: downsample then upsample back to canvas size."""
    arr = np.array(image)
    h, w = arr.shape[:2]
    small_h, small_w = max(1, int(h * factor)), max(1, int(w * factor))
    small = cv2.resize(arr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    upsampled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(upsampled)


PERTURBATION_FUNCS = {
    "gaussian_blur": apply_gaussian_blur,
    "gaussian_noise": apply_gaussian_noise,
    "rotation": apply_rotation,
    "brightness_gain": apply_brightness_gain,
    "resolution_scale": apply_resolution_scale,
}


def apply(name, image, parameter, seed=0):
    """Apply perturbation ``name`` with ``parameter`` to a PIL image."""
    return PERTURBATION_FUNCS[name](image, parameter, seed=seed)
