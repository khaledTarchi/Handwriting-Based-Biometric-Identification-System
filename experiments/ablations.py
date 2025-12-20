"""
Preprocessing ablation study.

Each variant removes exactly ONE stage of the HB-BIS preprocessing pipeline
(denoise / CLAHE lighting normalization / Otsu binarization / morphological
stroke normalization), keeping everything else identical. This isolates the
contribution of every preprocessing stage to identification accuracy.

The variant pipeline recomposes the SAME stage functions used by the core
pipeline, so this is an ablation of the existing algorithm - not a change of it.
"""

import numpy as np
import cv2
from PIL import Image

from config import TARGET_IMAGE_SIZE
from layers.preprocessing import (
    pil_to_cv2,
    denoise_image,
    normalize_lighting,
    binarize_image,
    normalize_size,
    normalize_strokes,
)
from layers.feature_engineering import extract_svm_features, extract_squeezenet_features


def extract_variant_features(image: Image.Image, variant: str, model_type: str) -> np.ndarray:
    """
    Extract features using a preprocessing variant.

    Args:
        image: raw PIL image (as acquired)
        variant: one of PIPELINE_VARIANTS ("full", "no_denoise", "no_clahe",
                 "no_binarize", "no_morph")
        model_type: "svm" or "squeezenet"

    Returns:
        feature vector (same as the core pipeline for variant="full")
    """
    img = pil_to_cv2(image)

    # Step 1: denoising (removed by "no_denoise")
    if variant != "no_denoise":
        img = denoise_image(img)

    # Step 2: lighting normalization / CLAHE (removed by "no_clahe")
    if variant != "no_clahe":
        img = normalize_lighting(img)

    # The core pipeline always operates on grayscale from step 2 onward
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Otsu binarization (removed by "no_binarize" -> grayscale kept)
    if variant != "no_binarize":
        img = binarize_image(img)

    # Step 4: size normalization (always applied, required for fixed input size)
    img = normalize_size(img, TARGET_IMAGE_SIZE)

    # Step 5: stroke normalization / morphological opening (removed by "no_morph")
    if variant != "no_morph":
        img = normalize_strokes(img)

    if model_type == "svm":
        return extract_svm_features(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return extract_squeezenet_features(rgb.astype(np.float32) / 255.0)
