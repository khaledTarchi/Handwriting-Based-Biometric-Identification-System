"""
Feature Engineering Layer
==========================
Layer 3 of the HB-BIS system

This layer extracts biometric features from preprocessed images using TWO
fundamentally different approaches:

1. SVM approach: Handcrafted features (42-dimensional)
   - Interpretable: each dimension has a clear meaning
   - Based on domain expertise (ink density, curvature, orientation, etc.)
   - Fast extraction (~20-30ms)

2. SqueezeNet approach: Deep learned features (512-dimensional)
   - Non-interpretable: learned automatically by the network
   - No domain expertise required
   - Slower extraction (~50-100ms on CPU)

The two approaches represent the evolution of feature extraction in computer
vision: from manual engineering to automatic learning. Both have pros and
cons - neither is universally "better".
"""

import os
import numpy as np
import cv2
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torchvision import models
from config import (
    SVM_FEATURE_DIM,
    SQUEEZENET_FEATURE_DIM,
    ORIENTATION_BINS,
    MODELS_ROOT,
    VERBOSE_MODE
)


def _create_pretrained_squeezenet() -> nn.Module:
    """
    Load ImageNet-pretrained SqueezeNet using the modern torchvision weights
    API, with a graceful fallback for older torchvision versions.
    """
    try:
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        return models.squeezenet1_1(weights=weights)
    except Exception:
        # Older torchvision (< 0.13): weights enum API unavailable
        return models.squeezenet1_1(pretrained=True)


FINETUNED_WEIGHTS_PATH = os.path.join(MODELS_ROOT, "squeezenet_finetuned.pth")


# ============================================================================
# SVM HANDCRAFTED FEATURES (Classical Approach)
# ============================================================================

def extract_svm_features(binary_image: np.ndarray) -> np.ndarray:
    """
    Extract 42-dimensional handcrafted feature vector from binary image.
    
    Feature breakdown (all interpretable!):
    - 4 features: Ink density (overall + per quadrant)
    - 8 features: Stroke orientation histogram
    - 6 features: Curvature statistics
    - 8 features: Geometric measurements
    - 6 features: Spacing and gap analysis
    - 10 features: Statistical moments
    
    Args:
        binary_image: Preprocessed binary image (0 or 255)
        
    Returns:
        42-dimensional feature vector (normalized)
        
    Note:
        These features were designed to capture handwriting characteristics:
        - Are strokes mostly vertical or diagonal?
        - How curvy is the writing?
        - How dense is the ink?
        - How are strokes spaced?
        
        Designing good features requires expertise in handwriting analysis!
    """
    if VERBOSE_MODE:
        print("[Feature Engineering] Extracting SVM features...")
    
    features = []
    
    # Ensure we're working with binary (0 or 1)
    binary = (binary_image > 127).astype(np.uint8)
    
    # === Group 1: Ink Density (4 features) ===
    ink_density_features = extract_ink_density(binary)
    features.extend(ink_density_features)
    
    # === Group 2: Stroke Orientation (8 features) ===
    orientation_features = extract_orientation_histogram(binary)
    features.extend(orientation_features)
    
    # === Group 3: Curvature (6 features) ===
    curvature_features = extract_curvature_features(binary)
    features.extend(curvature_features)
    
    # === Group 4: Geometric Measurements (8 features) ===
    geometric_features = extract_geometric_features(binary)
    features.extend(geometric_features)
    
    # === Group 5: Spacing & Gaps (6 features) ===
    spacing_features = extract_spacing_features(binary)
    features.extend(spacing_features)
    
    # === Group 6: Statistical Moments (10 features) ===
    statistical_features = extract_statistical_features(binary)
    features.extend(statistical_features)
    
    # Convert to numpy array
    feature_vector = np.array(features, dtype=np.float32)
    
    # Sanity check
    assert len(feature_vector) == SVM_FEATURE_DIM, \
        f"Feature dimension mismatch: expected {SVM_FEATURE_DIM}, got {len(feature_vector)}"
    
    # Normalize to unit length (L2 normalization)
    # This makes cosine distance easier to interpret
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    if VERBOSE_MODE:
        print(f"[Feature Engineering] OK Extracted {len(feature_vector)} SVM features")
    
    return feature_vector


def extract_ink_density(binary: np.ndarray) -> list:
    """
    Extract ink density features (4 features).
    
    Features:
    1. Overall ink density
    2-4. Density in each quadrant (helps capture layout patterns)
    
    Note:
        Some people write densely (lots of ink), others sparsely.
        Quadrant analysis captures if someone tends to write more in certain areas.
    """
    h, w = binary.shape
    total_pixels = h * w
    
    # Overall density
    overall_density = np.sum(binary) / total_pixels
    
    # Quadrant densities
    h_mid, w_mid = h // 2, w // 2
    
    q1_density = np.sum(binary[:h_mid, :w_mid]) / (h_mid * w_mid)  # Top-left
    q2_density = np.sum(binary[:h_mid, w_mid:]) / (h_mid * (w - w_mid))  # Top-right
    q3_density = np.sum(binary[h_mid:, :w_mid]) / ((h - h_mid) * w_mid)  # Bottom-left
    
    return [overall_density, q1_density, q2_density, q3_density]


def extract_orientation_histogram(binary: np.ndarray) -> list:
    """
    Extract stroke orientation histogram (8 features).
    
    Uses gradient direction to determine stroke orientation.
    Divides 360 degrees into 8 bins (45 degrees each).
    
    Note:
        Some people write with more vertical strokes, others more diagonal.
        The distribution of stroke angles is a distinctive biometric feature.
    """
    # Compute gradients using Sobel operators
    grad_x = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(binary.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and angle
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)  # Range: [-π, π]
    
    # Convert to degrees and normalize to [0, 360)
    angle_deg = np.degrees(angle) % 360
    
    # Only consider pixels with significant gradient (actual strokes)
    threshold = np.mean(magnitude) + np.std(magnitude)
    significant_angles = angle_deg[magnitude > threshold]
    
    if len(significant_angles) == 0:
        return [0.0] * ORIENTATION_BINS
    
    # Create histogram (8 bins)
    hist, _ = np.histogram(significant_angles, bins=ORIENTATION_BINS, range=(0, 360))
    
    # Normalize to sum to 1 (convert counts to probability distribution)
    hist_normalized = hist.astype(np.float32) / (np.sum(hist) + 1e-7)
    
    return hist_normalized.tolist()


def extract_curvature_features(binary: np.ndarray) -> list:
    """
    Extract curvature features (6 features).
    
    Measures how curvy the strokes are.
    
    Features:
    1. Mean curvature
    2. Std of curvature
    3. Max curvature
    4. Ratio of high curvature points
    5. Curvature entropy
    6. Curvature skewness
    
    Note:
        Some handwriting is very curvy (lots of loops), others more angular.
        Curvature captures the smoothness vs sharpness of strokes.
    """
    # Find contours (stroke boundaries)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_curvatures = []
    
    for contour in contours:
        if len(contour) < 5:  # Need at least 5 points for curvature
            continue
        
        # Approximate curvature using angle changes between consecutive segments
        points = contour[:, 0, :]
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Angle between vectors (curvature indicator)
            len_v1 = np.linalg.norm(v1)
            len_v2 = np.linalg.norm(v2)
            
            if len_v1 > 0 and len_v2 > 0:
                cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                all_curvatures.append(angle)
    
    if len(all_curvatures) == 0:
        return [0.0] * 6
    
    all_curvatures = np.array(all_curvatures)
    
    # Extract statistics
    mean_curv = np.mean(all_curvatures)
    std_curv = np.std(all_curvatures)
    max_curv = np.max(all_curvatures)
    
    # Ratio of high curvature points (sharp corners)
    high_curv_threshold = np.pi / 4  # 45 degrees
    high_curv_ratio = np.sum(all_curvatures > high_curv_threshold) / len(all_curvatures)
    
    # Entropy (measure of curvature variability)
    hist, _ = np.histogram(all_curvatures, bins=10)
    prob = hist.astype(np.float32) / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob * np.log(prob + 1e-7))
    
    # Skewness
    skewness = np.mean((all_curvatures - mean_curv)**3) / (std_curv**3 + 1e-7)
    
    return [mean_curv, std_curv, max_curv, high_curv_ratio, entropy, skewness]


def extract_geometric_features(binary: np.ndarray) -> list:
    """
    Extract geometric features (8 features).
    
    Features:
    1-2. Aspect ratio (height/width) of bounding box - mean and std
    3-4. Stroke length statistics
    5-6. Stroke width statistics
    7. Number of connected components (normalized)
    8. Fill ratio (ink area / bounding box area)
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return [0.0] * 8
    
    aspect_ratios = []
    lengths = []
    widths = []
    fill_ratios = []
    
    for contour in contours:
        if len(contour) < 5:
            continue
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        if w > 0 and h > 0:
            aspect_ratios.append(h / w)
            fill_ratios.append(cv2.contourArea(contour) / (w * h + 1e-7))
        
        # Contour length
        lengths.append(cv2.arcLength(contour, closed=False))
        
        # Approximate width (using minimum area rectangle)
        if len(contour) >= 5:
            rect = cv2.minAreaRect(contour)
            w_rect = min(rect[1])  # Smaller dimension = width
            widths.append(w_rect)
    
    # Compute statistics
    mean_aspect = np.mean(aspect_ratios) if aspect_ratios else 0.0
    std_aspect = np.std(aspect_ratios) if aspect_ratios else 0.0
    
    mean_length = np.mean(lengths) if lengths else 0.0
    std_length = np.std(lengths) if lengths else 0.0
    
    mean_width = np.mean(widths) if widths else 0.0
    std_width = np.std(widths) if widths else 0.0
    
    num_components = len(contours) / 100.0  # Normalize
    
    mean_fill = np.mean(fill_ratios) if fill_ratios else 0.0
    
    return [mean_aspect, std_aspect, mean_length, std_length, mean_width, std_width, num_components, mean_fill]


def extract_spacing_features(binary: np.ndarray) -> list:
    """
    Extract spacing and gap features (6 features).
    
    Analyzes white space patterns (gaps between strokes).
    
    Note:
        Some people write with tight spacing, others more spread out.
        Gap patterns are distinctive biometric features.
    """
    h, w = binary.shape
    
    # Horizontal projection (sum of ink per row)
    h_projection = np.sum(binary, axis=1)
    
    # Vertical projection (sum of ink per column)
    v_projection = np.sum(binary, axis=0)
    
    # Find gaps (rows/columns with no ink)
    h_gap = (h_projection == 0).astype(int)
    v_gap = (v_projection == 0).astype(int)
    
    # Gap statistics
    h_gap_ratio = np.sum(h_gap) / h
    v_gap_ratio = np.sum(v_gap) / w
    
    # Gap run lengths (consecutive gap regions)
    h_gap_runs = get_run_lengths(h_gap)
    v_gap_runs = get_run_lengths(v_gap)
    
    mean_h_gap = np.mean(h_gap_runs) if h_gap_runs else 0.0
    mean_v_gap = np.mean(v_gap_runs) if v_gap_runs else 0.0
    
    std_h_gap = np.std(h_gap_runs) if h_gap_runs else 0.0
    std_v_gap = np.std(v_gap_runs) if v_gap_runs else 0.0
    
    return [h_gap_ratio, v_gap_ratio, mean_h_gap, mean_v_gap, std_h_gap, std_v_gap]


def get_run_lengths(binary_array: np.ndarray) -> list:
    """Helper function to get run lengths of 1s in binary array."""
    runs = []
    current_run = 0
    
    for val in binary_array:
        if val == 1:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0
    
    if current_run > 0:
        runs.append(current_run)
    
    return runs


def extract_statistical_features(binary: np.ndarray) -> list:
    """
    Extract statistical features (10 features).
    
    Based on image moments and projections.
    
    Features:
    1-3. Horizontal projection: mean, std, skewness
    4-6. Vertical projection: mean, std, skewness
    7-10. Hu moments (first 4, invariant to rotation/scale)
    """
    h, w = binary.shape
    
    # Horizontal projection
    h_projection = np.sum(binary, axis=1)
    h_proj_mean = np.mean(h_projection)
    h_proj_std = np.std(h_projection)
    h_proj_skew = np.mean((h_projection - h_proj_mean)**3) / (h_proj_std**3 + 1e-7)
    
    # Vertical projection
    v_projection = np.sum(binary, axis=0)
    v_proj_mean = np.mean(v_projection)
    v_proj_std = np.std(v_projection)
    v_proj_skew = np.mean((v_projection - v_proj_mean)**3) / (v_proj_std**3 + 1e-7)
    
    # Hu moments (rotation/scale invariant)
    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Use log transform to normalize range
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    # Use first 4 Hu moments
    hu_features = hu_moments_log[:4].tolist()
    
    return [h_proj_mean, h_proj_std, h_proj_skew, v_proj_mean, v_proj_std, v_proj_skew] + hu_features


# ============================================================================
# SQUEEZENET DEEP FEATURES (Deep Learning Approach)
# ============================================================================

class SqueezeNetFeatureExtractor(nn.Module):
    """
    SqueezeNet model modified for feature extraction.
    
    Uses pretrained SqueezeNet1.1 from ImageNet.
    Removes final classifier, extracts 512D features from final conv layer.
    
    Note:
        SqueezeNet is a lightweight CNN designed for efficiency:
        - Only ~5MB (much smaller than ResNet/VGG)
        - Uses "fire modules" (squeeze + expand)
        - Good balance of accuracy vs size
        
        We don't train this from scratch - we use transfer learning!
        Even though ImageNet has no handwriting, the learned features
        (edges, textures, patterns) are still useful.
    """
    
    def __init__(self):
        super(SqueezeNetFeatureExtractor, self).__init__()
        
        # Load pretrained SqueezeNet1.1
        if VERBOSE_MODE:
            print("[Feature Engineering] Loading SqueezeNet1.1 (pretrained on ImageNet)...")

        squeezenet = _create_pretrained_squeezenet()
        
        # Extract features before the final classifier
        # SqueezeNet structure: features -> classifier
        # We want the output of 'features' module (512 channels)
        self.features = squeezenet.features
        
        # Add adaptive pooling to get fixed-size output
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Set to evaluation mode (disable dropout, batch norm updates)
        self.eval()
        
        if VERBOSE_MODE:
            print("[Feature Engineering] OK SqueezeNet loaded successfully")
    
    def forward(self, x):
        """
        Extract 512-dimensional feature vector.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
            
        Returns:
            Feature tensor (batch_size, 512)
        """
        # Pass through convolutional features
        x = self.features(x)
        
        # Global average pooling
        x = self.pooling(x)
        
        # Flatten to (batch_size, 512)
        x = torch.flatten(x, 1)
        
        return x


# Global model instances (loaded once, reused)
_squeezenet_model = None          # active model (auto / finetuned / custom)
_squeezenet_model_base = None     # always-pretrained model


def get_squeezenet_model(weights: Optional[str] = "auto"):
    """
    Get or create the SqueezeNet model (singleton pattern).

    Args:
        weights: 
            - "auto":  use the fine-tuned weights saved by the triplet-loss
                       retraining if they exist, otherwise pretrained
            - "pretrained": always use the ImageNet-pretrained weights
            - a file path: load the given weights file explicitly
            - None: always use pretrained weights

    Returns:
        SqueezeNetFeatureExtractor instance
    """
    global _squeezenet_model, _squeezenet_model_base

    if weights is None or weights == "pretrained":
        if _squeezenet_model_base is None:
            _squeezenet_model_base = SqueezeNetFeatureExtractor()
        return _squeezenet_model_base

    if _squeezenet_model is None:
        _squeezenet_model = SqueezeNetFeatureExtractor()

    if weights == "auto":
        if os.path.exists(FINETUNED_WEIGHTS_PATH):
            _squeezenet_model.load_state_dict(
                torch.load(FINETUNED_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            )
            if VERBOSE_MODE:
                print("[Feature Engineering] OK Loaded fine-tuned SqueezeNet weights")
        return _squeezenet_model

    # Explicit weights file path
    _squeezenet_model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True)
    )
    if VERBOSE_MODE:
        print(f"[Feature Engineering] OK Loaded SqueezeNet weights from {weights}")
    return _squeezenet_model


def extract_squeezenet_features(rgb_image: np.ndarray) -> np.ndarray:
    """
    Extract 512-dimensional deep feature vector using SqueezeNet.
    
    Args:
        rgb_image: Preprocessed RGB image (224x224x3, float32, normalized to [0,1])
        
    Returns:
        512-dimensional feature vector
        
    Note:
        Deep features are NOT interpretable - we don't know what each
        dimension represents. The network learned them automatically
        from millions of images. This is both a strength (no manual
        engineering) and a weakness (hard to debug/explain).
    """
    if VERBOSE_MODE:
        print("[Feature Engineering] Extracting SqueezeNet features...")
    
    # Get model
    model = get_squeezenet_model()
    
    # Prepare input tensor
    # PyTorch expects (batch, channels, height, width)
    # Our input is (height, width, channels)
    input_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)
    
    # Extract features (no gradient computation needed)
    with torch.no_grad():
        features = model(input_tensor)
    
    # Convert to numpy
    feature_vector = features.squeeze().numpy()
    
    # Sanity check
    assert len(feature_vector) == SQUEEZENET_FEATURE_DIM, \
        f"Feature dimension mismatch: expected {SQUEEZENET_FEATURE_DIM}, got {len(feature_vector)}"
    
    # Normalize to unit length (for cosine distance)
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    if VERBOSE_MODE:
        print(f"[Feature Engineering] OK Extracted {len(feature_vector)} SqueezeNet features")
    
    return feature_vector
