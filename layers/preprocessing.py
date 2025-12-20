"""
Preprocessing Layer
===================
Layer 2 of the HB-BIS system

This layer prepares raw handwriting images for feature extraction.
The preprocessing pipeline is UNIFIED for both SVM and SqueezeNet models.

Processing steps:
1. Denoising - Remove scanner artifacts and noise
2. Lighting Normalization - Compensate for varying illumination
3. Binarization - Separate ink from background (critical for handwriting!)
4. Size Normalization - Standardize dimensions
5. Stroke Normalization - Standardize relative thickness

Why preprocessing matters:
    Raw images vary in quality, lighting, noise, and scale. Feature extraction
    algorithms expect normalized input. Without preprocessing, variations
    unrelated to writing style (like lighting) would dominate the features,
    causing poor matching accuracy.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from config import (
    TARGET_IMAGE_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    MORPH_KERNEL_SIZE,
    VERBOSE_MODE,
    SHOW_PREPROCESSING_STEPS
)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image."""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Step 1: Remove noise from the image.
    
    Uses Gaussian blur to reduce high-frequency noise (scanner artifacts, dust).
    
    Note:
        Gaussian blur is a simple but effective denoising method. It averages
        each pixel with its neighbors using a bell-curve weight distribution.
        
        Trade-off: Too much blur removes fine details in handwriting!
        We use a small kernel (5x5) to preserve stroke details.
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Denoised image
    """
    if VERBOSE_MODE:
        print("[Preprocessing] Step 1: Denoising...")
    
    # Apply Gaussian blur
    denoised = cv2.GaussianBlur(image, GAUSSIAN_BLUR_KERNEL, 0)
    
    return denoised


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Step 2: Normalize lighting variations.
    
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle
    varying illumination across the image.
    
    Note:
        Global histogram equalization would over-amplify noise in uniform regions.
        CLAHE divides the image into tiles and equalizes each separately, then
        blends them smoothly. This handles shadows and uneven lighting better.
        
        Why this matters: A handwriting sample scanned under bright light vs
        dim light should produce similar features. Lighting is NOT a biometric!
    
    Args:
        image: Input image (BGR)
        
    Returns:
        Lighting-normalized image
    """
    if VERBOSE_MODE:
        print("[Preprocessing] Step 2: Normalizing lighting...")
    
    # Convert to grayscale for CLAHE
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    normalized = clahe.apply(gray)
    
    return normalized


def binarize_image(image: np.ndarray) -> np.ndarray:
    """
    Step 3: Binarize the image (separate ink from background).
    
    Uses Otsu's method to automatically find optimal threshold.
    
    Note:
        Binarization is CRITICAL for handwriting analysis. It:
        1. Removes paper texture and color variations
        2. Eliminates lighting gradients
        3. Focuses analysis purely on ink vs no-ink
        
        Otsu's method finds the threshold that minimizes intra-class variance
        (variance within foreground and background classes). It's optimal when
        the histogram has two clear peaks (bimodal distribution).
        
        Result: 255 = ink (foreground), 0 = background
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image (0 or 255)
    """
    if VERBOSE_MODE:
        print("[Preprocessing] Step 3: Binarization (Otsu's method)...")
    
    # Apply Otsu's thresholding
    # THRESH_BINARY inverts result (white text on black background)
    # THRESH_BINARY_INV gives black text on white (standard for documents)
    _, binary = cv2.threshold(
        image,
        0,  # Threshold value (ignored with Otsu)
        255,  # Max value
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    return binary


def normalize_size(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Step 4: Resize image to standard dimensions.
    
    Note:
        Feature extraction algorithms expect consistent input size. Variable
        sizing would cause:
        1. Inconsistent feature vector dimensions
        2. Scale-dependent features (large vs small handwriting)
        
        We use INTER_AREA for downsampling (best quality) and INTER_CUBIC
        for upsampling (smoother). Aspect ratio is NOT preserved - this is
        intentional to normalize both width and height variations.
    
    Args:
        image: Input image
        target_size: (width, height) tuple
        
    Returns:
        Resized image
    """
    if VERBOSE_MODE:
        print(f"[Preprocessing] Step 4: Resizing to {target_size}...")
    
    # Choose interpolation method based on whether we're upsampling or downsampling
    current_size = (image.shape[1], image.shape[0])  # OpenCV uses (height, width)
    
    if current_size[0] * current_size[1] > target_size[0] * target_size[1]:
        # Downsampling - use INTER_AREA (best for shrinking)
        interpolation = cv2.INTER_AREA
    else:
        # Upsampling - use INTER_CUBIC (smooth)
        interpolation = cv2.INTER_CUBIC
    
    resized = cv2.resize(image, target_size, interpolation=interpolation)
    
    return resized


def normalize_strokes(image: np.ndarray) -> np.ndarray:
    """
    Step 5: Normalize stroke thickness.
    
    Uses morphological operations to standardize relative stroke width.
    
    Note:
        Different pens, scan resolutions, and writing pressure cause stroke
        thickness variations. We want to capture the SHAPE of strokes, not
        their absolute thickness.
        
        Method: Apply gentle erosion followed by dilation (morphological opening).
        This removes small artifacts and standardizes thickness slightly.
        
        Warning: Too much morphological processing destroys fine details!
    
    Args:
        image: Binary image
        
    Returns:
        Stroke-normalized binary image
    """
    if VERBOSE_MODE:
        print("[Preprocessing] Step 5: Normalizing strokes...")
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    
    # Apply morphological opening (erosion followed by dilation)
    # This removes small noise and standardizes stroke thickness
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return opened


def preprocess_for_svm(image: Image.Image) -> np.ndarray:
    """
    Complete preprocessing pipeline for SVM feature extraction.
    
    Returns a binary image optimized for handcrafted feature extraction.
    
    Args:
        image: Input PIL Image (RGB)
        
    Returns:
        Binary numpy array (224x224, values 0 or 255)
        
    Note:
        SVM features (ink density, orientation, curvature) work best on
        clean binary images. The pipeline removes all variations except
        the actual writing strokes.
    """
    if VERBOSE_MODE:
        print("\n[Preprocessing] Starting SVM preprocessing pipeline...")
    
    # Convert PIL to OpenCV format
    cv2_image = pil_to_cv2(image)
    
    # Step 1: Denoise
    denoised = denoise_image(cv2_image)
    
    # Step 2: Normalize lighting
    normalized_lighting = normalize_lighting(denoised)
    
    # Step 3: Binarize
    binary = binarize_image(normalized_lighting)
    
    # Step 4: Resize
    resized = normalize_size(binary, TARGET_IMAGE_SIZE)
    
    # Step 5: Normalize strokes
    final = normalize_strokes(resized)
    
    if VERBOSE_MODE:
        print("[Preprocessing] OK SVM preprocessing complete\n")
    
    return final


def preprocess_for_squeezenet(image: Image.Image) -> np.ndarray:
    """
    Complete preprocessing pipeline for SqueezeNet feature extraction.
    
    Returns an RGB tensor ready for neural network input.
    
    Args:
        image: Input PIL Image (RGB)
        
    Returns:
        Float32 numpy array (224x224x3, normalized to [0, 1])
        
    Note:
        Deep learning models trained on ImageNet expect RGB images normalized
        to [0, 1] or mean-centered. We perform the same preprocessing as SVM
        to get a clean binary image, then convert back to 3-channel RGB.
        
        This approach gives the network clean, standardized input similar to
        what it saw during ImageNet training (even though handwriting is very
        different from natural images!).
    """
    if VERBOSE_MODE:
        print("\n[Preprocessing] Starting SqueezeNet preprocessing pipeline...")
    
    # Convert PIL to OpenCV format
    cv2_image = pil_to_cv2(image)
    
    # Step 1: Denoise
    denoised = denoise_image(cv2_image)
    
    # Step 2: Normalize lighting
    normalized_lighting = normalize_lighting(denoised)
    
    # Step 3: Binarize
    binary = binarize_image(normalized_lighting)
    
    # Step 4: Resize
    resized = normalize_size(binary, TARGET_IMAGE_SIZE)
    
    # Step 5: Normalize strokes
    normalized_strokes = normalize_strokes(resized)
    
    # Convert binary to RGB (3 channels) - network expects RGB
    rgb_image = cv2.cvtColor(normalized_strokes, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1] range (neural networks expect this)
    normalized_rgb = rgb_image.astype(np.float32) / 255.0
    
    if VERBOSE_MODE:
        print("[Preprocessing] OK SqueezeNet preprocessing complete\n")
    
    return normalized_rgb


def visualize_preprocessing_steps(image: Image.Image):
    """
    Visualize each preprocessing step.

    This function displays the image at each stage of preprocessing.

    Args:
        image: Input PIL Image
    """
    import matplotlib.pyplot as plt
    
    cv2_image = pil_to_cv2(image)
    
    # Apply each step
    step1 = denoise_image(cv2_image)
    step2 = normalize_lighting(step1)
    step3 = binarize_image(step2)
    step4 = normalize_size(step3, TARGET_IMAGE_SIZE)
    step5 = normalize_strokes(step4)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Denoised
    axes[0, 1].imshow(cv2.cvtColor(step1, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Step 1: Denoised')
    axes[0, 1].axis('off')
    
    # Lighting normalized
    axes[0, 2].imshow(step2, cmap='gray')
    axes[0, 2].set_title('Step 2: Lighting Normalized')
    axes[0, 2].axis('off')
    
    # Binarized
    axes[1, 0].imshow(step3, cmap='gray')
    axes[1, 0].set_title('Step 3: Binarized (Otsu)')
    axes[1, 0].axis('off')
    
    # Resized
    axes[1, 1].imshow(step4, cmap='gray')
    axes[1, 1].set_title('Step 4: Resized to 224x224')
    axes[1, 1].axis('off')
    
    # Final
    axes[1, 2].imshow(step5, cmap='gray')
    axes[1, 2].set_title('Step 5: Strokes Normalized')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPreprocessing reduces variations unrelated to writing style:")
    print("  Lighting variations removed")
    print("  Noise and artifacts removed")
    print("  Size standardized")
    print("  Stroke thickness normalized")
    print("  Result: Clean features representing only the writing style")


# ============================================================================
# Entry point (when run directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Preprocessing Layer")
    print("=" * 70)
    
    print("\nPreprocessing Pipeline Steps:")
    print("  1. Denoising - Remove scanner noise")
    print("  2. Lighting Normalization - Handle varying illumination (CLAHE)")
    print("  3. Binarization - Separate ink from background (Otsu)")
    print("  4. Size Normalization - Standardize to 224x224")
    print("  5. Stroke Normalization - Standardize relative thickness")
    
    print("\nWhy each step matters:")
    print("  • Denoising: Scanner dust and artifacts aren't biometrics")
    print("  • Lighting: Same handwriting in shadow vs sunlight should match")
    print("  • Binarization: Paper color and texture aren't biometrics")
    print("  • Size: Large vs small writing shouldn't affect matching")
    print("  • Strokes: Pen thickness isn't a reliable biometric")
    
    print("\nResult: Only the WRITING STYLE remains for feature extraction!")
    print("=" * 70)
