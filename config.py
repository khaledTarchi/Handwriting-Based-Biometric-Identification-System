"""
HB-BIS Configuration File
=========================
Centralized configuration for the Handwriting-Based Biometric Identification System.
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Root directory for database storage
DATABASE_ROOT = os.path.join(os.path.dirname(__file__), "database")

# Root directory for saved models
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models", "saved")

# Ensure directories exist
os.makedirs(DATABASE_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)

# ============================================================================
# IMAGE PREPROCESSING PARAMETERS
# ============================================================================

# Target image size for neural network input (SqueezeNet expects 224x224)
TARGET_IMAGE_SIZE = (224, 224)

# Gaussian blur kernel size for noise reduction (must be odd numbers)
GAUSSIAN_BLUR_KERNEL = (5, 5)

# CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters
CLAHE_CLIP_LIMIT = 2.0  # Threshold for contrast limiting
CLAHE_TILE_GRID_SIZE = (8, 8)  # Size of grid for histogram equalization

# Morphological operations for stroke normalization
MORPH_KERNEL_SIZE = (3, 3)  # Kernel size for erosion/dilation

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

# SVM feature vector dimensionality
# Breakdown: 4 (ink density) + 8 (orientation) + 6 (curvature) + 
#            8 (geometry) + 6 (spacing) + 10 (statistical) = 42
SVM_FEATURE_DIM = 42

# SqueezeNet embedding dimensionality (from final conv layer)
SQUEEZENET_FEATURE_DIM = 512

# Number of orientation bins for gradient histogram (0-360 degrees)
ORIENTATION_BINS = 8

# ============================================================================
# DECISION THRESHOLDS
# ============================================================================
# These thresholds determine the trade-off between security and usability.
# Lower threshold = stricter matching (higher security, more rejections)
# Higher threshold = looser matching (more convenient, less secure)

# SVM Thresholds (using cosine distance, range [0, 2])
SVM_THRESHOLD_ACCEPT = 0.15   # Below this: MATCH (high confidence)
SVM_THRESHOLD_REJECT = 0.30   # Above this: UNKNOWN (reject)
                              # Between: UNCERTAIN (manual review)

# SqueezeNet Thresholds (using cosine distance, range [0, 2])
SQUEEZENET_THRESHOLD_ACCEPT = 0.20   # Deep features generally need slightly higher threshold
SQUEEZENET_THRESHOLD_REJECT = 0.40

# ============================================================================
# ENROLLMENT & SIMILARITY WARNING
# ============================================================================

# Minimum number of samples recommended per user for robust enrollment
MIN_SAMPLES_RECOMMENDED = 3

# Maximum number of samples to store per user (to limit storage)
MAX_SAMPLES_PER_USER = 10

# Similarity warning threshold: warn if new user's features are too similar
# to existing users (possible duplicate enrollment or impersonation attempt)
SIMILARITY_WARNING_THRESHOLD = 0.10  # Very low distance = suspiciously similar

# ============================================================================
# SQUEEZENET RETRAINING PARAMETERS
# ============================================================================

# Minimum requirements for retraining
MIN_USERS_FOR_RETRAIN = 3       # Need at least 3 users
MIN_SAMPLES_FOR_RETRAIN = 2     # Each user needs at least 2 samples

# Triplet loss parameters (for fine-tuning SqueezeNet)
TRIPLET_MARGIN = 0.2            # Margin for triplet loss
LEARNING_RATE = 0.0001          # Small learning rate for fine-tuning
RETRAIN_EPOCHS = 10             # Number of epochs for retraining
BATCH_SIZE = 8                  # Batch size for training

# ============================================================================
# ENCRYPTION CONFIGURATION
# ============================================================================

# Enable/disable encryption
ENCRYPTION_ENABLED = True

# Encryption key (XOR demonstration)
ENCRYPTION_KEY = b"HB-BIS-XOR-KEY-2026"

# ============================================================================
# GUI CONFIGURATION
# ============================================================================

# Main window size
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 750
WINDOW_SIZE = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

# Tkinter theme (options: 'clam', 'alt', 'default', 'classic')
TTK_THEME = "clam"

# Colors (for result display)
COLOR_MATCH = "#2ecc71"      # Green for successful match
COLOR_UNCERTAIN = "#f39c12"  # Orange for uncertain
COLOR_UNKNOWN = "#e74c3c"    # Red for unknown/rejected
COLOR_BG = "#ecf0f1"         # Light gray background
COLOR_PRIMARY = "#3498db"    # Blue primary color

# Image preview size in GUI
PREVIEW_IMAGE_SIZE = (200, 200)

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Minimum image dimensions (pixels)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100

# Maximum image dimensions (before resizing)
MAX_IMAGE_WIDTH = 4000
MAX_IMAGE_HEIGHT = 4000

# Minimum contrast ratio (to ensure image is not blank)
MIN_CONTRAST_RATIO = 0.1

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

# Enable verbose logging
VERBOSE_MODE = True

# Show preprocessing steps visualization
SHOW_PREPROCESSING_STEPS = False  # Set to True to see intermediate images

# ============================================================================
# INFO MESSAGES
# ============================================================================

MODEL_COMPARISON_INFO = """
SVM vs SqueezeNet - Understanding the Difference:

SVM (Classical Approach):
• Uses 42 handcrafted features (ink density, strokes, curvature, etc.)
• Features are interpretable - we know what each dimension means
• Fast feature extraction (~20-30ms)
• Works well with small datasets
• Requires domain expertise to design features

SqueezeNet (Deep Learning Approach):
• Uses 512-dimensional learned representation
• Features are NOT interpretable - learned automatically
• Slower feature extraction (~50-100ms on CPU)
• Generally more powerful with sufficient data
• Can fine-tune with triplet loss for better discrimination
• No feature engineering needed

Both approaches are valid and offer different strengths in biometric systems.
"""
