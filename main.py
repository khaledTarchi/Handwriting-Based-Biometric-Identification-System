"""
HB-BIS - Handwriting-Based Biometric Identification System

MAIN ENTRY POINT
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import VERBOSE_MODE
from layers.gui import launch_gui


def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    This helps users get clear error messages if they're missing packages.
    """
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image'
    }
    
    missing = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print("=" * 70)
        print("ERROR: Missing Required Dependencies")
        print("=" * 70)
        print("\nThe following packages are not installed:")
        for pkg in missing:
            print(f"  • {pkg}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        print("\nOr install all dependencies at once:")
        print("  pip install -r requirements.txt")
        print("=" * 70)
        return False
    
    return True


def print_banner():
    """Print welcome banner."""
    banner = """
    ===================================================================
    
              HB-BIS - Biometric Identification System
                    v1.0
    
     Features:
       - Image Preprocessing Pipeline
       - Feature Engineering (SVM + SqueezeNet)
       - Encrypted Database Storage
       - Similarity-Based Identification
       - Model Retraining (Triplet Loss)
    
    ===================================================================
    """
    print(banner)


def main():
    """
    Main entry point for the HB-BIS system.
    
    Workflow:
    1. Check dependencies
    2. Display welcome message
    3. Initialize system
    4. Launch GUI
    """
    # Print banner
    print_banner()
    
    # Check dependencies
    print("\n[System Check] Verifying dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("[System Check] All dependencies found\n")
    
    print("\n[Initialization] Starting HB-BIS...")
    
    try:
        # Pre-load SqueezeNet model (download if needed)
        if VERBOSE_MODE:
            print("[Initialization] Loading SqueezeNet model...")
        
        from layers.feature_engineering import get_squeezenet_model
        _ = get_squeezenet_model()
        
        if VERBOSE_MODE:
            print("[Initialization] Model loaded successfully\n")
        
        # Launch GUI
        print("[GUI] Launching graphical interface...\n")
        launch_gui()
        
    except KeyboardInterrupt:
        print("\n\n[Shutdown] User interrupted. Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred:")
        print(f"  {str(e)}")
        print("\nPlease check:")
        print("  1. All dependencies are installed correctly")
        print("  2. You have sufficient disk space")
        print("  3. Internet connection (for model download)")
        sys.exit(1)


if __name__ == "__main__":
    main()
