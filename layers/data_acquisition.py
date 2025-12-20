"""
Data Acquisition Layer
=======================
Layer 1 of the HB-BIS system

This layer handles:
- Loading images from files (scanner/camera sources)
- Validating image format and quality
- Encrypting images for secure storage
- Decrypting images for processing

It is the first step in any biometric system: acquiring the raw biometric
sample (handwriting image) and ensuring it meets quality standards before
further processing.
"""

from PIL import Image
import os
from typing import Tuple, Optional
from utils import encrypt_image, decrypt_image, validate_image_format, validate_image_quality
from config import VERBOSE_MODE


def load_image(filepath: str) -> Tuple[Optional[Image.Image], str]:
    """
    Load an image file with validation.
    
    This is the entry point for all biometric samples in the system.
    We perform strict validation to ensure data quality.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Tuple of (Image or None, status_message)
        
    Note:
        Quality control at acquisition is critical. Poor quality samples
        lead to feature extraction errors and matching failures later.
        It's better to reject bad samples early than to have undefined
        behavior during matching.
    """
    if VERBOSE_MODE:
        print(f"[Data Acquisition] Loading image: {filepath}")
    
    # Step 1: Validate file format
    is_valid_format, format_msg = validate_image_format(filepath)
    if not is_valid_format:
        return None, f"Format validation failed: {format_msg}"
    
    # Step 2: Load the image
    try:
        image = Image.open(filepath)
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        return None, f"Failed to load image: {str(e)}"
    
    # Step 3: Validate image quality
    is_valid_quality, quality_msg = validate_image_quality(image)
    if not is_valid_quality:
        return None, f"Quality validation failed: {quality_msg}"
    
    if VERBOSE_MODE:
        print(f"[Data Acquisition] OK Image loaded successfully: {image.size}")
    
    return image, "Image loaded successfully"


def save_encrypted_image(
    image: Image.Image, 
    user_id: str, 
    sample_id: str,
    database_root: str
) -> Tuple[bool, str]:
    """
    Encrypt and save an image to the user's database folder.
    
    Storage structure: database/users/{user_id}/raw_images/{sample_id}.enc
    
    Args:
        image: PIL Image to save
        user_id: Unique user identifier
        sample_id: Unique sample identifier for this user
        database_root: Root database directory
        
    Returns:
        Tuple of (success, message)
        
    Note:
        We store raw images for several reasons:
        1. Allows re-extraction of features if algorithm improves
        2. Enables visual verification during disputes
        3. Provides training data for future model improvements
        
        Encryption protects user privacy.
    """
    if VERBOSE_MODE:
        print(f"[Data Acquisition] Saving encrypted image for user {user_id}, sample {sample_id}")
    
    # Create directory structure
    raw_images_dir = os.path.join(database_root, "users", user_id, "raw_images")
    os.makedirs(raw_images_dir, exist_ok=True)
    
    # Encrypt the image
    try:
        encrypted_bytes = encrypt_image(image)
    except Exception as e:
        return False, f"Encryption failed: {str(e)}"
    
    # Save to file
    output_path = os.path.join(raw_images_dir, f"{sample_id}.enc")
    try:
        with open(output_path, 'wb') as f:
            f.write(encrypted_bytes)
    except Exception as e:
        return False, f"Failed to save encrypted image: {str(e)}"
    
    if VERBOSE_MODE:
        print(f"[Data Acquisition] OK Encrypted image saved: {output_path}")
    
    return True, output_path


def load_encrypted_image(
    user_id: str,
    sample_id: str,
    database_root: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Load and decrypt an image from the database.
    
    Args:
        user_id: Unique user identifier
        sample_id: Unique sample identifier
        database_root: Root database directory
        
    Returns:
        Tuple of (Image or None, status_message)
    """
    if VERBOSE_MODE:
        print(f"[Data Acquisition] Loading encrypted image for user {user_id}, sample {sample_id}")
    
    # Construct file path
    encrypted_path = os.path.join(
        database_root, "users", user_id, "raw_images", f"{sample_id}.enc"
    )
    
    # Check if file exists
    if not os.path.exists(encrypted_path):
        return None, f"Encrypted image not found: {encrypted_path}"
    
    # Load encrypted bytes
    try:
        with open(encrypted_path, 'rb') as f:
            encrypted_bytes = f.read()
    except Exception as e:
        return None, f"Failed to load encrypted file: {str(e)}"
    
    # Decrypt the image
    try:
        image = decrypt_image(encrypted_bytes)
    except Exception as e:
        return None, f"Decryption failed: {str(e)}"
    
    if VERBOSE_MODE:
        print(f"[Data Acquisition] OK Image decrypted successfully")
    
    return image, "Image decrypted successfully"


def get_user_sample_ids(user_id: str, database_root: str) -> list:
    """
    Get list of all sample IDs for a user.
    
    Args:
        user_id: Unique user identifier
        database_root: Root database directory
        
    Returns:
        List of sample IDs
    """
    raw_images_dir = os.path.join(database_root, "users", user_id, "raw_images")
    
    if not os.path.exists(raw_images_dir):
        return []
    
    # Get all .enc files and extract sample IDs
    sample_ids = []
    for filename in os.listdir(raw_images_dir):
        if filename.endswith('.enc'):
            sample_id = filename[:-4]  # Remove .enc extension
            sample_ids.append(sample_id)
    
    return sorted(sample_ids)
