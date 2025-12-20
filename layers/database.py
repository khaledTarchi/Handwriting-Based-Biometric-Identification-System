"""
Biometric Database Layer
=========================
Layer 4 of the HB-BIS system

This layer manages persistent storage of all biometric data:
- Encrypted raw images
- Encrypted SVM feature vectors (42D)
- Encrypted SqueezeNet embeddings (512D)
- User metadata

Storage Structure:
    database/
    ├── metadata.json          # User registry
    └── users/
        └── {user_id}/
            ├── raw_images/    # {sample_id}.enc
            ├── svm_features/  # {sample_id}.enc
            └── squeezenet_embeddings/  # {sample_id}.enc

The design covers the key requirements of a biometric database:
- Encryption (privacy protection)
- Indexing (fast lookup)
- Backup/recovery (prevent data loss)
- Access control (security)
"""

import os
import json
import uuid
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from config import DATABASE_ROOT, SVM_FEATURE_DIM, SQUEEZENET_FEATURE_DIM, VERBOSE_MODE
from utils.encryption import encrypt_features, decrypt_features
from layers.data_acquisition import save_encrypted_image, load_encrypted_image


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def get_metadata_path() -> str:
    """Get path to metadata file."""
    return os.path.join(DATABASE_ROOT, "metadata.json")


def load_metadata() -> dict:
    """
    Load database metadata.
    
    Returns:
        Dictionary mapping user_id -> user info
        {
            "user_123": {
                "name": "John Doe",
                "num_samples": 3,
                "enrollment_date": "2026-01-18"
            }
        }
    """
    metadata_path = get_metadata_path()
    
    if not os.path.exists(metadata_path):
        return {}
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        if VERBOSE_MODE:
            print(f"[Database] Warning: Could not load metadata: {e}")
        return {}


def save_metadata(metadata: dict):
    """
    Save database metadata.
    
    Args:
        metadata: Dictionary of user information
    """
    metadata_path = get_metadata_path()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        raise Exception(f"Failed to save metadata: {e}")


# ============================================================================
# USER ENROLLMENT
# ============================================================================

def enroll_user(
    name: str,
    images: List[Image.Image],
    svm_features_list: List[np.ndarray],
    squeezenet_features_list: List[np.ndarray]
) -> Tuple[bool, str]:
    """
    Enroll a new user in the biometric database.
    
    This is the core enrollment function. It:
    1. Creates a unique user ID
    2. Stores encrypted images
    3. Stores encrypted feature vectors
    4. Updates metadata
    
    Args:
        name: User's name (already validated)
        images: List of PIL Images (handwriting samples)
        svm_features_list: List of SVM feature vectors (42D each)
        squeezenet_features_list: List of SqueezeNet features (512D each)
        
    Returns:
        Tuple of (success, user_id or error_message)
        
    Note:
        In production systems, enrollment typically requires:
        - Multiple samples (3-5) for robustness
        - Quality checks (reject poor samples)
        - Liveness detection (prevent photo attacks)
        - Duplicate detection (prevent re-enrollment)
    """
    if VERBOSE_MODE:
        print(f"\n[Database] Enrolling new user: {name}")
        print(f"[Database] Number of samples: {len(images)}")
    
    # Validate inputs
    if len(images) == 0:
        return False, "No images provided for enrollment"
    
    if len(images) != len(svm_features_list) or len(images) != len(squeezenet_features_list):
        return False, "Mismatch between number of images and features"
    
    # Generate unique user ID
    user_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
    
    if VERBOSE_MODE:
        print(f"[Database] Generated user ID: {user_id}")
    
    try:
        # Create user directories
        user_dir = os.path.join(DATABASE_ROOT, "users", user_id)
        os.makedirs(os.path.join(user_dir, "raw_images"), exist_ok=True)
        os.makedirs(os.path.join(user_dir, "svm_features"), exist_ok=True)
        os.makedirs(os.path.join(user_dir, "squeezenet_embeddings"), exist_ok=True)
        
        # Store each sample
        for idx, (image, svm_feat, sqz_feat) in enumerate(zip(images, svm_features_list, squeezenet_features_list)):
            sample_id = f"sample_{idx:03d}"
            
            if VERBOSE_MODE:
                print(f"[Database]   Storing {sample_id}...")
            
            # Save encrypted image
            success, msg = save_encrypted_image(image, user_id, sample_id, DATABASE_ROOT)
            if not success:
                raise Exception(f"Failed to save image {sample_id}: {msg}")
            
            # Save encrypted SVM features
            save_encrypted_svm_features(user_id, sample_id, svm_feat)
            
            # Save encrypted SqueezeNet features
            save_encrypted_squeezenet_features(user_id, sample_id, sqz_feat)
        
        # Update metadata
        metadata = load_metadata()
        metadata[user_id] = {
            "name": name,
            "num_samples": len(images),
            "enrollment_date": get_current_date()
        }
        save_metadata(metadata)
        
        if VERBOSE_MODE:
            print(f"[Database] OK User enrolled successfully: {user_id}")
        
        return True, user_id
    
    except Exception as e:
        # Rollback: delete user directory if enrollment failed
        try:
            import shutil
            user_dir = os.path.join(DATABASE_ROOT, "users", user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
        except:
            pass
        
        return False, f"Enrollment failed: {str(e)}"


# ============================================================================
# FEATURE STORAGE & RETRIEVAL
# ============================================================================

def save_encrypted_svm_features(user_id: str, sample_id: str, features: np.ndarray):
    """Save encrypted SVM features."""
    if VERBOSE_MODE:
        print(f"[Database]     Encrypting SVM features ({len(features)}D)...")
    
    # Encrypt
    encrypted = encrypt_features(features)
    
    # Save
    output_path = os.path.join(
        DATABASE_ROOT, "users", user_id, "svm_features", f"{sample_id}.enc"
    )
    with open(output_path, 'wb') as f:
        f.write(encrypted)


def load_encrypted_svm_features(user_id: str, sample_id: str) -> np.ndarray:
    """Load and decrypt SVM features."""
    input_path = os.path.join(
        DATABASE_ROOT, "users", user_id, "svm_features", f"{sample_id}.enc"
    )
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"SVM features not found: {input_path}")
    
    with open(input_path, 'rb') as f:
        encrypted = f.read()
    
    # Decrypt
    return decrypt_features(encrypted, SVM_FEATURE_DIM)


def save_encrypted_squeezenet_features(user_id: str, sample_id: str, features: np.ndarray):
    """Save encrypted SqueezeNet features."""
    if VERBOSE_MODE:
        print(f"[Database]     Encrypting SqueezeNet embeddings ({len(features)}D)...")
    
    # Encrypt
    encrypted = encrypt_features(features)
    
    # Save
    output_path = os.path.join(
        DATABASE_ROOT, "users", user_id, "squeezenet_embeddings", f"{sample_id}.enc"
    )
    with open(output_path, 'wb') as f:
        f.write(encrypted)


def load_encrypted_squeezenet_features(user_id: str, sample_id: str) -> np.ndarray:
    """Load and decrypt SqueezeNet features."""
    input_path = os.path.join(
        DATABASE_ROOT, "users", user_id, "squeezenet_embeddings", f"{sample_id}.enc"
    )
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"SqueezeNet features not found: {input_path}")
    
    with open(input_path, 'rb') as f:
        encrypted = f.read()
    
    # Decrypt
    return decrypt_features(encrypted, SQUEEZENET_FEATURE_DIM)


# ============================================================================
# USER QUERIES
# ============================================================================

def get_all_users() -> List[Dict]:
    """
    Get list of all enrolled users.
    
    Returns:
        List of user dictionaries with keys: user_id, name, num_samples, enrollment_date
    """
    metadata = load_metadata()
    
    users = []
    for user_id, info in metadata.items():
        users.append({
            "user_id": user_id,
            "name": info["name"],
            "num_samples": info["num_samples"],
            "enrollment_date": info.get("enrollment_date", "Unknown")
        })
    
    return users


def get_user_features(user_id: str, feature_type: str) -> List[np.ndarray]:
    """
    Load all feature vectors for a user.
    
    Args:
        user_id: Unique user identifier
        feature_type: "svm" or "squeezenet"
        
    Returns:
        List of feature vectors (numpy arrays)
        
    Note:
        For matching, we often average multiple samples per user.
        This creates a more robust template that reduces the impact
        of sample quality variations.
    """
    if feature_type == "svm":
        features_dir = os.path.join(DATABASE_ROOT, "users", user_id, "svm_features")
        load_func = load_encrypted_svm_features
    elif feature_type == "squeezenet":
        features_dir = os.path.join(DATABASE_ROOT, "users", user_id, "squeezenet_embeddings")
        load_func = load_encrypted_squeezenet_features
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    
    if not os.path.exists(features_dir):
        return []
    
    features = []
    sample_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.enc')])
    
    for filename in sample_files:
        sample_id = filename[:-4]  # Remove .enc
        try:
            feat = load_func(user_id, sample_id)
            features.append(feat)
        except Exception as e:
            if VERBOSE_MODE:
                print(f"[Database] Warning: Could not load {filename}: {e}")
    
    return features


def delete_user(user_id: str) -> bool:
    """
    Delete a user from the database.
    
    Args:
        user_id: User to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Delete user directory
        import shutil
        user_dir = os.path.join(DATABASE_ROOT, "users", user_id)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        
        # Update metadata
        metadata = load_metadata()
        if user_id in metadata:
            del metadata[user_id]
            save_metadata(metadata)
        
        if VERBOSE_MODE:
            print(f"[Database] OK User {user_id} deleted")
        
        return True
    
    except Exception as e:
        if VERBOSE_MODE:
            print(f"[Database] Error deleting user: {e}")
        return False


def get_database_stats() -> Dict:
    """
    Get database statistics for display.
    
    Returns:
        Dictionary with stats: total_users, total_samples, avg_samples_per_user
    """
    metadata = load_metadata()
    
    total_users = len(metadata)
    total_samples = sum(info["num_samples"] for info in metadata.values())
    avg_samples = total_samples / total_users if total_users > 0 else 0
    
    return {
        "total_users": total_users,
        "total_samples": total_samples,
        "avg_samples_per_user": avg_samples
    }


# ============================================================================
# HELPERS
# ============================================================================

def get_current_date() -> str:
    """Get current date as string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
