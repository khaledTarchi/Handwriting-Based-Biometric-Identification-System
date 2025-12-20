"""
Decision & Similarity Layer
============================
Layer 5 of the HB-BIS system

This layer implements the "brain" of the biometric system:
- Computing similarity between feature vectors
- Making accept/reject/uncertain decisions based on thresholds
- Detecting suspicious similarity during enrollment

It is built around the fundamental trade-off in biometrics:

FALSE ACCEPTANCE RATE (FAR):
    Probability of accepting an imposter as genuine
    → Security risk!

FALSE REJECTION RATE (FRR):
    Probability of rejecting a genuine user
    → Usability problem!

Adjusting the threshold moves the trade-off:
- Lower threshold → Higher security (more FRR, less FAR)
- Higher threshold → More convenient (less FRR, more FAR)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from config import (
    SVM_THRESHOLD_ACCEPT,
    SVM_THRESHOLD_REJECT,
    SQUEEZENET_THRESHOLD_ACCEPT,
    SQUEEZENET_THRESHOLD_REJECT,
    SIMILARITY_WARNING_THRESHOLD,
    VERBOSE_MODE
)
from layers.database import get_all_users, get_user_features


# ============================================================================
# DISTANCE METRICS
# ============================================================================

def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.
    
    Cosine distance = 1 - cosine similarity
    Range: [0, 2]
    - 0 = identical vectors (same direction)
    - 1 = orthogonal vectors
    - 2 = opposite vectors
    
    Formula:
        distance = 1 - (vec1 · vec2) / (||vec1|| * ||vec2||)
    
    Note:
        Cosine distance measures the ANGLE between vectors, not magnitude.
        This is good for comparing distributions or normalized features.
        
        Example: [1, 0] and [2, 0] have cosine distance = 0 (same direction)
                 even though Euclidean distance = 1
    
    Args:
        vec1, vec2: Feature vectors (should be same dimensionality)
        
    Returns:
        Cosine distance (lower = more similar)
    """
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle zero vectors
    if norm1 == 0 or norm2 == 0:
        return 2.0  # Maximum distance
    
    # Cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)
    
    # Clip to valid range [-1, 1] (numerical stability)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
    
    # Convert to distance
    distance = 1.0 - cosine_sim
    
    return float(distance)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Formula:
        distance = ||vec1 - vec2|| = sqrt(sum((vec1_i - vec2_i)^2))
    
    Note:
        Euclidean distance measures straight-line distance in feature space.
        Sensitive to both direction AND magnitude differences.
        
        For high-dimensional spaces, distances tend to become uniform
        (curse of dimensionality), so cosine often works better.
    
    Args:
        vec1, vec2: Feature vectors
        
    Returns:
        Euclidean distance (lower = more similar)
    """
    return float(np.linalg.norm(vec1 - vec2))


def normalized_euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Normalized Euclidean distance (divided by sqrt of dimensionality).
    
    This normalization helps compare distances across different feature dimensions.
    """
    euclidean = euclidean_distance(vec1, vec2)
    normalized = euclidean / np.sqrt(len(vec1))
    return float(normalized)


# ============================================================================
# DECISION LOGIC
# ============================================================================

def make_decision(distance: float, threshold_accept: float, threshold_reject: float) -> Tuple[str, float]:
    """
    Make biometric decision based on distance and thresholds.
    
    Decision zones:
    - distance < θ_accept → MATCH (high confidence)
    - θ_accept ≤ distance < θ_reject → UNCERTAIN (manual review needed)
    - distance ≥ θ_reject → UNKNOWN (reject)
    
    Args:
        distance: Computed distance
        threshold_accept: Accept threshold
        threshold_reject: Reject threshold
        
    Returns:
        Tuple of (decision, confidence_score)
        decision: "MATCH", "UNCERTAIN", or "UNKNOWN"
        confidence_score: 0-100 percentage
        
    Note:
        The three-zone decision allows for manual review of uncertain cases.
        Production systems often have:
        - Fully automated: MATCH/UNKNOWN only
        - Semi-automated: UNCERTAIN cases go to human operator
        - Paranoid mode: All matches require human confirmation
    """
    if distance < threshold_accept:
        # Clear match - high confidence
        # Confidence decreases as distance approaches threshold
        confidence = 100.0 * (1.0 - distance / threshold_accept)
        return "MATCH", min(confidence, 100.0)
    
    elif distance < threshold_reject:
        # Uncertain zone - medium confidence
        # Linearly interpolate between accept and reject thresholds
        range_width = threshold_reject - threshold_accept
        distance_from_accept = distance - threshold_accept
        confidence = 50.0 * (1.0 - distance_from_accept / range_width)
        return "UNCERTAIN", max(min(confidence, 50.0), 0.0)
    
    else:
        # Clear reject - low confidence in match
        # Confidence decreases as distance increases beyond reject threshold
        confidence = max(0.0, 10.0 - distance * 5.0)  # Quickly drops to 0
        return "UNKNOWN", max(confidence, 0.0)


# ============================================================================
# IDENTIFICATION
# ============================================================================

def identify_user(
    query_features: np.ndarray,
    model_type: str
) -> Tuple[Optional[str], Optional[str], float, str, float]:
    """
    Identify a user by comparing query features against all enrolled users.
    
    This performs 1:N matching (one query against N enrolled users).
    
    Args:
        query_features: Feature vector to identify
        model_type: "svm" or "squeezenet"
        
    Returns:
        Tuple of (user_id, user_name, distance, decision, confidence)
        - user_id: ID of best match (None if no acceptable match)
        - user_name: Name of best match (None if no match)
        - distance: Distance to best match
        - decision: "MATCH", "UNCERTAIN", or "UNKNOWN"
        - confidence: Confidence score (0-100)
        
    Note:
        Identification (1:N) is harder than verification (1:1):
        - More comparisons → higher chance of false match
        - FAR multiplies by number of users
        - Need stricter thresholds for large databases
        
        Production systems use indexing (e.g., LSH) to avoid comparing
        against all N users.
    """
    if VERBOSE_MODE:
        print(f"\n[Decision] Starting identification with {model_type} model...")
    
    # Get thresholds
    if model_type == "svm":
        threshold_accept = SVM_THRESHOLD_ACCEPT
        threshold_reject = SVM_THRESHOLD_REJECT
    elif model_type == "squeezenet":
        threshold_accept = SQUEEZENET_THRESHOLD_ACCEPT
        threshold_reject = SQUEEZENET_THRESHOLD_REJECT
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # Get all users
    users = get_all_users()
    
    if len(users) == 0:
        if VERBOSE_MODE:
            print("[Decision] ✗ No users enrolled in database")
        return None, None, float('inf'), "UNKNOWN", 0.0
    
    # Compare against all users
    best_match = None
    best_distance = float('inf')
    
    for user in users:
        user_id = user["user_id"]
        user_name = user["name"]
        
        # Load all features for this user
        user_features_list = get_user_features(user_id, model_type)
        
        if len(user_features_list) == 0:
            continue
        
        # Compute average template (more robust than single sample)
        avg_template = np.mean(user_features_list, axis=0)
        
        # Compute distance
        distance = cosine_distance(query_features, avg_template)
        
        if VERBOSE_MODE:
            print(f"[Decision]   User '{user_name}': distance = {distance:.4f}")
        
        # Track best match
        if distance < best_distance:
            best_distance = distance
            best_match = (user_id, user_name)
    
    # Make decision based on best distance
    decision, confidence = make_decision(best_distance, threshold_accept, threshold_reject)
    
    if decision == "MATCH":
        if VERBOSE_MODE:
            print(f"[Decision] OK MATCH: {best_match[1]} (distance={best_distance:.4f}, conf={confidence:.1f}%)")
        return best_match[0], best_match[1], best_distance, decision, confidence
    
    elif decision == "UNCERTAIN":
        if VERBOSE_MODE:
            print(f"[Decision] ! UNCERTAIN: {best_match[1]} (distance={best_distance:.4f}, conf={confidence:.1f}%)")
        return best_match[0], best_match[1], best_distance, decision, confidence
    
    else:
        if VERBOSE_MODE:
            print(f"[Decision] ✗ UNKNOWN: No match found (best_distance={best_distance:.4f})")
        return None, None, best_distance, decision, confidence


def compare_with_database(
    query_features: np.ndarray,
    model_type: str
) -> List[Tuple[str, str, float]]:
    """
    Compare query features against all users and return sorted list.
    
    Useful for showing top-N matches in GUI.
    
    Args:
        query_features: Feature vector
        model_type: "svm" or "squeezenet"
        
    Returns:
        List of (user_id, user_name, distance) tuples, sorted by distance
    """
    users = get_all_users()
    results = []
    
    for user in users:
        user_id = user["user_id"]
        user_name = user["name"]
        
        user_features_list = get_user_features(user_id, model_type)
        
        if len(user_features_list) == 0:
            continue
        
        avg_template = np.mean(user_features_list, axis=0)
        distance = cosine_distance(query_features, avg_template)
        
        results.append((user_id, user_name, distance))
    
    # Sort by distance (ascending)
    results.sort(key=lambda x: x[2])
    
    return results


# ============================================================================
# ENROLLMENT SIMILARITY WARNING
# ============================================================================

def check_similarity_warning(
    new_features: np.ndarray,
    model_type: str
) -> Tuple[bool, Optional[str], float]:
    """
    Check if new enrollment features are suspiciously similar to existing users.
    
    This detects potential:
    - Duplicate enrollment (same person enrolling twice)
    - Impersonation attempt (features too similar to another user)
    
    Args:
        new_features: Features of new user to enroll
        model_type: "svm" or "squeezenet"
        
    Returns:
        Tuple of (has_warning, similar_user_name, distance)
        
    Note:
        Biometric systems should prevent duplicate enrollments!
        If someone enrolls twice, the system has two templates for one person,
        which can confuse matching and violate privacy regulations.
    """
    users = get_all_users()
    
    min_distance = float('inf')
    most_similar_user = None
    
    for user in users:
        user_id = user["user_id"]
        user_name = user["name"]
        
        user_features_list = get_user_features(user_id, model_type)
        
        if len(user_features_list) == 0:
            continue
        
        avg_template = np.mean(user_features_list, axis=0)
        distance = cosine_distance(new_features, avg_template)
        
        if distance < min_distance:
            min_distance = distance
            most_similar_user = user_name
    
    # Check if suspiciously similar
    if min_distance < SIMILARITY_WARNING_THRESHOLD:
        return True, most_similar_user, min_distance
    else:
        return False, None, min_distance
