"""
SqueezeNet Model Wrapper
=========================

This module wraps the SqueezeNet feature extractor with optional retraining
capability using triplet loss.

Transfer learning and fine-tuning workflow:
- Start with a pre-trained model (ImageNet)
- Use it as a feature extractor initially
- Optionally fine-tune with triplet loss for better discrimination
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from PIL import Image
from config import (
    MODELS_ROOT,
    TRIPLET_MARGIN,
    LEARNING_RATE,
    RETRAIN_EPOCHS,
    BATCH_SIZE,
    MIN_USERS_FOR_RETRAIN,
    MIN_SAMPLES_FOR_RETRAIN,
    VERBOSE_MODE
)
from layers.feature_engineering import SqueezeNetFeatureExtractor


# ============================================================================
# TRIPLET LOSS DATASET
# ============================================================================

class TripletDataset(Dataset):
    """
    Dataset for triplet loss training.
    
    Each sample is a triplet: (anchor, positive, negative)
    - Anchor: User's handwriting sample
    - Positive: Different sample from same user
    - Negative: Sample from different user
    
    Note:
        Triplet loss trains the network to make:
        distance(anchor, positive) < distance(anchor, negative)
        
        This forces similar samples (same writer) to cluster together
        and dissimilar samples (different writers) to spread apart.
    """
    
    def __init__(self, users_data: List[Tuple[str, List[np.ndarray]]]):
        """
        Args:
            users_data: List of (user_id, list_of_images)
                where each image is a preprocessed numpy array
        """
        self.users_data = users_data
        self.user_indices = list(range(len(users_data)))
    
    def __len__(self):
        # Generate multiple triplets per epoch
        return len(self.users_data) * 10
    
    def __getitem__(self, idx):
        """Generate a random triplet."""
        # Pick a random user as anchor
        anchor_user_idx = np.random.choice(self.user_indices)
        user_id, images = self.users_data[anchor_user_idx]
        
        # Need at least 2 samples for anchor + positive
        if len(images) < 2:
            # Fallback: use same image for anchor and positive
            anchor_img = positive_img = images[0]
        else:
            # Pick two different samples from same user
            anchor_idx, positive_idx = np.random.choice(len(images), size=2, replace=False)
            anchor_img = images[anchor_idx]
            positive_img = images[positive_idx]
        
        # Pick different user for negative
        negative_user_idx = np.random.choice(
            [i for i in self.user_indices if i != anchor_user_idx]
        )
        negative_images = self.users_data[negative_user_idx][1]
        negative_img = negative_images[np.random.randint(len(negative_images))]
        
        # Convert to tensors (C, H, W)
        anchor_tensor = torch.from_numpy(anchor_img).permute(2, 0, 1)
        positive_tensor = torch.from_numpy(positive_img).permute(2, 0, 1)
        negative_tensor = torch.from_numpy(negative_img).permute(2, 0, 1)
        
        return anchor_tensor, positive_tensor, negative_tensor


# ============================================================================
# TRIPLET LOSS
# ============================================================================

class TripletLoss(nn.Module):
    """
    Triplet loss with margin.
    
    Loss = max(0, d(a,p) - d(a,n) + margin)
    
    Where:
    - d(a,p) = distance between anchor and positive
    - d(a,n) = distance between anchor and negative
    - margin = minimum separation required
    
    Note:
        The margin enforces that negatives must be at least 'margin'
        further than positives. This creates a "safety zone" for
        discriminating between users.
    """
    
    def __init__(self, margin=TRIPLET_MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor, positive, negative: Feature embeddings (batch_size, feature_dim)
        """
        # Compute distances (Euclidean)
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def retrain_squeezenet(users_data: List[Tuple[str, List[np.ndarray]]], base_model=None) -> Tuple[bool, str]:
    """
    Fine-tune SqueezeNet using triplet loss.

    Args:
        users_data: List of (user_id, [preprocessed_images])
        base_model: Optional SqueezeNetFeatureExtractor to start from.
            If None, a fresh ImageNet-pretrained model is used.

    Returns:
        Tuple of (success, message)

    Note:
        Fine-tuning adjusts the pre-trained weights to better discriminate
        between our specific users. This is more data-efficient than training
        from scratch, but still requires multiple users with multiple samples.
    """
    if VERBOSE_MODE:
        print("\n[SqueezeNet] Starting retraining with triplet loss...")
    
    # Validate data requirements
    num_users = len(users_data)
    if num_users < MIN_USERS_FOR_RETRAIN:
        return False, f"Need at least {MIN_USERS_FOR_RETRAIN} users for retraining (have {num_users})"
    
    # Check samples per user
    for user_id, images in users_data:
        if len(images) < MIN_SAMPLES_FOR_RETRAIN:
            return False, f"User {user_id} has only {len(images)} samples (need {MIN_SAMPLES_FOR_RETRAIN})"
    
    # Create dataset and loader
    dataset = TripletDataset(users_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Get model
    model = base_model if base_model is not None else SqueezeNetFeatureExtractor()
    model.train()  # Set to training mode
    
    # Loss and optimizer
    criterion = TripletLoss(margin=TRIPLET_MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if VERBOSE_MODE:
        print(f"[SqueezeNet] Training for {RETRAIN_EPOCHS} epochs with {len(dataset)} triplets...")
    
    # Training loop
    for epoch in range(RETRAIN_EPOCHS):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            # Extract features
            anchor_feat = model(anchor)
            positive_feat = model(positive)
            negative_feat = model(negative)
            
            # Compute loss
            loss = criterion(anchor_feat, positive_feat, negative_feat)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if VERBOSE_MODE:
            print(f"[SqueezeNet]   Epoch {epoch+1}/{RETRAIN_EPOCHS}: Loss = {avg_loss:.4f}")
    
    # Save fine-tuned model
    model.eval()
    save_path = os.path.join(MODELS_ROOT, "squeezenet_finetuned.pth")
    os.makedirs(MODELS_ROOT, exist_ok=True)
    
    try:
        torch.save(model.state_dict(), save_path)
        if VERBOSE_MODE:
            print(f"[SqueezeNet] ✓ Fine-tuned model saved: {save_path}")
        return True, "Retraining successful"
    except Exception as e:
        return False, f"Failed to save model: {e}"
