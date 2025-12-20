"""
Fine-tuning experiment: ImageNet-pretrained vs. triplet-loss fine-tuned
SqueezeNet embeddings.

Reuses the EXACT triplet dataset/loss of the core system
(models.squeezenet_model.TripletDataset / TripletLoss) so the algorithm is
unchanged; only the training orchestration and evaluation are added.

Protocol:
    * N_WRITERS synthetic writers are split into a train set (fine-tuning)
      and a held-out set (evaluation).
    * The held-out set is NEVER seen during fine-tuning.
    * rank-1 identification is measured with the pretrained embeddings and
      again with the fine-tuned embeddings on the held-out writers.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from layers.preprocessing import preprocess_for_squeezenet
from layers.feature_engineering import SqueezeNetFeatureExtractor
from models.squeezenet_model import TripletDataset, TripletLoss


def preprocess_images(images):
    """Convert PIL images to the normalized RGB tensor input used by the core model."""
    return [preprocess_for_squeezenet(im) for im in images]


def train_triplet(
    train_writers,
    epochs,
    batch_size,
    learning_rate,
    margin,
    base_model=None,
    verbose=False,
):
    """
    Fine-tune SqueezeNet with triplet loss on the given writers.

    Args:
        train_writers: list of (writer_id, [PIL images])
        ...
        base_model: optional starting model (pretrained by default)

    Returns:
        (trained_model, loss_history_per_epoch)
    """
    users_data = [(wid, preprocess_images(imgs)) for wid, imgs in train_writers]

    dataset = TripletDataset(users_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = base_model if base_model is not None else SqueezeNetFeatureExtractor()
    model.train()

    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
    for epoch in range(epochs):
        epoch_losses = []
        for anchor, positive, negative in loader:
            anchor_feat = model(anchor)
            positive_feat = model(positive)
            negative_feat = model(negative)
            loss = criterion(anchor_feat, positive_feat, negative_feat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        if verbose:
            print(f"[Finetune] Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")

    model.eval()
    return model, loss_history
