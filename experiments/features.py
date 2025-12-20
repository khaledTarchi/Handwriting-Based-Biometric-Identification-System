"""
Batched feature-extraction pipeline.

Wraps the EXISTING preprocessing + feature extraction algorithms unchanged;
only the inference is vectorized (batched) so experiments are fast enough to
run on a CPU-only machine. The per-image computation is identical to the
single-image path used by the GUI.
"""

import os
import numpy as np
import torch

from layers.preprocessing import preprocess_for_svm, preprocess_for_squeezenet
from layers.feature_engineering import (
    extract_svm_features,
    SqueezeNetFeatureExtractor,
    get_squeezenet_model,
    FINETUNED_WEIGHTS_PATH,
)


class FeaturePipeline:
    """Extract features for a given model ("svm" or "squeezenet")."""

    def __init__(self, model_type, weights="pretrained", model=None):
        self.model_type = model_type
        self._model = model
        if model_type == "squeezenet" and self._model is None:
            if weights == "auto":
                self._model = get_squeezenet_model(weights="auto")
            elif weights == "finetuned":
                self._model = SqueezeNetFeatureExtractor()
                self._model.load_state_dict(
                    torch.load(FINETUNED_WEIGHTS_PATH, map_location="cpu", weights_only=True)
                )
            else:  # "pretrained" (default)
                self._model = SqueezeNetFeatureExtractor()

    # ------------------------------------------------------------------
    def extract(self, images):
        """Extract features for a list of PIL images (batched)."""
        if self.model_type == "svm":
            feats = [extract_svm_features(preprocess_for_svm(im)) for im in images]
            return np.stack(feats)

        preprocessed = [preprocess_for_squeezenet(im) for im in images]
        batch = torch.from_numpy(np.stack(preprocessed)).permute(0, 3, 1, 2)
        with torch.no_grad():
            out = self._model(batch).numpy()
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / (norms + 1e-9)
        return out

    def extract_one(self, image):
        """Extract features for a single PIL image (same algorithm)."""
        return self.extract([image])[0]

    def parameter_count(self):
        if self._model is None:
            return None
        return sum(p.numel() for p in self._model.parameters())
