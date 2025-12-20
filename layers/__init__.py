"""
Layers package for HB-BIS
=========================

This package contains the 6 modular layers:
1. data_acquisition.py - Image loading and encryption
2. preprocessing.py - Enhancement, binarization, normalization
3. feature_engineering.py - SVM and SqueezeNet feature extraction
4. database.py - Encrypted storage and retrieval
5. decision.py - Similarity computation and decisions
6. gui.py - User interface
"""

__all__ = [
    'data_acquisition',
    'preprocessing',
    'feature_engineering',
    'database',
    'decision',
    'gui'
]
