"""
HB-BIS Experimental Evaluation Framework
=========================================
A rigorous, reproducible evaluation layer built on top of the existing
HB-BIS pipeline. It does NOT modify any of the original algorithms - it
wraps the existing preprocessing, feature extraction and decision modules
with a formal experimental protocol (fixed seeds, repeated cross-validation,
robustness testing, ablations and metric learning comparison).

This package silences the verbose logging of the core pipeline while
experiments are running.
"""

import config as _config

# Keep experiment logs clean; core modules read VERBOSE_MODE at import time,
# so this must run before any `layers.*` module is imported.
_config.VERBOSE_MODE = False
