"""
Dataset helpers for the experimental protocol.
"""

import numpy as np
from collections import defaultdict

from . import config
from .synthetic_data import generate_dataset


def build_benchmark():
    """Create the synthetic benchmark: (writers, images[(writer_id, PIL.Image)])."""
    return generate_dataset(
        n_writers=config.N_WRITERS,
        samples_per_writer=config.SAMPLES_PER_WRITER,
        seed=config.SEED,
    )


def sample_splits(images, n_enroll, repeat_seed):
    """
    Partition sample indices into enrollment (gallery) and probe (query) sets.

    This is a WRITER-DISJOINT split: enrollment and test samples for the same
    writer are distinct handwriting samples (no sample appears in both sets).

    Args:
        images: list of (writer_id, image)
        n_enroll: number of samples per writer reserved for enrollment
        repeat_seed: seed for this repetition (fixed -> reproducible)

    Returns:
        (enroll_indices, probe_indices)
    """
    by_writer = defaultdict(list)
    for idx, (writer_id, _image) in enumerate(images):
        by_writer[writer_id].append(idx)

    rng = np.random.RandomState(int(repeat_seed))
    enroll, probe = [], []
    for writer_id, indices in by_writer.items():
        perm = rng.permutation(indices)
        enroll.extend(list(perm[:n_enroll]))
        probe.extend(list(perm[n_enroll:]))
    return enroll, probe
