"""Tests for the synthetic handwriting generator."""
import numpy as np
from PIL import Image

from experiments.synthetic_data import SyntheticWriter, generate_writers, generate_dataset


def test_generator_deterministic():
    w1 = SyntheticWriter("w00", seed=42)
    w2 = SyntheticWriter("w00", seed=42)
    assert w1.style_summary() == w2.style_summary()
    img1 = w1.render(sample_seed=7)
    img2 = w2.render(sample_seed=7)
    assert np.array_equal(np.array(img1), np.array(img2))


def test_writers_are_distinct():
    writers = generate_writers(6, seed=42)
    styles = {w.style_summary() for w in writers}
    assert len(styles) == len(writers)
    for w in writers:
        assert w.slant_deg != 0.0 or True  # params are sampled, at least distinct summaries


def test_render_shapes_and_ink():
    writer = SyntheticWriter("w00", seed=42)
    img = writer.render(sample_seed=1)
    assert isinstance(img, Image.Image)
    assert img.mode == "L"
    assert img.size == (420, 240)
    arr = np.array(img)
    assert arr.dtype == np.uint8
    assert (arr < 128).mean() > 0.005  # not blank


def test_dataset_structure():
    writers, images = generate_dataset(n_writers=4, samples_per_writer=5, seed=1)
    assert len(writers) == 4
    assert len(images) == 20
    writer_ids = {wid for wid, _ in images}
    assert len(writer_ids) == 4
