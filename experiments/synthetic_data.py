"""
Synthetic Handwriting Dataset Generator
=======================================
Procedurally generates distinct synthetic "writers". Each writer is defined by
a set of interpretable style parameters (slant, stroke width, character size,
spacing, curvature, baseline wobble) plus a private set of random glyphs, so
that:

  * same writer, different draws  -> similar but not identical images
                                     (within-class / intra-writer variability)
  * different writers             -> visibly different handwriting
                                     (between-class / inter-writer variability)

Ground-truth identity is known by construction, which makes this benchmark
fully labelled, anonymized and reproducible (seeded).

IMPORTANT (honesty note): this is a SYNTHETIC benchmark, not a corpus of human
handwriting. It is used to validate the evaluation protocol, the preprocessing
pipeline and the comparison methodology. Results on synthetic data do not
transfer unchanged to real-world handwriting.
"""

import math
import numpy as np
from PIL import Image, ImageDraw

# Fixed text for all writers (text-dependent protocol: same content, different hands)
DEFAULT_TEXT = "the quick brown fox jumps over the lazy dog and its companion"

# Rendering canvas size. Kept modest so that strokes remain visible after the
# pipeline resizes images to the 224x224 target.
CANVAS_SIZE = (420, 240)

# Magnitude of per-draw endpoint jitter (models intra-writer variability).
# Chosen so the benchmark is challenging but informative: the classical
# (handcrafted) pipeline reaches ~0.95 rank-1 while deep embeddings reach
# ~1.0, leaving room for meaningful robustness and ablation analysis.
JITTER = 0.15


class SyntheticWriter:
    """A synthetic handwriting identity defined by style parameters + glyphs."""

    def __init__(self, writer_id, seed, text=DEFAULT_TEXT, canvas_size=CANVAS_SIZE):
        rng = np.random.RandomState(int(seed))
        self.writer_id = str(writer_id)
        self.text = text
        self.canvas_size = canvas_size
        self.seed = int(seed)

        # ---- Interpretable style parameters (each controls one handwriting trait)
        self.slant_deg = float(rng.uniform(-25.0, 25.0))      # italic slant
        self.stroke_width = float(rng.uniform(5.0, 9.0))      # pen thickness
        self.char_height = float(rng.uniform(34.0, 60.0))     # writing size
        self.char_spacing = float(rng.uniform(6.0, 16.0))     # letter density
        self.word_spacing = float(rng.uniform(14.0, 32.0))    # word density
        self.curvature = float(rng.uniform(0.0, 1.0))         # 0 angular -> 1 loopy
        self.baseline_wiggle = float(rng.uniform(0.0, 8.0))   # baseline wobble

        # ---- Per-writer glyph set (the actual letter shapes)
        glyph_rng = np.random.RandomState(int(seed) + 1000)
        self._glyphs = self._make_glyphs(glyph_rng)

    # ------------------------------------------------------------------
    # Glyph construction
    # ------------------------------------------------------------------
    def _make_glyphs(self, rng):
        glyphs = {}
        for ch in sorted(set(self.text.replace(" ", ""))):
            glyphs[ch] = self._make_glyph(rng)
        return glyphs

    def _make_glyph(self, rng):
        """A glyph is a list of quadratic-Bezier strokes in a unit cell.

        Each stroke: (x0, y0, x1, y1, cx, cy) with control point chosen
        perpendicular to the chord; perpendicular offset scales with the
        writer's curvature so loopy writers bend more.
        """
        n_strokes = int(rng.randint(2, 4))
        strokes = []
        for _ in range(n_strokes):
            x0 = float(rng.uniform(0.0, 0.25))
            y0 = float(rng.uniform(0.05, 0.95))
            x1 = float(rng.uniform(0.60, 1.0))
            y1 = float(rng.uniform(0.05, 0.95))
            dx, dy = x1 - x0, y1 - y0
            length = max(math.hypot(dx, dy), 1e-6)
            px, py = -dy / length, dx / length
            offset = float(rng.uniform(-0.35, 0.35)) * (0.4 + 0.6 * self.curvature)
            cx = (x0 + x1) / 2.0 + px * offset
            cy = (y0 + y1) / 2.0 + py * offset
            strokes.append((x0, y0, x1, y1, cx, cy))
        return strokes

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, sample_seed=None):
        """Render one handwriting sample of this writer.

        Args:
            sample_seed: per-sample RNG seed controlling the small drawing
                jitter. Fixed glyphs + sampled jitter model intra-writer
                variability (a person re-writing the same text).

        Returns:
            PIL grayscale image (writer's handwriting on white paper).
        """
        rng = np.random.RandomState(
            int(sample_seed) if sample_seed is not None else self.seed
        )
        width, height = self.canvas_size
        image = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(image)

        slant_tan = math.tan(math.radians(self.slant_deg))
        char_w = self.char_height * 0.55
        line_h = self.char_height * 1.40
        margin = self.stroke_width * 2.5

        x = margin
        y = margin + self.char_height  # baseline of the current line

        for word in self.text.split():
            word_w = sum(char_w + self.char_spacing for _ in word)
            if x + word_w + self.word_spacing > width - margin and x > margin:
                x = margin
                y += line_h
            for ch in word:
                self._draw_char(draw, rng, ch, x, y, slant_tan, char_w)
                x += char_w + self.char_spacing
            x += self.word_spacing

        return image

    def _draw_char(self, draw, rng, ch, x0, baseline_y, slant_tan, char_w):
        glyph = self._glyphs[ch]
        wiggle = float(rng.uniform(-self.baseline_wiggle, self.baseline_wiggle))
        ch_h = self.char_height
        for (sx, sy, ex, ey, cx, cy) in glyph:
            # small per-draw endpoint jitter (intra-writer variability)
            j1 = float(rng.uniform(-JITTER, JITTER))
            j2 = float(rng.uniform(-JITTER, JITTER))
            # shear for slant: x' = x + y * tan(slant)
            p0 = (x0 + (sx + j1) * char_w + sy * ch_h * slant_tan,
                  baseline_y + wiggle + sy * ch_h)
            p1 = (x0 + (ex + j2) * char_w + ey * ch_h * slant_tan,
                  baseline_y + wiggle + ey * ch_h)
            cm = (x0 + cx * char_w + cy * ch_h * slant_tan,
                  baseline_y + wiggle + cy * ch_h)
            self._bezier(draw, p0, p1, cm)

    def _bezier(self, draw, p0, p1, ctrl, steps=16):
        pts = []
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
            pts.append((x, y))
        draw.line(pts, fill=0, width=int(round(self.stroke_width)), joint="curve")

    def style_summary(self):
        """Human-readable description of this writer's style parameters."""
        return (
            f"slant={self.slant_deg:+.1f}deg stroke={self.stroke_width:.1f}px "
            f"size={self.char_height:.0f}px spacing={self.char_spacing:.0f}px "
            f"curvature={self.curvature:.2f} wiggle={self.baseline_wiggle:.1f}px"
        )


def generate_writers(n_writers, seed=42, text=DEFAULT_TEXT):
    """Create ``n_writers`` distinct synthetic writers (deterministic)."""
    writers = []
    for w in range(n_writers):
        writer_seed = int(seed) * 100 + w
        writers.append(SyntheticWriter(f"w{w:02d}", writer_seed, text=text))
    return writers


def generate_dataset(n_writers, samples_per_writer, seed=42):
    """Return (writers, images) where images is [(writer_id, PIL.Image), ...].

    Each writer renders ``samples_per_writer`` draws with distinct per-sample
    seeds, providing intra-writer variability.
    """
    writers = generate_writers(n_writers, seed=seed)
    images = []
    for writer in writers:
        for s in range(samples_per_writer):
            sample_seed = int(seed) * 1000 + int(writer.seed) * 10 + s
            images.append((writer.writer_id, writer.render(sample_seed=sample_seed)))
    return writers, images
