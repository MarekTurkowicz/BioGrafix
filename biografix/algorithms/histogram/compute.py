"""Luminance histogram."""

from ._common import _luma


def compute_histogram(pixels):
    """Return a 256-bin luminance histogram of ``pixels``."""
    hist = [0] * 256
    for r, g, b in pixels:
        y = _luma(r, g, b)
        hist[y] += 1
    return hist
