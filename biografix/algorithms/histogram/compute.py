"""Luminance histogram."""

import numpy as np

from ._common import luminance_levels


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """256-bin luminance histogram."""
    return np.bincount(luminance_levels(image).ravel(), minlength=256)
