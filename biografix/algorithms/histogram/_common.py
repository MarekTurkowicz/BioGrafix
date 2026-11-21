"""Helpers shared by the histogram operations."""

import numpy as np

from ..point_ops import luminance_levels  # noqa: F401  (re-exported for the histogram modules)


def gray_image(levels: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 image with the given gray level in every channel."""
    return np.stack([levels.astype(np.uint8)] * 3, axis=-1)
