"""Filling the image with one colour."""

import numpy as np


def fill_color(image: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Fill the image with a single colour."""
    out = np.empty_like(image)
    out[:, :] = (r, g, b)
    return out
