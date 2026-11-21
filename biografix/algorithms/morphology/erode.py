"""Binary erosion."""

import numpy as np

from ._common import element_offsets, neighbour


def erode(mask: np.ndarray, se) -> np.ndarray:
    """A pixel stays 1 only if every element cell equal to 1 covers an object pixel inside the image."""
    out = np.ones_like(mask, dtype=bool)
    for dy, dx in element_offsets(se, 1):
        out &= neighbour(mask, dy, dx)
    return out
