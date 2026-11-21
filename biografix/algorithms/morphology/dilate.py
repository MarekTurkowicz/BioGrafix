"""Binary dilation."""

import numpy as np

from ._common import element_offsets, neighbour


def dilate(mask: np.ndarray, se) -> np.ndarray:
    """A pixel becomes 1 if any element cell equal to 1 covers an object pixel."""
    out = np.zeros_like(mask, dtype=bool)
    for dy, dx in element_offsets(se, 1):
        out |= neighbour(mask, dy, dx)
    return out
