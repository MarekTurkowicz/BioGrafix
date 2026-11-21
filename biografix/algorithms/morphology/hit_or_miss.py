"""Hit-or-miss transform."""

import numpy as np

from ._common import element_offsets, neighbour


def hit_or_miss(mask: np.ndarray, se) -> np.ndarray:
    """Match where cells 1 cover object pixels and cells -1 cover background (0 = don't care).

    A cell 1 outside the image fails the match; a cell -1 outside the image counts as background.
    """
    out = np.ones_like(mask, dtype=bool)
    for dy, dx in element_offsets(se, 1):
        out &= neighbour(mask, dy, dx)
    for dy, dx in element_offsets(se, -1):
        out &= ~neighbour(mask, dy, dx)
    return out
