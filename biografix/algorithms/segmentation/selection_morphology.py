"""Growing and shrinking a selection mask with binary dilation and erosion.

The structuring element is a square, a cross or a disk of the given radius.
Pixels outside the image count as not selected, so erosion also shrinks a
selection away from the image border.
"""

from enum import StrEnum
from typing import assert_never

import numpy as np


class StructuringElementShape(StrEnum):
    SQUARE = "square"  # all pixels of the (2r+1) × (2r+1) block
    CROSS = "cross"    # only the middle row and column
    DISK = "disk"      # pixels within distance r of the centre


def structuring_element(radius: int, shape: StructuringElementShape = StructuringElementShape.SQUARE) -> np.ndarray:
    """(2r+1) × (2r+1) boolean structuring element centred on the middle pixel."""
    if radius < 0:
        raise ValueError("The radius must be >= 0.")
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    match shape:
        case StructuringElementShape.SQUARE:
            return np.ones(dy.shape, dtype=bool)
        case StructuringElementShape.CROSS:
            return (dx == 0) | (dy == 0)
        case StructuringElementShape.DISK:
            return dx * dx + dy * dy <= radius * radius
        case _:
            assert_never(shape)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``mask`` moved by (dx, dy); pixels uncovered by the move are False."""
    height, width = mask.shape
    shifted = np.zeros_like(mask)
    shifted[max(dy, 0):height + min(dy, 0), max(dx, 0):width + min(dx, 0)] = \
        mask[max(-dy, 0):height + min(-dy, 0), max(-dx, 0):width + min(-dx, 0)]
    return shifted


def element_offsets(radius: int, shape: StructuringElementShape) -> list[tuple[int, int]]:
    """(dx, dy) offsets of the structuring element pixels relative to its centre."""
    element = structuring_element(radius, shape)
    return [(int(col) - radius, int(row) - radius) for row, col in zip(*np.nonzero(element), strict=True)]


def dilate_selection(mask: np.ndarray, radius: int = 1,
                     shape: StructuringElementShape = StructuringElementShape.SQUARE) -> np.ndarray:
    """Dilation: a pixel becomes selected if the element centred on it touches the selection."""
    grown = np.zeros_like(mask, dtype=bool)
    for dx, dy in element_offsets(radius, shape):
        grown |= shift_mask(mask, dx, dy)
    return grown


def erode_selection(mask: np.ndarray, radius: int = 1,
                    shape: StructuringElementShape = StructuringElementShape.SQUARE) -> np.ndarray:
    """Erosion: a pixel stays selected only if the whole element centred on it is selected."""
    shrunk = np.ones_like(mask, dtype=bool)
    for dx, dy in element_offsets(radius, shape):
        shrunk &= shift_mask(mask, -dx, -dy)
    return shrunk
