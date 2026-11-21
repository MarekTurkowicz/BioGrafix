"""Conversion between RGB images and binary masks, and access to shifted neighbours."""

import numpy as np

from ..segmentation.selection_morphology import shift_mask


def image_to_binary(image: np.ndarray) -> np.ndarray:
    """Binary mask: True where the channel average (R+G+B)/3 is at least 128."""
    return np.asarray(image)[:, :, :3].astype(np.int32).sum(axis=2) >= 3 * 128


def binary_to_image(mask: np.ndarray) -> np.ndarray:
    """Black and white RGB image of a binary mask."""
    return np.stack([np.where(mask, 255, 0).astype(np.uint8)] * 3, axis=-1)


def element_offsets(se, value: int) -> list[tuple[int, int]]:
    """(dy, dx) offsets from the centre of the structuring element cells equal to ``value``."""
    cy, cx = len(se) // 2, len(se[0]) // 2
    return [(j - cy, i - cx) for j, row in enumerate(se) for i, v in enumerate(row) if v == value]


def neighbour(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Value of the pixel at offset (dy, dx) for every pixel; outside the image counts as False."""
    return shift_mask(mask, -dx, -dy)
