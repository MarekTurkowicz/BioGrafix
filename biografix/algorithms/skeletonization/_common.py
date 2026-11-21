"""Helpers shared by the thinning algorithms: binary masks and neighbour weights.

The KMM and K3M papers describe a pixel's 8-neighbourhood by one number, the sum
of the weights of its object neighbours::

    128   1   2
     64   x   4
     32  16   8

Starting from the pixel above and going clockwise, each neighbour is one bit.
"""

from enum import StrEnum
from typing import assert_never

import numpy as np

OBJECT_THRESHOLD = 128  # gray value separating dark and light pixels

# (dy, dx, weight) of the eight neighbours, clockwise from the pixel above
NEIGHBOUR_WEIGHTS = ((-1, 0, 1), (-1, 1, 2), (0, 1, 4), (1, 1, 8),
                     (1, 0, 16), (1, -1, 32), (0, -1, 64), (-1, -1, 128))


class ObjectColour(StrEnum):
    """Which pixels form the object to be thinned."""

    BLACK = "black"  # dark ridges on a light background (fingerprint after binarization)
    WHITE = "white"  # light object on a dark background


def object_mask(image: np.ndarray, object_colour: ObjectColour = ObjectColour.BLACK) -> np.ndarray:
    """Boolean mask of object pixels; the gray average is compared with ``OBJECT_THRESHOLD``."""
    gray = np.asarray(image, dtype=np.float64)[:, :, :3].mean(axis=2)
    match object_colour:
        case ObjectColour.BLACK:
            return gray < OBJECT_THRESHOLD
        case ObjectColour.WHITE:
            return gray >= OBJECT_THRESHOLD
        case _:
            assert_never(object_colour)


def mask_to_image(mask: np.ndarray, object_colour: ObjectColour = ObjectColour.BLACK) -> np.ndarray:
    """(H, W, 3) image with the object pixels in ``object_colour`` and the rest in the opposite colour."""
    object_value, background_value = (0, 255) if object_colour == ObjectColour.BLACK else (255, 0)
    gray = np.where(mask, object_value, background_value).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def neighbour_weight_sums(mask: np.ndarray) -> np.ndarray:
    """Weight sum (0..255) of the object neighbours of every pixel; outside the image counts as background."""
    padded = np.pad(mask.astype(np.int32), 1)
    height, width = mask.shape
    sums = np.zeros((height, width), dtype=np.int32)
    for dy, dx, weight in NEIGHBOUR_WEIGHTS:
        sums += weight * padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
    return sums


def lookup_table(values) -> np.ndarray:
    """Boolean array of length 256 that is True at the given weight sums."""
    table = np.zeros(256, dtype=np.bool_)
    table[list(values)] = True
    return table
