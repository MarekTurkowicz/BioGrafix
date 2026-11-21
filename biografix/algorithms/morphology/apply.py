"""Dilation, erosion, opening, closing, thinning and thickening of an RGB image."""

import numpy as np

from ._common import binary_to_image, image_to_binary
from .dilate import dilate
from .erode import erode
from .hit_or_miss import hit_or_miss

MODES = ("dilate", "erode", "open", "close", "thin", "thicken")


def apply_morphology(image: np.ndarray, se, mode: str) -> np.ndarray:
    """Threshold the image at 128, apply the operation ``mode`` and return a black and white image."""
    mask = image_to_binary(image)
    if mode == "dilate":
        out = dilate(mask, se)
    elif mode == "erode":
        out = erode(mask, se)
    elif mode == "open":
        out = dilate(erode(mask, se), se)
    elif mode == "close":
        out = erode(dilate(mask, se), se)
    elif mode == "thin":
        out = mask & ~hit_or_miss(mask, se)
    elif mode == "thicken":
        out = mask | hit_or_miss(mask, se)
    else:
        raise ValueError(f"Unknown morphology mode: {mode}")
    return binary_to_image(out)
