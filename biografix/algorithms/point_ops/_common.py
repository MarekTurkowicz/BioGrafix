"""Helpers shared by the point operations."""

import numpy as np


def round_to_byte(values: np.ndarray) -> np.ndarray:
    """Round to the nearest integer (halves to even, like Python's ``round``) and clip to 0..255 as uint8."""
    return np.clip(np.round(values), 0, 255).astype(np.uint8)


def rgb_channels(image: np.ndarray) -> np.ndarray:
    """The R, G, B channels of an (H, W, 3/4) image."""
    return np.asarray(image)[:, :, :3]
