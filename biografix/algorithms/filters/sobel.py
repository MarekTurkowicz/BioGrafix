"""Sobel edge detection."""

import numpy as np

from ._common import correlate_replicated, round_to_byte

SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def filter_sobel(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude sqrt(gx² + gy²) of the channel average, as a grayscale image."""
    rgb = np.asarray(image)[:, :, :3].astype(np.int32)
    gray = np.round(rgb.sum(axis=2) / 3.0)
    gx = correlate_replicated(gray, SOBEL_X)
    gy = correlate_replicated(gray, SOBEL_Y)
    magnitude = round_to_byte(np.sqrt(gx * gx + gy * gy))
    return np.stack([magnitude] * 3, axis=-1)
