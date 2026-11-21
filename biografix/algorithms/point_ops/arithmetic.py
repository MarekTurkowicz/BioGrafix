"""Adding, multiplying and dividing every R, G, B value by a constant (clipped to 0..255)."""

import numpy as np

from ._common import rgb_channels, round_to_byte


def add_constant(image: np.ndarray, value: int) -> np.ndarray:
    """Add a constant to every channel."""
    return round_to_byte(rgb_channels(image).astype(np.int32) + value)


def mul_constant(image: np.ndarray, value: float) -> np.ndarray:
    """Multiply every channel by a constant."""
    return round_to_byte(rgb_channels(image).astype(np.float64) * value)


def div_constant(image: np.ndarray, value: float) -> np.ndarray:
    """Divide every channel by a non-zero constant."""
    if value == 0:
        raise ValueError("Division by zero is not allowed.")
    return round_to_byte(rgb_channels(image).astype(np.float64) / value)
