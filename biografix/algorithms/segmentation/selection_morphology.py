"""Growing and shrinking a selection mask with binary dilation and erosion.

The structuring element is a square, a cross or a disk of the given radius.
Pixels outside the image count as not selected, so erosion also shrinks a
selection away from the image border.
"""

import numpy as np

SE_SHAPES = ("square", "cross", "disk")


def structuring_element(radius: int, shape: str = "square") -> np.ndarray:
    """(2r+1) × (2r+1) boolean structuring element centred on the middle pixel."""
    if radius < 0:
        raise ValueError("The radius must be >= 0.")
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    if shape == "square":
        return np.ones(dy.shape, dtype=bool)
    if shape == "cross":
        return (dx == 0) | (dy == 0)
    if shape == "disk":
        return dx * dx + dy * dy <= radius * radius
    raise ValueError(f"Unknown structuring element shape: {shape}")


def _shifted(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """``mask`` moved by (dx, dy); uncovered pixels are False."""
    h, w = mask.shape
    out = np.zeros_like(mask)
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def dilate_mask(mask: np.ndarray, radius: int = 1, shape: str = "square") -> np.ndarray:
    """Dilation: a pixel is selected if the element centred on it touches the selection."""
    se = structuring_element(radius, shape)
    out = np.zeros_like(mask, dtype=bool)
    for dy, dx in zip(*np.nonzero(se), strict=True):
        out |= _shifted(mask, int(dx) - radius, int(dy) - radius)
    return out


def erode_mask(mask: np.ndarray, radius: int = 1, shape: str = "square") -> np.ndarray:
    """Erosion: a pixel stays selected only if the whole element centred on it is selected."""
    se = structuring_element(radius, shape)
    out = np.ones_like(mask, dtype=bool)
    for dy, dx in zip(*np.nonzero(se), strict=True):
        out &= _shifted(mask, radius - int(dx), radius - int(dy))
    return out
