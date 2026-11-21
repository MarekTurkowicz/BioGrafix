"""Mean (box) filter."""

import numpy as np

from ._common import convolve_rgb


def filter_box_blur(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean (box) filter with a size × size kernel."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")
    n = size * size
    return convolve_rgb(image, [[1.0 / n] * size for _ in range(size)])
