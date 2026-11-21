"""Histogram computation, stretching and equalisation, one operation per module.

- Luminance functions return grayscale images.
- RGB functions process each channel independently.
"""

from ._common import gray_image, luminance_levels  # noqa: F401
from .compute import compute_histogram  # noqa: F401
from .equalize import equalize_histogram_rgb, histogram_equalize  # noqa: F401
from .stretch import histogram_stretch, stretch_histogram_rgb  # noqa: F401
