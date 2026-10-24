"""Histogram computation, stretching and equalisation, one operation per module.

- Pixel-list functions work on luminance and return grayscale images.
- numpy functions process each RGB channel independently.
"""

from ._common import _luma  # noqa: F401
from .compute import compute_histogram  # noqa: F401
from .stretch import histogram_stretch, stretch_histogram_rgb  # noqa: F401
from .equalize import histogram_equalize, equalize_histogram_rgb  # noqa: F401
