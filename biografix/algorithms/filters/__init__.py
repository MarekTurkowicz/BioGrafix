"""Spatial filters, one filter per module.

- Pixel-list functions are implemented in pure Python with replicated borders.
- numpy functions use scipy or an explicit loop (zero border).
"""

from ._common import _clamp_byte, _conv_rgb, _to_1d, _to_2d  # noqa: F401
from .box_blur import filter_box_blur  # noqa: F401
from .convolution import PREDEFINED_MASKS, convolve_manual, convolve_scipy  # noqa: F401
from .custom_kernel import filter_custom  # noqa: F401
from .gaussian import filter_gaussian  # noqa: F401
from .median import filter_median, median_scipy  # noqa: F401
from .sharpen import filter_sharpen  # noqa: F401
from .sobel import _to_gray_avg, filter_sobel  # noqa: F401
