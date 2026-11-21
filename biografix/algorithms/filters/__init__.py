"""Spatial filters, one filter per module.

- Mean, Gaussian, sharpen, Sobel, median and custom kernels use a replicated border (numpy).
- ``convolution`` offers scipy (symmetric border) and an explicit loop (zero border).
"""

from ._common import convolve_rgb, correlate_replicated  # noqa: F401
from .box_blur import filter_box_blur  # noqa: F401
from .convolution import PREDEFINED_MASKS, convolve_manual, convolve_scipy  # noqa: F401
from .custom_kernel import filter_custom  # noqa: F401
from .gaussian import filter_gaussian  # noqa: F401
from .median import filter_median, median_scipy  # noqa: F401
from .sharpen import filter_sharpen  # noqa: F401
from .sobel import filter_sobel  # noqa: F401
