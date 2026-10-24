"""Thresholding (binarisation), one method per module.

- Pixel-list methods (manual, Percent Black, Mean Iterative, Kapur entropy)
  use luminance; values below T become black.
- numpy methods (manual, Otsu, Niblack, Sauvola, Phansalkar) use the channel
  average (R+G+B)/3; values above T become white.
"""

from ._common import (  # noqa: F401
    _apply_threshold_to_pixels,
    _gray_average,
    _luma,
    _to_gray_list,
    apply_threshold,
    binary_to_rgb,
    local_mean_std,
)
from .entropy_kapur import threshold_entropy  # noqa: F401
from .manual import manual_threshold, threshold_manual  # noqa: F401
from .mean_iterative import threshold_mean_iterative  # noqa: F401
from .niblack import niblack_threshold  # noqa: F401
from .otsu import otsu_threshold, otsu_threshold_value  # noqa: F401
from .percent_black import threshold_percent_black  # noqa: F401
from .phansalkar import phansalkar_threshold  # noqa: F401
from .sauvola import sauvola_threshold  # noqa: F401
