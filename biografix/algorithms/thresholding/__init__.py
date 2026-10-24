"""Thresholding (binarisation), one method per module.

- Pixel-list methods (manual, Percent Black, Mean Iterative, Kapur entropy)
  use luminance; values below T become black.
- numpy methods (manual, Otsu, Niblack, Sauvola, Phansalkar) use the channel
  average (R+G+B)/3; values above T become white.
"""

from ._common import _luma, _to_gray_list, _apply_threshold_to_pixels, _gray_average, binary_to_rgb, apply_threshold, local_mean_std  # noqa: F401
from .manual import threshold_manual, manual_threshold  # noqa: F401
from .percent_black import threshold_percent_black  # noqa: F401
from .mean_iterative import threshold_mean_iterative  # noqa: F401
from .entropy_kapur import threshold_entropy  # noqa: F401
from .otsu import otsu_threshold_value, otsu_threshold  # noqa: F401
from .niblack import niblack_threshold  # noqa: F401
from .sauvola import sauvola_threshold  # noqa: F401
from .phansalkar import phansalkar_threshold  # noqa: F401
