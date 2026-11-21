"""Thresholding (binarisation), one method per module.

- Luminance methods (manual, Percent Black, Mean Iterative, Kapur entropy):
  values below T become black.
- Channel-average methods (manual, Otsu, Niblack, Sauvola, Phansalkar):
  values above T become white.
"""

from ._common import (  # noqa: F401
    _gray_average,
    apply_threshold,
    binary_to_rgb,
    local_mean_std,
    luminance_levels,
    threshold_luminance,
)
from .entropy_kapur import threshold_entropy  # noqa: F401
from .manual import manual_threshold, threshold_manual  # noqa: F401
from .mean_iterative import threshold_mean_iterative  # noqa: F401
from .niblack import niblack_threshold  # noqa: F401
from .otsu import otsu_threshold, otsu_threshold_value  # noqa: F401
from .percent_black import threshold_percent_black  # noqa: F401
from .phansalkar import phansalkar_threshold  # noqa: F401
from .sauvola import sauvola_threshold  # noqa: F401
