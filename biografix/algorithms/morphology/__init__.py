"""Binary morphology, one operation per module.

A binary image is a flat list of 0/1 values of length ``w * h``. A structuring
element is a list of integer rows: 1 = object, 0 = don't care, -1 = background
(hit-or-miss only).
"""

from ._common import binary_to_pixels, pixels_to_binary  # noqa: F401
from .apply import MODES, apply_morphology  # noqa: F401
from .dilate import dilate  # noqa: F401
from .erode import erode  # noqa: F401
from .hit_or_miss import hit_or_miss  # noqa: F401
from .structuring_element import parse_structuring_element  # noqa: F401
