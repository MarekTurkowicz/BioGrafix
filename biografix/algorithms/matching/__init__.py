"""Fingerprint verification and identification, one step per module.

- ``enhancement``: oriented Gabor filtering that repairs broken ridges.
- ``template``: minutiae (position, ridge orientation, type) of an enhanced print.
- ``matching``: Hough-transform alignment, minutiae pairing and the similarity score.
"""

from .enhancement import gabor_enhance  # noqa: F401
from .matching import MIN_PAIRED_MINUTIAE, SAME_FINGER_THRESHOLD, MatchResult, match_templates  # noqa: F401
from .template import FingerprintTemplate, cached_template, extract_template  # noqa: F401
