"""Fingerprint template: the minutiae used for matching.

Gabor enhancement → ridges (negative response inside the fingerprint area) →
K3M skeleton → Crossing Number minutiae. Only endings and bifurcations are kept,
at least ``BORDER_MARGIN`` px inside the fingerprint area, where the print is
cut off and every ridge seems to end. Each minutia stores its position, the
local ridge orientation (modulo π) and its type.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..minutiae import Minutia, MinutiaType, find_minutiae
from .enhancement import gabor_enhance

BORDER_MARGIN = 8  # px
CACHE_SIZE = 256   # templates kept in memory
MATCHED_TYPES = (MinutiaType.ENDING, MinutiaType.BIFURCATION)


@dataclass(frozen=True)
class FingerprintTemplate:
    """Minutiae of one fingerprint and the images they were found on."""

    points: np.ndarray       # (n, 4): x, y, ridge orientation [rad, mod π], type (0 ending, 1 bifurcation)
    minutiae: list[Minutia]  # the same minutiae, for drawing
    skeleton: np.ndarray     # boolean one-pixel-wide ridges
    area: np.ndarray         # boolean fingerprint area

    def __len__(self) -> int:
        return len(self.minutiae)


def extract_template(image: np.ndarray) -> FingerprintTemplate:
    """Template of a grayscale or RGB fingerprint image (dark ridges on a light background)."""
    from ..skeletonization.k3m import k3m_thinning  # loads numba (preloaded at start-up)

    gray = np.asarray(image, dtype=np.float64)
    if gray.ndim == 3:
        gray = gray[:, :, :3].mean(axis=2)
    enhanced, area, ridge_orientation = gabor_enhance(gray)
    skeleton, _iterations = k3m_thinning((enhanced < 0) & area)
    inner = ndimage.binary_erosion(area, iterations=BORDER_MARGIN)
    minutiae = [m for m in find_minutiae(skeleton) if m.type in MATCHED_TYPES and inner[m.y, m.x]]
    points = np.array([(m.x, m.y, ridge_orientation[m.y, m.x], MATCHED_TYPES.index(m.type)) for m in minutiae],
                      dtype=np.float64).reshape(-1, 4)
    return FingerprintTemplate(points, minutiae, skeleton, area)


_cache: OrderedDict[bytes, FingerprintTemplate] = OrderedDict()


def cached_template(image: np.ndarray) -> FingerprintTemplate:
    """``extract_template`` with a memory cache keyed by the image content.

    Identification compares the query with every image in the scene; their
    templates are computed once and reused while the pixels do not change.
    """
    array = np.ascontiguousarray(image)
    key = hashlib.blake2b(array.tobytes(), digest_size=16, key=repr(array.shape).encode()[:64]).digest()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    template = extract_template(array)
    _cache[key] = template
    if len(_cache) > CACHE_SIZE:
        _cache.popitem(last=False)
    return template
