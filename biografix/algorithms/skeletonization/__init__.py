"""Skeletonization (thinning) of binary images, one algorithm per module.

- ``zhang_suen``: parallel two-sub-iteration thinning, vectorised with numpy.
- ``k3m``: sequential thinning with six phases and lookup arrays (Saeed et al., 2010).
- ``kmm``: sequential thinning, the predecessor of K3M (Saeed et al., 2001).

``k3m`` and ``kmm`` compile their loops with numba, which takes several seconds to
load, so they are not imported here: use ``from .k3m import k3m_thinning`` when needed.
"""

from ._common import (  # noqa: F401
    LOCAL_WINDOW,
    Binarization,
    ObjectColour,
    binarized_object_mask,
    mask_to_image,
    neighbour_weight_sums,
    object_mask,
)
from .zhang_suen import zhang_suen_thinning
