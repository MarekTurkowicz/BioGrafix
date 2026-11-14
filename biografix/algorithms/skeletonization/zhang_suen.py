"""Zhang-Suen thinning (T. Y. Zhang, C. Y. Suen, 1984), vectorised with numpy.

Neighbours of a pixel P1 are numbered clockwise from the pixel above::

    P9 P2 P3
    P8 P1 P4
    P7 P6 P5

B(P1) = number of object neighbours, A(P1) = number of 0→1 transitions in the
sequence P2, P3, ..., P9, P2. An object pixel is deleted when

- 2 <= B <= 6 (not an end point, not an interior point),
- A == 1 (deleting it does not split the object locally),
- sub-iteration 1: P2·P4·P6 == 0 and P4·P6·P8 == 0 (south-east boundary or north-west corner),
- sub-iteration 2: P2·P4·P8 == 0 and P2·P6·P8 == 0 (north-west boundary or south-east corner).

The algorithm is parallel: within a sub-iteration every decision is made on the
image from before that sub-iteration, so all deletable pixels can be found at once
with array operations. Sub-iterations alternate until nothing is deleted.
"""

import numpy as np


def _neighbours(padded: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    """P2..P9 as arrays aligned with the image (``padded`` has a one-pixel background border)."""
    offsets = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))  # P2..P9
    return [padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] for dy, dx in offsets]


def deletable_in_subiteration(mask: np.ndarray, subiteration: int) -> np.ndarray:
    """Pixels that Zhang-Suen deletes in sub-iteration 1 or 2."""
    height, width = mask.shape
    padded = np.pad(mask.astype(np.uint8), 1)
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbours(padded, height, width)
    ring = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    neighbour_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
    transitions_0_to_1 = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
    candidate = mask & (neighbour_count >= 2) & (neighbour_count <= 6) & (transitions_0_to_1 == 1)
    if subiteration == 1:
        return candidate & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
    return candidate & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)


def zhang_suen_thinning(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin a boolean object mask to a skeleton; returns ``(skeleton, iterations)``."""
    skeleton = np.array(mask, dtype=bool)
    iterations = 0
    while True:
        iterations += 1
        deleted_any = False
        for subiteration in (1, 2):
            deletable = deletable_in_subiteration(skeleton, subiteration)
            if deletable.any():
                skeleton[deletable] = False
                deleted_any = True
        if not deleted_any:
            return skeleton, iterations
