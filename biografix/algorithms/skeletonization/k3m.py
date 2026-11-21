"""K3M thinning (K. Saeed, M. Tabędzki, M. Rybnik, M. Adamski, 2010).

Source: "K3M: A universal algorithm for image skeletonization and a review of
thinning techniques", Int. J. Appl. Math. Comput. Sci. 20(2), 2010, pp. 317-335.

Each iteration:

- Phase 0: mark as borders the object pixels whose neighbour weight sum is in A0.
- Phases 1-5: delete the borders that have 3 (phase 1), 3-4 (phase 2), ...,
  3-7 (phase 5) neighbours sticking to each other (weight sum in A1..A5).
- Phase 6: unmark the remaining borders.

Iterations repeat while anything was deleted; a final pass with A1pix thins the
remaining two-pixel-wide parts to a one-pixel-wide skeleton.

The phases are sequential: a deletion changes the neighbours of the pixels
examined after it, in raster order. That is why the loops are compiled with numba.
"""

import numpy as np
from numba import njit

from ._common import lookup_table

# Lookup arrays of neighbour weight sums, copied from the paper (section 3.3).
A0 = (3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124, 126, 127, 129, 131, 135,
      143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241, 243, 247, 248, 249,
      251, 252, 253, 254)
A1 = (7, 14, 28, 56, 112, 131, 193, 224)
A2 = (7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240)
A3 = (7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193, 195, 199, 224, 225, 227,
      240, 241, 248)
A4 = (7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 193, 195, 199, 207,
      224, 225, 227, 231, 240, 241, 243, 248, 249, 252)
A5 = (7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 191, 193, 195, 199,
      207, 224, 225, 227, 231, 239, 240, 241, 243, 248, 249, 251, 252, 254)
A1PIX = A0  # the paper lists the same values for the one-pixel-width pass

# row i = lookup table of phase i (0 = marking borders, 1..5 = deleting); row 6 = A1pix
PHASE_TABLES = np.stack([lookup_table(t) for t in (A0, A1, A2, A3, A4, A5, A1PIX)])


@njit(cache=True)
def _weight_sum(image, y, x):
    """Neighbour weight sum of (y, x): 128 1 2 / 64 x 4 / 32 16 8 (image has a background border)."""
    return (image[y - 1, x] * 1 + image[y - 1, x + 1] * 2 + image[y, x + 1] * 4 + image[y + 1, x + 1] * 8
            + image[y + 1, x] * 16 + image[y + 1, x - 1] * 32 + image[y, x - 1] * 64 + image[y - 1, x - 1] * 128)


@njit(cache=True)
def _k3m_iterations(image, tables):
    """Run K3M in place on a 0/1 uint8 image with a one-pixel background border; returns the iteration count."""
    height, width = image.shape
    border = np.zeros_like(image)
    iterations = 0
    changed = True
    while changed:
        changed = False
        iterations += 1
        # phase 0: mark borders
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if image[y, x] and tables[0, _weight_sum(image, y, x)]:
                    border[y, x] = 1
        # phases 1-5: delete borders with a growing number of sticking neighbours
        for phase in range(1, 6):
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if border[y, x] and tables[phase, _weight_sum(image, y, x)]:
                        image[y, x] = 0
                        border[y, x] = 0
                        changed = True
        # phase 6: unmark the remaining borders
        border[:, :] = 0
    # thinning to a one-pixel-wide skeleton
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if image[y, x] and tables[6, _weight_sum(image, y, x)]:
                image[y, x] = 0
    return iterations


def k3m_thinning(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin a boolean object mask to a one-pixel-wide skeleton; returns ``(skeleton, iterations)``."""
    image = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    iterations = _k3m_iterations(image, PHASE_TABLES)
    return image[1:-1, 1:-1].astype(bool), int(iterations)
