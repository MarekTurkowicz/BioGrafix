"""KMM thinning (K. Saeed, M. Rybnik, M. Tabędzki, 2001).

Source: "Implementation and Advanced Results on the Non-interrupted Skeletonization
Algorithm", CAIP 2001, LNCS 2124, pp. 601-609.

Each iteration:

1. Object pixels touching the background with an edge are marked 2, those
   touching it only with a corner are marked 3, the rest stay 1.
2. Marked contour pixels with 2, 3 or 4 neighbours sticking to each other
   (weight sums in ``FOUR_SUMS``) are marked 4.
3. The 4s are deleted.
4. The 2s, then the 3s, are examined in raster order: a pixel whose weight sum is
   in the deletion array is deleted, otherwise it is kept (marked 1).

Iterations repeat while anything was deleted.

The 4s are deleted one by one and each is checked again just before deletion, so
a block where every pixel is a 4 (e.g. 2×2) keeps its last pixel instead of
vanishing. As in K3M, the steps are sequential, so the loops are compiled with numba.
"""

import numpy as np
from numba import njit

from ._common import lookup_table

# Contour points with 2, 3 or 4 sticking neighbours (the paper's 24 sums).
FOUR_SUMS = (3, 6, 12, 24, 48, 96, 192, 129,
             7, 14, 28, 56, 112, 224, 193, 131,
             15, 30, 60, 120, 240, 225, 195, 135)

# Deletion array copied from the paper. For two neighbours only every other
# orientation is deleted (3 yes, 6 no, 12 yes, ...) so that a two-pixel-wide line
# loses one side, not both.
DELETION_SUMS = (3, 5, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 48,
                 52, 53, 54, 55, 56, 60, 61, 62, 63, 65, 67, 69, 71, 77, 79, 80,
                 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 97, 99, 101,
                 103, 109, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126,
                 127, 131, 133, 135, 141, 143, 149, 151, 157, 159, 181, 183, 189, 191, 192, 193,
                 195, 197, 199, 205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 219, 220,
                 221, 222, 223, 224, 225, 227, 229, 231, 237, 239, 240, 241, 243, 244, 245, 246,
                 247, 248, 249, 251, 252, 253, 254, 255)

FOUR_TABLE = lookup_table(FOUR_SUMS)
DELETION_TABLE = lookup_table(DELETION_SUMS)

INTERIOR = 1
EDGE_CONTOUR = 2     # touches the background with an edge
CORNER_CONTOUR = 3   # touches the background only with a corner
STICKING_CONTOUR = 4  # contour pixel with 2-4 sticking neighbours


@njit(cache=True)
def _weight_sum(image, y, x):
    """Neighbour weight sum of (y, x): 128 1 2 / 64 x 4 / 32 16 8 (image has a background border)."""
    return (image[y - 1, x] * 1 + image[y - 1, x + 1] * 2 + image[y, x + 1] * 4 + image[y + 1, x + 1] * 8
            + image[y + 1, x] * 16 + image[y + 1, x - 1] * 32 + image[y, x - 1] * 64 + image[y - 1, x - 1] * 128)


@njit(cache=True)
def _kmm_iterations(image, four_table, deletion_table):
    """Run KMM in place on a 0/1 uint8 image with a one-pixel background border; returns the iteration count."""
    height, width = image.shape
    marks = np.zeros_like(image)
    iterations = 0
    changed = True
    while changed:
        changed = False
        iterations += 1
        # step 1: contour pixels touching the background with an edge (2) or only a corner (3)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                marks[y, x] = 0
                if not image[y, x]:
                    continue
                if image[y - 1, x] == 0 or image[y + 1, x] == 0 or image[y, x - 1] == 0 or image[y, x + 1] == 0:
                    marks[y, x] = EDGE_CONTOUR
                elif (image[y - 1, x - 1] == 0 or image[y - 1, x + 1] == 0
                      or image[y + 1, x - 1] == 0 or image[y + 1, x + 1] == 0):
                    marks[y, x] = CORNER_CONTOUR
                else:
                    marks[y, x] = INTERIOR
        # step 2: contour pixels with 2, 3 or 4 sticking neighbours become 4s
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if marks[y, x] >= EDGE_CONTOUR and four_table[_weight_sum(image, y, x)]:
                    marks[y, x] = STICKING_CONTOUR
        # step 3: delete the 4s (each checked again on the current image)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if marks[y, x] == STICKING_CONTOUR and four_table[_weight_sum(image, y, x)]:
                    image[y, x] = 0
                    marks[y, x] = 0
                    changed = True
        # step 4: delete the unnecessary 2s, then 3s; keep the others as 1s
        for contour in (EDGE_CONTOUR, CORNER_CONTOUR):
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if marks[y, x] == contour:
                        if deletion_table[_weight_sum(image, y, x)]:
                            image[y, x] = 0
                            changed = True
                        marks[y, x] = INTERIOR
    return iterations


def kmm_thinning(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin a boolean object mask to a skeleton; returns ``(skeleton, iterations)``."""
    image = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    iterations = _kmm_iterations(image, FOUR_TABLE, DELETION_TABLE)
    return image[1:-1, 1:-1].astype(bool), int(iterations)
