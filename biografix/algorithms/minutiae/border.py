"""Removing false minutiae at the border of the fingerprint.

Every ridge ends where the scanner stopped seeing the finger, so the edge of
the print is lined with false endings (and false bifurcations where ridges meet
the outer contour). The fingerprint area is estimated from the skeleton itself:

1. dilation by about one ridge spacing merges neighbouring ridges into one blob;
2. filling the holes closes the gaps left inside it;
3. erosion by the ridge spacing plus a margin shrinks the blob back and away from
   the edge, including the image border (outside the image counts as background).

Minutiae outside the resulting area are discarded.
"""

import numpy as np

from ..segmentation.selection_morphology import StructuringElementShape, dilate_selection, erode_selection
from .crossing_number import Minutia


def _spread_along_rows(background: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """Every horizontal run of background that contains an ``outside`` pixel becomes outside."""
    height = background.shape[0]
    # a False column in front of every row keeps runs from continuing into the next row
    padded = np.hstack([np.zeros((height, 1), dtype=bool), background]).ravel()
    run_starts = padded & ~np.roll(padded, 1)
    run_id = np.cumsum(run_starts) * padded  # 1..n inside runs, 0 elsewhere
    seen_outside = np.hstack([np.zeros((height, 1), dtype=bool), outside]).ravel()
    touches_outside = np.bincount(run_id, weights=seen_outside, minlength=run_id.max() + 1) > 0
    touches_outside[0] = False
    return touches_outside[run_id].reshape(height, -1)[:, 1:]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """``mask`` with its holes filled: background not reachable from the image border becomes object.

    Starting from the background on the image border, whole background runs are
    marked as outside, alternately along rows and columns, until nothing changes
    (4-connected background). Each sweep is a few array operations.
    """
    background = ~np.asarray(mask, dtype=bool)
    outside = np.zeros_like(background)
    outside[[0, -1], :] = background[[0, -1], :]
    outside[:, [0, -1]] = background[:, [0, -1]]
    while True:
        spread = _spread_along_rows(background, outside)
        spread = _spread_along_rows(background.T, spread.T).T
        if np.array_equal(spread, outside):
            return ~outside
        outside = spread


def fingerprint_area(skeleton: np.ndarray, ridge_spacing: int = 8, margin: int = 12) -> np.ndarray:
    """Mask of the fingerprint area at least ``margin`` px away from its edge."""
    merged = dilate_selection(np.asarray(skeleton, dtype=bool), ridge_spacing, StructuringElementShape.DISK)
    return erode_selection(fill_holes(merged), ridge_spacing + margin, StructuringElementShape.DISK)


def remove_border_minutiae(minutiae: list[Minutia], area: np.ndarray) -> tuple[list[Minutia], list[Minutia]]:
    """Split the minutiae into those inside ``area`` (kept) and those at the border (removed)."""
    kept = [m for m in minutiae if area[m.y, m.x]]
    removed = [m for m in minutiae if not area[m.y, m.x]]
    return kept, removed
