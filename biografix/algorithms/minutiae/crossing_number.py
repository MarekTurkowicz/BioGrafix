"""Minutiae extraction with the Crossing Number (Rutovitz, 1966; Arcelli and Baja, 1984).

For a pixel ``P`` of a one-pixel-wide skeleton, with neighbours ``P1 … P8`` taken
in order around it (``P9 = P1``)::

    CN(P) = ½ · Σ |Pi − Pi+1|,   i = 1 … 8

CN counts how many times the ring of neighbours switches between object and
background, i.e. how many ridge branches leave the pixel: 1 at a ridge ending,
2 in the middle of a ridge, 3 at a bifurcation, 4 where two ridges cross.
The method needs a skeleton without 2×2 blocks, which K3M guarantees.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ..skeletonization._common import NEIGHBOUR_WEIGHTS

RIDGE_CN = 2  # an ordinary ridge pixel – not a minutia


class MinutiaType(StrEnum):
    ISOLATED_POINT = "isolated point"  # CN = 0: a single pixel, usually noise – not counted as a minutia
    ENDING = "ending"                  # CN = 1
    BIFURCATION = "bifurcation"        # CN = 3
    CROSSING = "crossing"              # CN >= 4


COUNTED_TYPES = (MinutiaType.ENDING, MinutiaType.BIFURCATION, MinutiaType.CROSSING)


@dataclass(frozen=True)
class Minutia:
    y: int
    x: int
    type: MinutiaType
    crossing_number: int


def crossing_numbers(skeleton: np.ndarray) -> np.ndarray:
    """Crossing Number of every skeleton pixel; background pixels get ``RIDGE_CN`` so they are never minutiae."""
    skeleton = np.asarray(skeleton, dtype=bool)
    height, width = skeleton.shape
    padded = np.pad(skeleton.astype(np.int8), 1)  # outside the image counts as background
    # the neighbours in clockwise order, so that consecutive entries are adjacent on the ring
    ring = [padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width] for dy, dx, _weight in NEIGHBOUR_WEIGHTS]
    transitions = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8))
    return np.where(skeleton, transitions // 2, RIDGE_CN).astype(np.int8)


def minutia_type(crossing_number: int) -> MinutiaType | None:
    """The minutia type for a Crossing Number, or None when the pixel is not a minutia."""
    if crossing_number == RIDGE_CN:
        return None
    if crossing_number == 0:
        return MinutiaType.ISOLATED_POINT
    if crossing_number == 1:
        return MinutiaType.ENDING
    if crossing_number == 3:
        return MinutiaType.BIFURCATION
    return MinutiaType.CROSSING


def find_minutiae(skeleton: np.ndarray) -> list[Minutia]:
    """All minutiae of a one-pixel-wide skeleton, in row-major order."""
    cn = crossing_numbers(skeleton)
    minutiae = []
    for y, x in zip(*np.nonzero(cn != RIDGE_CN), strict=True):  # only a few hundred candidate pixels
        value = int(cn[y, x])
        kind = minutia_type(value)
        if kind is not None:
            minutiae.append(Minutia(int(y), int(x), kind, value))
    return minutiae
