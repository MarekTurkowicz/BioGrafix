"""Kapur's entropy method."""

import math

import numpy as np

from ._common import luminance_levels, threshold_luminance


def threshold_entropy(image: np.ndarray) -> np.ndarray:
    """Kapur's method: maximise the sum of background and object entropies.

    The search runs over the 256-bin histogram, so its cost does not depend on the image size.
    """
    levels = luminance_levels(image)
    total = float(levels.size)
    p = [h / total for h in np.bincount(levels.ravel(), minlength=256).tolist()]

    best_T = 0
    best_H = -1e9

    for T in range(0, 255):
        # background: [0..T], object: [T+1..255]
        p1 = sum(p[0 : T + 1])
        p2 = sum(p[T + 1 : 256])
        if p1 <= 0.0 or p2 <= 0.0:
            continue

        H1 = 0.0
        for i in range(0, T + 1):
            if p[i] > 0.0:
                pi = p[i] / p1
                H1 -= pi * math.log(pi + 1e-12)

        H2 = 0.0
        for i in range(T + 1, 256):
            if p[i] > 0.0:
                pi = p[i] / p2
                H2 -= pi * math.log(pi + 1e-12)

        H = H1 + H2
        if H > best_H:
            best_H = H
            best_T = T

    return threshold_luminance(levels, best_T)
