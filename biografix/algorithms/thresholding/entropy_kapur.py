"""Kapur's entropy method."""

import math

from ._common import _apply_threshold_to_pixels, _to_gray_list


def threshold_entropy(pixels):
    """Kapur's method: maximise the sum of background and object entropies."""
    vals = _to_gray_list(pixels)
    n = len(vals)
    if n == 0:
        return pixels[:]

    hist = [0] * 256
    for v in vals:
        hist[v] += 1

    total = float(n)
    p = [h / total for h in hist]

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

    return _apply_threshold_to_pixels(pixels, best_T)
