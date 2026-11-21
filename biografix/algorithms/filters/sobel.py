"""Sobel edge detection."""

import math

from ._common import _clamp_byte


def _to_gray_avg(pixels):
    g_list = []
    for r, g, b in pixels:
        g_list.append(int(round((r + g + b) / 3.0)))
    return g_list


def filter_sobel(pixels, w, h):
    """Sobel gradient magnitude as a grayscale edge image."""
    gray = _to_gray_avg(pixels)
    src = [gray[y * w : (y + 1) * w] for y in range(h)]

    gx_k = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]
    gy_k = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ]

    out = [[0 for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            gx = gy = 0.0
            for j in range(3):
                for i in range(3):
                    yy = y + j - 1
                    xx = x + i - 1
                    if yy < 0:
                        yy = 0
                    if yy >= h:
                        yy = h - 1
                    if xx < 0:
                        xx = 0
                    if xx >= w:
                        xx = w - 1
                    v = src[yy][xx]
                    gx += v * gx_k[j][i]
                    gy += v * gy_k[j][i]
            mag = math.sqrt(gx * gx + gy * gy)
            out[y][x] = _clamp_byte(mag)

    flat = []
    for row in out:
        for g in row:
            flat.append((g, g, g))
    return flat
