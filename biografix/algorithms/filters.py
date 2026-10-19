"""Spatial filters: convolution, median and edge detection.

- Pixel-list functions are implemented in pure Python with replicated borders.
- numpy functions use scipy or an explicit loop (zero border).
"""

import math

import numpy as np
from scipy.signal import convolve2d, medfilt


# ============================================================ pixel lists ==

def _clamp_byte(v: float) -> int:
    v = int(round(v))
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _to_2d(pixels, w, h):
    return [pixels[y * w : (y + 1) * w] for y in range(h)]


def _to_1d(rows):
    out = []
    for row in rows:
        out.extend(row)
    return out


def _conv_rgb(pixels, w, h, kernel):
    """Convolve each RGB channel with the kernel."""
    src = _to_2d(pixels, w, h)
    kh = len(kernel)
    kw = len(kernel[0])
    ky = kh // 2
    kx = kw // 2

    out = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            acc_r = acc_g = acc_b = 0.0
            for j in range(kh):
                for i in range(kw):
                    yy = y + j - ky
                    xx = x + i - kx
                    if yy < 0:
                        yy = 0
                    if yy >= h:
                        yy = h - 1
                    if xx < 0:
                        xx = 0
                    if xx >= w:
                        xx = w - 1
                    wgt = kernel[j][i]
                    r, g, b = src[yy][xx]
                    acc_r += r * wgt
                    acc_g += g * wgt
                    acc_b += b * wgt
            out[y][x] = (
                _clamp_byte(acc_r),
                _clamp_byte(acc_g),
                _clamp_byte(acc_b),
            )
    return _to_1d(out)


def filter_box_blur(pixels, w, h, size=3):
    """Mean (box) filter."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")
    n = size * size
    k = [[1.0 / n for _ in range(size)] for _ in range(size)]
    return _conv_rgb(pixels, w, h, k)


def filter_median(pixels, w, h, size=3):
    """Median filter."""
    if size % 2 == 0 or size < 1:
        raise ValueError("The kernel size must be odd and >= 1.")

    src = _to_2d(pixels, w, h)
    r = size // 2
    out = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            neigh_r = []
            neigh_g = []
            neigh_b = []
            for j in range(-r, r + 1):
                for i in range(-r, r + 1):
                    yy = y + j
                    xx = x + i
                    if yy < 0:
                        yy = 0
                    if yy >= h:
                        yy = h - 1
                    if xx < 0:
                        xx = 0
                    if xx >= w:
                        xx = w - 1
                    rr, gg, bb = src[yy][xx]
                    neigh_r.append(rr)
                    neigh_g.append(gg)
                    neigh_b.append(bb)
            neigh_r.sort()
            neigh_g.sort()
            neigh_b.sort()
            m_idx = len(neigh_r) // 2
            out[y][x] = (neigh_r[m_idx], neigh_g[m_idx], neigh_b[m_idx])
    return _to_1d(out)


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


def filter_sharpen(pixels, w, h):
    """High-pass sharpening filter."""
    kernel = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
    return _conv_rgb(pixels, w, h, kernel)


def filter_gaussian(pixels, w, h):
    """5×5 Gaussian blur."""
    kernel = [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]
    norm = 1.0 / 256.0
    kernel = [[v * norm for v in row] for row in kernel]
    return _conv_rgb(pixels, w, h, kernel)


def filter_custom(pixels, w, h, kernel):
    """Convolve with an arbitrary kernel given as a list of rows."""
    return _conv_rgb(pixels, w, h, kernel)


# ============================================================ numpy arrays ==

PREDEFINED_MASKS = {
    "Low-pass": np.array([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
    ]),
    "Prewitt": np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1],
    ]),
    "Sobel": np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]),
    "Laplace": np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ]),
    "Corner detection": np.array([
        [1, 1, 1],
        [1, -2, -1],
        [1, -1, -1],
    ]),
}


def convolve_scipy(image: np.ndarray, mask) -> np.ndarray:
    """Convolve with ``scipy.signal.convolve2d`` using a symmetric border."""
    image_array = np.asarray(image)
    mask = np.asarray(mask, dtype=np.float64)
    # float so that negative responses (Sobel, Laplace) do not wrap around in uint8 before clipping
    convolved_image = np.zeros(image_array.shape, dtype=np.float64)

    for i in range(3):
        convolved_image[:, :, i] = convolve2d(image_array[:, :, i], mask, mode="same", boundary="symm")

    return np.clip(convolved_image, 0, 255).astype(np.uint8)


def convolve_manual(image: np.ndarray, mask) -> np.ndarray:
    """Convolve with an explicit per-pixel loop and a zero border."""
    image_array = np.asarray(image)
    mask = np.asarray(mask, dtype=np.float64)
    mask_height, mask_width = mask.shape
    img_height, img_width, channels = image_array.shape

    convolved_image = np.zeros_like(image_array, dtype=np.float32)

    pad_h = mask_height // 2
    pad_w = mask_width // 2

    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    for c in range(channels):
        for y in range(img_height):
            for x in range(img_width):
                region = padded_image[y:y + mask_height, x:x + mask_width, c]
                convolved_image[y, x, c] = np.sum(region * mask)

    return np.clip(convolved_image, 0, 255).astype(np.uint8)


def median_scipy(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter on each RGB channel (``scipy.signal.medfilt``)."""
    image_array = np.asarray(image)
    median_filtered = np.zeros_like(image_array)
    for i in range(3):
        median_filtered[:, :, i] = medfilt(image_array[:, :, i], kernel_size=size)
    return median_filtered
