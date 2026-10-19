"""Thresholding (binarisation).

- Pixel-list methods (manual, Percent Black, Mean Iterative, Kapur entropy)
  use luminance; values below T become black.
- numpy methods (manual, Otsu, Niblack, Sauvola, Phansalkar) use the channel
  average (R+G+B)/3; values above T become white.
"""

import math

import numpy as np
from scipy.ndimage import uniform_filter


# ============================================================ pixel lists ==

def _luma(r, g, b):
    y = 0.299 * r + 0.587 * g + 0.114 * b
    if y < 0:
        y = 0
    elif y > 255:
        y = 255
    return int(round(y))


def _to_gray_list(pixels):
    """Return the luminance (0..255) of every pixel."""
    return [_luma(r, g, b) for (r, g, b) in pixels]


def _apply_threshold_to_pixels(pixels, T):
    """Apply threshold T: luminance below T becomes black, the rest white."""
    out = []
    for r, g, b in pixels:
        y = _luma(r, g, b)
        v = 0 if y < T else 255
        out.append((v, v, v))
    return out


def threshold_manual(pixels, T):
    """Threshold at a user-given T."""
    if T < 0:
        T = 0
    elif T > 255:
        T = 255
    return _apply_threshold_to_pixels(pixels, T)


def threshold_percent_black(pixels, percent_black):
    """Percent Black: choose T so that about ``percent_black`` % of pixels become black."""
    if percent_black < 0:
        percent_black = 0.0
    elif percent_black > 100:
        percent_black = 100.0

    vals = _to_gray_list(pixels)
    n = len(vals)
    if n == 0:
        return pixels[:]

    # cumulative histogram
    hist = [0] * 256
    for v in vals:
        hist[v] += 1

    target = n * (percent_black / 100.0)
    cumsum = 0
    T = 0
    for i in range(256):
        cumsum += hist[i]
        if cumsum >= target:
            T = i
            break

    return _apply_threshold_to_pixels(pixels, T)


def threshold_mean_iterative(pixels, max_iter=100, eps=0.5):
    """Mean Iterative Selection: iterate T towards the mean of both class means."""
    vals = _to_gray_list(pixels)
    n = len(vals)
    if n == 0:
        return pixels[:]

    # start from the global mean
    T = sum(vals) / n

    for _ in range(max_iter):
        g1 = [v for v in vals if v <= T]
        g2 = [v for v in vals if v > T]
        if not g1 or not g2:
            break
        m1 = sum(g1) / len(g1)
        m2 = sum(g2) / len(g2)
        new_T = (m1 + m2) / 2.0
        if abs(new_T - T) < eps:
            T = new_T
            break
        T = new_T

    T_int = int(round(T))
    return _apply_threshold_to_pixels(pixels, T_int)


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


# ============================================================ numpy arrays ==

def _gray_average(image: np.ndarray) -> np.ndarray:
    """Channel average (R+G+B)/3 as uint8."""
    return np.clip(np.asarray(image, dtype=np.float32).mean(axis=2), 0, 255).astype(np.uint8)


def binary_to_rgb(binary_image: np.ndarray) -> np.ndarray:
    """Expand a binary (H, W) image to RGB for display."""
    return np.stack([binary_image] * 3, axis=-1)


def apply_threshold(gray: np.ndarray, threshold) -> np.ndarray:
    """Threshold with a scalar or per-pixel threshold: values above it become 255."""
    binary_image = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary_to_rgb(binary_image)


def manual_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """Threshold the channel average at a user-given T."""
    grayscale = _gray_average(image)
    return apply_threshold(grayscale, threshold)


def otsu_threshold_value(grayscale: np.ndarray) -> int:
    """Otsu threshold: maximise the between-class variance ω0·ω1·(μ0−μ1)²."""
    histogram, _ = np.histogram(grayscale, bins=256, range=(0, 256))

    total_pixels = grayscale.size
    histogram_normalized = histogram / total_pixels

    cumsum = np.cumsum(histogram_normalized)
    cumulative_mean = np.cumsum(histogram_normalized * np.arange(256))

    global_mean = cumulative_mean[-1]

    max_between_class_variance = 0
    best_threshold = 0

    for t in range(256):
        weight_background = cumsum[t]
        weight_foreground = 1 - weight_background

        if weight_background == 0 or weight_foreground == 0:
            continue

        mean_background = cumulative_mean[t] / weight_background
        mean_foreground = (global_mean - cumulative_mean[t]) / weight_foreground

        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            best_threshold = t

    return best_threshold


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Threshold with Otsu's method."""
    grayscale = _gray_average(image)
    return apply_threshold(grayscale, otsu_threshold_value(grayscale))


def local_mean_std(gray: np.ndarray, window_size: int):
    """Local mean ``m`` and standard deviation ``s`` in a square window.

    Uses s² = E[x²] − E[x]²; the input is converted to float so squares do not overflow.
    """
    gray = np.asarray(gray, dtype=np.float64)
    mean = uniform_filter(gray, size=window_size, mode="reflect")
    squared_mean = uniform_filter(gray ** 2, size=window_size, mode="reflect")
    std_dev = np.sqrt(np.maximum(squared_mean - mean ** 2, 0))
    return mean, std_dev


def niblack_threshold(image: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    """Niblack local threshold: T = m + k·s."""
    gray = _gray_average(image).astype(np.float64)
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean + k * std_dev
    return apply_threshold(gray, threshold)


def sauvola_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    """Sauvola local threshold: T = m · (1 + k·(s/R − 1))."""
    gray = _gray_average(image).astype(np.float64)
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean * (1 + k * (std_dev / R - 1))
    return apply_threshold(gray, threshold)


def phansalkar_threshold(image: np.ndarray, window_size: int = 15, k: float = 0.25,
                         p: float = 2.0, q: float = 10.0, R: float = 0.5) -> np.ndarray:
    """Phansalkar local threshold: T = m · (1 + p·exp(−q·m) + k·(s/R − 1)).

    Computed on the image normalised to [0, 1], hence R = 0.5.
    """
    gray = _gray_average(image).astype(np.float64) / 255.0
    mean, std_dev = local_mean_std(gray, window_size)
    threshold = mean * (1 + p * np.exp(-q * mean) + k * (std_dev / R - 1))
    return apply_threshold(gray, threshold)
