"""Fingerprint enhancement with oriented Gabor filters (Hong, Wan, Jain, 1998).

A plain threshold leaves real fingerprints with broken ridges and pores; every
break becomes two false ridge endings. The Gabor filter is a cosine wave along
the direction across the ridges, limited by a Gaussian. Tuned to the local ridge
direction and the ridge spacing, it strengthens the ridges, bridges small breaks
and suppresses noise:

1. uneven lighting is removed by subtracting the local mean brightness, then the
   image is normalised to mean 0 and variance 1;
2. the fingerprint area is where the local variance is high (ridges and valleys);
3. the ridge orientation field comes from smoothed products of the image
   gradients (doubled angles, so that opposite gradients do not cancel out);
4. every pixel takes the response of the filter whose orientation is closest
   to its local orientation (a bank of ``ORIENTATIONS`` filters).

Task 5 allows any library, so the convolutions use scipy (FFT for the large
Gabor kernels).
"""

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

RIDGE_PERIOD = 9.0      # px between ridges at 500 dpi
GABOR_SIGMA = 4.0       # px, width of the Gaussian envelope
ORIENTATIONS = 16       # filters in the bank (11.25° apart)
LIGHTING_BLOCK = 32     # px, window of the local mean brightness that is subtracted
VARIANCE_BLOCK = 16     # px, window of the local variance for the fingerprint area
AREA_MIN_VARIANCE = 0.25
ORIENTATION_SMOOTHING = 7.0  # sigma of the Gaussian smoothing of the gradient products


def normalize(gray: np.ndarray) -> np.ndarray:
    """Gray image without its local mean brightness, scaled to mean 0 and standard deviation 1."""
    gray = np.asarray(gray, dtype=np.float64)
    detail = gray - ndimage.uniform_filter(gray, LIGHTING_BLOCK)
    return (detail - detail.mean()) / (detail.std() + 1e-9)


def fingerprint_area(normalized: np.ndarray) -> np.ndarray:
    """Mask of the fingerprint: high local variance, cleaned with opening, hole filling and erosion."""
    mean = ndimage.uniform_filter(normalized, VARIANCE_BLOCK)
    variance = ndimage.uniform_filter(normalized * normalized, VARIANCE_BLOCK) - mean * mean
    area = ndimage.binary_opening(variance > AREA_MIN_VARIANCE, iterations=4)
    area = ndimage.binary_fill_holes(area)
    return ndimage.binary_erosion(area, iterations=6)


def gradient_orientation(normalized: np.ndarray) -> np.ndarray:
    """Dominant gradient direction (across the ridges) of every pixel, in [-π/2, π/2]."""
    gx = ndimage.sobel(normalized, axis=1)
    gy = ndimage.sobel(normalized, axis=0)
    # doubled angles: the gradients on both sides of a ridge point in opposite directions
    sin_2a = ndimage.gaussian_filter(2 * gx * gy, ORIENTATION_SMOOTHING)
    cos_2a = ndimage.gaussian_filter(gx * gx - gy * gy, ORIENTATION_SMOOTHING)
    return 0.5 * np.arctan2(sin_2a, cos_2a)


def gabor_kernel(angle: float, period: float = RIDGE_PERIOD, sigma: float = GABOR_SIGMA) -> np.ndarray:
    """Even Gabor kernel: a cosine along ``angle`` (across the ridges) under a Gaussian envelope."""
    half = int(round(3 * sigma))
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    across = x * np.cos(angle) + y * np.sin(angle)
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) * np.cos(2 * np.pi * across / period)
    return kernel - kernel.mean()  # zero mean: flat areas give no response


def gabor_enhance(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(enhanced, area, ridge_orientation)``.

    ``enhanced`` is negative on ridges (dark in the input) and positive in valleys;
    ``ridge_orientation`` is the ridge direction of every pixel, modulo π.
    """
    normalized = normalize(gray)
    area = fingerprint_area(normalized)
    across = gradient_orientation(normalized)
    step = np.pi / ORIENTATIONS
    nearest_filter = np.round((across % np.pi) / step).astype(int) % ORIENTATIONS
    enhanced = np.zeros_like(normalized)
    for k in range(ORIENTATIONS):
        uses_k = nearest_filter == k
        if uses_k.any():
            response = fftconvolve(normalized, gabor_kernel(k * step), mode="same")
            enhanced[uses_k] = response[uses_k]
    return enhanced, area, (across + np.pi / 2) % np.pi
