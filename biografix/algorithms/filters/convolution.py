"""Convolution with predefined or custom masks (numpy: scipy and manual loop)."""

import numpy as np
from scipy.signal import convolve2d


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
