"""Marker-based watershed segmentation (Meyer's flooding algorithm).

1. The image is converted to gray, smoothed and its Sobel gradient magnitude
   is computed; edges are "ridges" of this relief, flat areas are "valleys".
2. Markers are connected areas of the lowest gradient (the given percentage of
   pixels); markers smaller than ``min_marker_size`` pixels are dropped (unless
   none would be left, e.g. on noise, in which case every area is kept).
3. Flooding starts from all markers at once. Pixels are taken from a priority
   queue in order of increasing gradient; a pixel reached by one basin joins it,
   a pixel where two basins meet becomes a watershed line.
"""

import numpy as np

WATERSHED_LINE = -1  # same value as in watershed_kernels


def mean_blur_3x3(gray: np.ndarray, passes: int = 1) -> np.ndarray:
    """3×3 mean filter applied ``passes`` times (replicated border)."""
    blurred = gray.astype(np.float64)
    height, width = blurred.shape
    for _ in range(passes):
        padded = np.pad(blurred, 1, mode="edge")
        window_sum = np.zeros_like(blurred)
        for dy in range(3):
            for dx in range(3):
                window_sum += padded[dy:dy + height, dx:dx + width]
        blurred = window_sum / 9.0
    return blurred


def sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude sqrt(gx² + gy²) with the 3×3 Sobel kernels (replicated border)."""
    padded = np.pad(gray.astype(np.float64), 1, mode="edge")
    height, width = gray.shape

    def neighbour(dy, dx):
        """The image shifted so that each pixel sees its neighbour at (dy, dx)."""
        return padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]

    gx = (neighbour(-1, 1) + 2 * neighbour(0, 1) + neighbour(1, 1)) \
        - (neighbour(-1, -1) + 2 * neighbour(0, -1) + neighbour(1, -1))
    gy = (neighbour(1, -1) + 2 * neighbour(1, 0) + neighbour(1, 1)) \
        - (neighbour(-1, -1) + 2 * neighbour(-1, 0) + neighbour(-1, 1))
    return np.hypot(gx, gy)


def label_connected_regions(mask: np.ndarray, min_size: int = 1) -> tuple[np.ndarray, int]:
    """Label 4-connected regions of ``mask`` as 1..n in row-major order; regions below ``min_size`` stay 0."""
    from .watershed_kernels import label_connected_regions_compiled  # loads numba

    labels, region_count = label_connected_regions_compiled(np.asarray(mask, dtype=np.bool_), int(min_size))
    return labels, int(region_count)


def flood_from_markers(gradient: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Flood ``gradient`` from the labelled ``markers``; returns labels with ``WATERSHED_LINE`` between basins.

    Pixels leave a priority queue in order of increasing gradient (ties first in,
    first out). A pixel next to exactly one basin joins it and puts its unvisited
    neighbours in the queue; a pixel next to several basins becomes a watershed line.
    """
    from .watershed_kernels import flood_from_markers_compiled  # loads numba

    return flood_from_markers_compiled(np.asarray(gradient, dtype=np.float64), np.asarray(markers, dtype=np.int32))


def watershed_segmentation(image: np.ndarray, smoothing_passes: int = 2, marker_percent: float = 15.0,
                           min_marker_size: int = 20) -> tuple[np.ndarray, int]:
    """Segment the image; returns ``(labels, region_count)``, labels 1..n and ``WATERSHED_LINE`` (-1)."""
    gray = np.asarray(image, dtype=np.float64)[:, :, :3].mean(axis=2)
    gradient = sobel_gradient_magnitude(mean_blur_3x3(gray, smoothing_passes))
    flattest_pixels = gradient <= np.percentile(gradient, marker_percent)
    markers, region_count = label_connected_regions(flattest_pixels, min_marker_size)
    if region_count == 0:
        markers, region_count = label_connected_regions(flattest_pixels, 1)
    return flood_from_markers(gradient, markers), region_count


def paint_regions_with_mean_colour(image: np.ndarray, labels: np.ndarray,
                                   line_color=(255, 255, 255)) -> np.ndarray:
    """Paint every region with its mean colour and the watershed lines with ``line_color``."""
    rgb = np.asarray(image, dtype=np.float64)[:, :, :3]
    region_index = np.maximum(labels, 0)  # lines use bin 0, which is overwritten below
    pixels_per_region = np.bincount(region_index.ravel()).astype(np.float64)
    painted = np.zeros_like(rgb)
    for channel in range(3):
        channel_sum = np.bincount(region_index.ravel(), weights=rgb[:, :, channel].ravel(),
                                  minlength=pixels_per_region.size)
        channel_mean = np.divide(channel_sum, pixels_per_region, out=np.zeros_like(channel_sum),
                                 where=pixels_per_region > 0)
        painted[:, :, channel] = channel_mean[region_index]
    painted[labels == WATERSHED_LINE] = line_color
    return np.clip(painted, 0, 255).round().astype(np.uint8)
