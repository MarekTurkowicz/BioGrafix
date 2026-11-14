"""Marker-based watershed segmentation (Meyer's flooding algorithm).

1. The image is converted to gray, smoothed and its Sobel gradient magnitude
   is computed; edges are "ridges" of this relief, flat areas are "valleys".
2. Markers are connected areas of the lowest gradient (the given percentage of
   pixels); markers smaller than ``min_marker`` pixels are dropped (unless none
   would be left, e.g. on noise, in which case every area is kept).
3. Flooding starts from all markers at once. Pixels are taken from a priority
   queue in order of increasing gradient; a pixel reached by one basin joins it,
   a pixel where two basins meet becomes a watershed line.
"""

import heapq
from collections import deque

import numpy as np

LINE = -1
NEIGHBOURS = ((0, -1), (-1, 0), (1, 0), (0, 1))


def box_blur(gray: np.ndarray, passes: int = 1) -> np.ndarray:
    """3×3 mean filter applied ``passes`` times (replicated border)."""
    out = gray.astype(np.float64)
    h, w = out.shape
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                acc += p[dy:dy + h, dx:dx + w]
        out = acc / 9.0
    return out


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude sqrt(gx² + gy²) with the 3×3 Sobel kernels (replicated border)."""
    p = np.pad(gray.astype(np.float64), 1, mode="edge")
    h, w = gray.shape

    def at(dy, dx):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    gx = (at(-1, 1) + 2 * at(0, 1) + at(1, 1)) - (at(-1, -1) + 2 * at(0, -1) + at(1, -1))
    gy = (at(1, -1) + 2 * at(1, 0) + at(1, 1)) - (at(-1, -1) + 2 * at(-1, 0) + at(-1, 1))
    return np.hypot(gx, gy)


def label_components(mask: np.ndarray, min_size: int = 1) -> tuple[np.ndarray, int]:
    """Label 4-connected components of ``mask`` as 1..n; components below ``min_size`` stay 0."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    n = 0
    for sy, sx in zip(*np.nonzero(mask), strict=True):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        component = [(sy, sx)]
        queue = deque(component)
        while queue:
            y, x = queue.popleft()
            for dx, dy in NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = n
                    component.append((ny, nx))
                    queue.append((ny, nx))
        if len(component) < min_size:
            ys, xs = zip(*component, strict=True)
            labels[list(ys), list(xs)] = 0
            n -= 1
    return labels, n


def flood(relief: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Flood ``relief`` from the labelled ``markers``; returns labels with ``LINE`` on watershed lines."""
    h, w = relief.shape
    labels = markers.astype(np.int32).copy()
    queued = labels != 0
    heap: list[tuple[float, int, int, int]] = []
    order = 0  # tie-breaker: pixels of equal height are processed first in, first out

    def push_neighbours(y, x):
        nonlocal order
        for dx, dy in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not queued[ny, nx]:
                queued[ny, nx] = True
                heapq.heappush(heap, (relief[ny, nx], order, ny, nx))
                order += 1

    for y, x in zip(*np.nonzero(labels), strict=True):
        push_neighbours(y, x)

    while heap:
        _height, _, y, x = heapq.heappop(heap)
        around = set()
        for dx, dy in NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] > 0:
                around.add(int(labels[ny, nx]))
        if len(around) == 1:
            labels[y, x] = around.pop()
            push_neighbours(y, x)
        else:
            labels[y, x] = LINE
    labels[labels == 0] = LINE  # pixels enclosed by lines on every side
    return labels


def watershed(image: np.ndarray, smoothing: int = 2, marker_percent: float = 15.0,
              min_marker: int = 20) -> tuple[np.ndarray, int]:
    """Segment the image; returns ``(labels, regions)``, labels 1..regions and ``LINE`` (-1)."""
    gray = np.asarray(image, dtype=np.float64)[:, :, :3].mean(axis=2)
    relief = sobel_magnitude(box_blur(gray, smoothing))
    low = relief <= np.percentile(relief, marker_percent)
    markers, regions = label_components(low, min_marker)
    if regions == 0:
        markers, regions = label_components(low, 1)
    return flood(relief, markers), regions


def render_regions(image: np.ndarray, labels: np.ndarray, line_color=(255, 255, 255)) -> np.ndarray:
    """Paint every region with its mean colour and the watershed lines with ``line_color``."""
    rgb = np.asarray(image, dtype=np.float64)[:, :, :3]
    index = np.maximum(labels, 0)  # lines use bin 0, which is overwritten below
    count = np.bincount(index.ravel()).astype(np.float64)
    out = np.zeros_like(rgb)
    for c in range(3):
        total = np.bincount(index.ravel(), weights=rgb[:, :, c].ravel(), minlength=count.size)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        out[:, :, c] = mean[index]
    out[labels == LINE] = line_color
    return np.clip(out, 0, 255).round().astype(np.uint8)
