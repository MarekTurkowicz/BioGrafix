"""Compiled loops of the watershed: region labelling and priority-queue flooding.

Both are sequential pixel-by-pixel searches, so they are compiled with numba.
Importing this module loads numba (a few seconds), which is why ``watershed``
imports it only when needed.
"""

import heapq

import numpy as np
from numba import njit

WATERSHED_LINE = -1
NEIGHBOUR_DY = np.array([-1, 0, 0, 1])
NEIGHBOUR_DX = np.array([0, -1, 1, 0])


@njit(cache=True)
def label_connected_regions_compiled(mask, min_size):
    """Label 4-connected regions of ``mask`` as 1..n in row-major order; regions below ``min_size`` stay 0."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros((height, width), dtype=np.bool_)
    queue_y = np.empty(height * width, dtype=np.int64)
    queue_x = np.empty(height * width, dtype=np.int64)
    region_count = 0
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue
            region_count += 1
            visited[start_y, start_x] = True
            queue_y[0], queue_x[0] = start_y, start_x
            head, tail = 0, 1
            while head < tail:  # the queue keeps every pixel of the region, so it can be unlabelled below
                y, x = queue_y[head], queue_x[head]
                head += 1
                labels[y, x] = region_count
                for k in range(4):
                    ny, nx = y + NEIGHBOUR_DY[k], x + NEIGHBOUR_DX[k]
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue_y[tail], queue_x[tail] = ny, nx
                        tail += 1
            if tail < min_size:
                for i in range(tail):
                    labels[queue_y[i], queue_x[i]] = 0
                region_count -= 1
    return labels, region_count


@njit(cache=True)
def flood_from_markers_compiled(gradient, markers):
    """Meyer's flooding: pixels leave the queue in order of increasing gradient (ties first in, first out)."""
    height, width = gradient.shape
    labels = markers.astype(np.int32)
    already_queued = labels != 0
    queue = [(0.0, np.int64(0), np.int64(0), np.int64(0))]  # typed by its first item, emptied below
    queue.pop()
    insertion_counter = 0

    for y in range(height):
        for x in range(width):
            if labels[y, x] != 0:
                for k in range(4):
                    ny, nx = y + NEIGHBOUR_DY[k], x + NEIGHBOUR_DX[k]
                    if 0 <= ny < height and 0 <= nx < width and not already_queued[ny, nx]:
                        already_queued[ny, nx] = True
                        heapq.heappush(queue, (gradient[ny, nx], np.int64(insertion_counter), ny, nx))
                        insertion_counter += 1

    while len(queue) > 0:
        _gradient, _counter, y, x = heapq.heappop(queue)
        basin = 0           # label of the first neighbouring basin found
        several_basins = False
        for k in range(4):
            ny, nx = y + NEIGHBOUR_DY[k], x + NEIGHBOUR_DX[k]
            if 0 <= ny < height and 0 <= nx < width and labels[ny, nx] > 0:
                if basin == 0:
                    basin = labels[ny, nx]
                elif labels[ny, nx] != basin:
                    several_basins = True
        if basin > 0 and not several_basins:
            labels[y, x] = basin
            for k in range(4):
                ny, nx = y + NEIGHBOUR_DY[k], x + NEIGHBOUR_DX[k]
                if 0 <= ny < height and 0 <= nx < width and not already_queued[ny, nx]:
                    already_queued[ny, nx] = True
                    heapq.heappush(queue, (gradient[ny, nx], np.int64(insertion_counter), ny, nx))
                    insertion_counter += 1
        else:
            labels[y, x] = WATERSHED_LINE
    for y in range(height):
        for x in range(width):
            if labels[y, x] == 0:
                labels[y, x] = WATERSHED_LINE  # pixels enclosed by lines on every side
    return labels
