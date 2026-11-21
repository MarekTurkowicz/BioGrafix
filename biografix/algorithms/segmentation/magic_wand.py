"""Magic wand: select pixels whose value lies in a range around the clicked pixel.

A pixel is accepted when every compared channel value ``v`` satisfies
``seed - range_below <= v <= seed + range_above``. Giving the lower and upper
limit separately turns Paint's single tolerance into an interval.

- Local mode grows the selection from the clicked pixel like a flood fill
  (breadth-first search over 4- or 8-connected neighbours).
- Global mode ("global flood mode") accepts matching pixels anywhere in the
  image, without checking that they are connected to the click.

Both modes can be limited to ``max_pixels`` pixels. The local search stops after
that many pixels, so the selection grows outwards from the click. The global mode
keeps the pixels closest in value to the seed, then the ones nearest the click.
"""

from collections import deque
from enum import IntEnum, StrEnum
from typing import assert_never

import numpy as np


class CompareMode(StrEnum):
    """Which pixel values are compared with the clicked pixel."""

    GRAY = "gray"  # channel average (R+G+B)/3
    RGB = "rgb"    # each RGB channel separately


class Connectivity(IntEnum):
    """Which neighbours count as connected in local mode."""

    FOUR = 4   # edge neighbours
    EIGHT = 8  # edge and corner neighbours


FOUR_NEIGHBOUR_OFFSETS = ((0, -1), (-1, 0), (1, 0), (0, 1))
EIGHT_NEIGHBOUR_OFFSETS = FOUR_NEIGHBOUR_OFFSETS + ((-1, -1), (1, -1), (-1, 1), (1, 1))


def channels_to_compare(image: np.ndarray, compare_mode: CompareMode) -> np.ndarray:
    """Values compared with the seed: (H, W, 1) channel average or (H, W, 3) RGB, as float."""
    rgb = np.asarray(image, dtype=np.float64)[:, :, :3]
    match compare_mode:
        case CompareMode.GRAY:
            return rgb.mean(axis=2, keepdims=True)
        case CompareMode.RGB:
            return rgb
        case _:
            assert_never(compare_mode)


def pixels_in_value_range(image: np.ndarray, x: int, y: int, range_below: float, range_above: float,
                          compare_mode: CompareMode = CompareMode.GRAY) -> np.ndarray:
    """Boolean mask of pixels within ``[seed - range_below, seed + range_above]`` in every compared channel."""
    channels = channels_to_compare(image, compare_mode)
    seed_value = channels[y, x]
    within_range = (channels >= seed_value - range_below) & (channels <= seed_value + range_above)
    return within_range.all(axis=2)


def flood_fill_from_seed(in_range: np.ndarray, x: int, y: int, max_pixels: int = 0,
                         connectivity: Connectivity = Connectivity.FOUR) -> np.ndarray:
    """Pixels of ``in_range`` connected to ``(x, y)``, found by breadth-first search.

    Args:
        in_range: boolean mask of pixels that may be selected.
        max_pixels: stop after this many pixels (0 = no limit).
        connectivity: 4 (edge neighbours) or 8 (edge and corner neighbours).
    """
    height, width = in_range.shape
    selected = np.zeros((height, width), dtype=bool)
    if not in_range[y, x]:
        return selected
    neighbour_offsets = EIGHT_NEIGHBOUR_OFFSETS if connectivity == Connectivity.EIGHT else FOUR_NEIGHBOUR_OFFSETS
    pixel_limit = max_pixels if max_pixels > 0 else height * width

    selected[y, x] = True
    selected_count = 1
    pixels_to_visit = deque([(x, y)])
    while pixels_to_visit and selected_count < pixel_limit:
        current_x, current_y = pixels_to_visit.popleft()
        for dx, dy in neighbour_offsets:
            nx, ny = current_x + dx, current_y + dy
            if 0 <= nx < width and 0 <= ny < height and in_range[ny, nx] and not selected[ny, nx]:
                selected[ny, nx] = True
                selected_count += 1
                if selected_count >= pixel_limit:
                    break
                pixels_to_visit.append((nx, ny))
    return selected


def select_matching_pixels_globally(image: np.ndarray, in_range: np.ndarray, x: int, y: int, max_pixels: int = 0,
                                    compare_mode: CompareMode = CompareMode.GRAY) -> np.ndarray:
    """All pixels of ``in_range``; above ``max_pixels`` keep those closest in value, then closest to the click."""
    if max_pixels <= 0 or in_range.sum() <= max_pixels:
        return in_range.copy()
    channels = channels_to_compare(image, compare_mode)
    distance_in_value = np.abs(channels - channels[y, x]).max(axis=2)
    ys, xs = np.nonzero(in_range)
    squared_distance_to_click = (xs - x) ** 2 + (ys - y) ** 2
    # np.lexsort sorts by the last key first: value distance, then distance to the click
    closest_first = np.lexsort((squared_distance_to_click, distance_in_value[ys, xs]))[:max_pixels]
    selected = np.zeros_like(in_range)
    selected[ys[closest_first], xs[closest_first]] = True
    return selected


def magic_wand_selection(image: np.ndarray, x: int, y: int, range_below: float = 20, range_above: float = 20,
                         compare_mode: CompareMode = CompareMode.GRAY, max_pixels: int = 0,
                         connectivity: Connectivity = Connectivity.FOUR, global_mode: bool = False) -> np.ndarray:
    """Select pixels similar to the pixel at ``(x, y)``; returns an (H, W) boolean mask.

    Args:
        image: (H, W, 3) or (H, W, 4) array; only RGB is compared.
        x, y: clicked pixel (column, row).
        range_below, range_above: how far below and above the seed value a pixel may lie.
        compare_mode: compare the channel average or each RGB channel separately.
        max_pixels: largest number of selected pixels (0 = no limit).
        connectivity: 4 or 8 neighbours (local mode only).
        global_mode: select matching pixels in the whole image, connected or not.
    """
    height, width = image.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("The clicked point is outside the image.")
    in_range = pixels_in_value_range(image, x, y, range_below, range_above, compare_mode)
    if global_mode:
        return select_matching_pixels_globally(image, in_range, x, y, max_pixels, compare_mode)
    return flood_fill_from_seed(in_range, x, y, max_pixels, connectivity)
