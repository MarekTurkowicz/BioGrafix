"""Magic wand: select pixels whose value lies in a range around the clicked pixel.

A pixel is accepted when every compared value ``v`` satisfies
``seed - below <= v <= seed + above``. Giving the lower and upper limit
separately turns Paint's single tolerance into an interval.

- Local mode grows the selection from the clicked pixel like a flood fill
  (breadth-first search over 4- or 8-connected neighbours).
- Global mode ("global flood mode") accepts matching pixels anywhere in the
  image, without checking that they are connected to the click.

Both modes can be limited to ``max_pixels`` pixels. The local search stops after
that many pixels, so the selection grows outwards from the click. The global mode
keeps the pixels closest in value to the seed, then the ones nearest the click.
"""

from collections import deque

import numpy as np

COMPARE_MODES = ("gray", "rgb")
NEIGHBOURS_4 = ((0, -1), (-1, 0), (1, 0), (0, 1))
NEIGHBOURS_8 = NEIGHBOURS_4 + ((-1, -1), (1, -1), (-1, 1), (1, 1))


def _values(image: np.ndarray, compare: str) -> np.ndarray:
    """Compared values: (H, W, 1) channel average or (H, W, 3) RGB, as float."""
    rgb = np.asarray(image, dtype=np.float64)[:, :, :3]
    if compare == "gray":
        return rgb.mean(axis=2, keepdims=True)
    if compare == "rgb":
        return rgb
    raise ValueError(f"Unknown compare mode: {compare}")


def range_mask(image: np.ndarray, x: int, y: int, below: float, above: float, compare: str = "gray") -> np.ndarray:
    """Boolean mask of pixels within ``[seed - below, seed + above]`` in every compared channel."""
    values = _values(image, compare)
    seed = values[y, x]
    inside = (values >= seed - below) & (values <= seed + above)
    return inside.all(axis=2)


def flood_select(accept: np.ndarray, x: int, y: int, max_pixels: int = 0, connectivity: int = 4) -> np.ndarray:
    """Pixels of ``accept`` connected to ``(x, y)``, found by breadth-first search.

    Args:
        accept: boolean mask of pixels that may be selected.
        max_pixels: stop after this many pixels (0 = no limit).
        connectivity: 4 (edge neighbours) or 8 (edge and corner neighbours).
    """
    h, w = accept.shape
    selected = np.zeros((h, w), dtype=bool)
    if not accept[y, x]:
        return selected
    steps = NEIGHBOURS_8 if connectivity == 8 else NEIGHBOURS_4
    limit = max_pixels if max_pixels > 0 else h * w

    selected[y, x] = True
    count = 1
    queue = deque([(x, y)])
    while queue and count < limit:
        cx, cy = queue.popleft()
        for dx, dy in steps:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and accept[ny, nx] and not selected[ny, nx]:
                selected[ny, nx] = True
                count += 1
                if count >= limit:
                    break
                queue.append((nx, ny))
    return selected


def global_select(image: np.ndarray, accept: np.ndarray, x: int, y: int, max_pixels: int = 0,
                  compare: str = "gray") -> np.ndarray:
    """All pixels of ``accept``; above ``max_pixels`` keep those closest in value, then closest to the click."""
    if max_pixels <= 0 or accept.sum() <= max_pixels:
        return accept.copy()
    values = _values(image, compare)
    value_dist = np.abs(values - values[y, x]).max(axis=2)
    ys, xs = np.nonzero(accept)
    space_dist = (xs - x) ** 2 + (ys - y) ** 2
    order = np.lexsort((space_dist, value_dist[ys, xs]))[:max_pixels]  # last key is the primary one
    selected = np.zeros_like(accept)
    selected[ys[order], xs[order]] = True
    return selected


def magic_wand(image: np.ndarray, x: int, y: int, below: float = 20, above: float = 20, compare: str = "gray",
               max_pixels: int = 0, connectivity: int = 4, global_mode: bool = False) -> np.ndarray:
    """Select pixels similar to the pixel at ``(x, y)``; returns an (H, W) boolean mask.

    Args:
        image: (H, W, 3) or (H, W, 4) array; only RGB is compared.
        x, y: clicked pixel (column, row).
        below, above: how far below and above the seed value a pixel may lie.
        compare: ``"gray"`` compares the channel average, ``"rgb"`` each channel separately.
        max_pixels: largest number of selected pixels (0 = no limit).
        connectivity: 4 or 8 neighbours (local mode only).
        global_mode: select matching pixels in the whole image, connected or not.
    """
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("The clicked point is outside the image.")
    accept = range_mask(image, x, y, below, above, compare)
    if global_mode:
        return global_select(image, accept, x, y, max_pixels, compare)
    return flood_select(accept, x, y, max_pixels, connectivity)
