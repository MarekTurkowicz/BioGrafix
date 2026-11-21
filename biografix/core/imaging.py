"""Image array conversions shared by the processing layers."""

import numpy as np


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a (H, W), (H, W, 3) or (H, W, 4) image to RGB uint8.

    Transparent pixels (e.g. after background removal) are composited onto white.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr.astype(np.uint8)] * 3, axis=-1)
    if arr.shape[2] == 3:
        return arr.astype(np.uint8, copy=False)
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        return np.clip(rgb * alpha + 255.0 * (1 - alpha), 0, 255).round().astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def normalize_result(result: np.ndarray) -> np.ndarray:
    """Normalise an operation result to a uint8 (H, W, 3) or (H, W, 4) array."""
    arr = np.asarray(result)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Operation returned an image of invalid shape: {arr.shape}")
    return arr


def paste_region(full: np.ndarray, region: np.ndarray, roi) -> np.ndarray:
    """Return a copy of ``base`` with ``region`` pasted at ``roi``."""
    x0, y0, x1, y1 = roi
    out = np.array(full, copy=True)
    if out.shape[2] == 4 and region.shape[2] == 3:
        out[y0:y1, x0:x1, :3] = region
        out[y0:y1, x0:x1, 3] = 255
    elif out.shape[2] == 3 and region.shape[2] == 4:
        out[y0:y1, x0:x1] = to_rgb(region)
    else:
        out[y0:y1, x0:x1] = region
    return out


def bounding_box_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box ``(x0, y0, x1, y1)`` of the True pixels of a non-empty mask (x1, y1 exclusive)."""
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def paste_region_where_selected(full: np.ndarray, region: np.ndarray, roi, mask: np.ndarray) -> np.ndarray:
    """Like :func:`paste_region`, but only the pixels where ``mask`` (region-sized) is True change."""
    x0, y0, x1, y1 = roi
    pasted = paste_region(full, region, roi)  # keeps the channel count of ``full``
    keep = ~np.asarray(mask, dtype=bool)
    pasted[y0:y1, x0:x1][keep] = np.asarray(full)[y0:y1, x0:x1][keep]
    return pasted
