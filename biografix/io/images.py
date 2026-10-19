"""Read and write raster images as uint8 arrays of shape (H, W, 3) or (H, W, 4)."""

import os

import numpy as np
from PIL import Image

from .ppm import read_ppm_auto

OPEN_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.ppm *.pnm *.pgm *.pbm);;All files (*)"
SAVE_FILTER = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;PPM (*.ppm)"

_PPM_EXT = (".ppm", ".pnm")


def load_image(path: str) -> np.ndarray:
    """Read an image; PPM P3/P6 uses the built-in reader, other formats use Pillow."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _PPM_EXT:
        with open(path, "rb") as f:
            magic = f.read(2)
        if magic in (b"P3", b"P6"):
            w, h, pixels, _fmt = read_ppm_auto(path)
            return np.array(pixels, dtype=np.uint8).reshape(h, w, 3)

    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        return np.asarray(img.convert("RGBA")).copy()
    return np.asarray(img.convert("RGB")).copy()


def save_image(path: str, image: np.ndarray, jpeg_quality: int = 90) -> None:
    """Write an image in the format given by the file extension (JPEG drops alpha)."""
    ext = os.path.splitext(path)[1].lower()
    img = Image.fromarray(np.asarray(image))
    if ext in (".jpg", ".jpeg"):
        # subsampling=0 keeps full chroma resolution
        img.convert("RGB").save(path, format="JPEG", quality=int(jpeg_quality), optimize=True, subsampling=0)
    elif ext in (".ppm", ".pnm", ".bmp"):
        img.convert("RGB").save(path)
    else:
        img.save(path)
