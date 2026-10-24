"""Dilation, erosion, opening, closing, thinning and thickening on RGB pixels."""

from ._common import binary_to_pixels, pixels_to_binary
from .dilate import dilate
from .erode import erode
from .hit_or_miss import hit_or_miss

MODES = ("dilate", "erode", "open", "close", "thin", "thicken")


def apply_morphology(pixels, w, h, se, mode):
    """Apply the morphological operation ``mode`` and return black and white pixels."""
    bin_img = pixels_to_binary(pixels)

    if mode == "dilate":
        out = dilate(bin_img, w, h, se)
    elif mode == "erode":
        out = erode(bin_img, w, h, se)
    elif mode == "open":
        tmp = erode(bin_img, w, h, se)
        out = dilate(tmp, w, h, se)
    elif mode == "close":
        tmp = dilate(bin_img, w, h, se)
        out = erode(tmp, w, h, se)
    elif mode == "thin":
        hm = hit_or_miss(bin_img, w, h, se)
        out = [1 if (bin_img[i] == 1 and hm[i] == 0) else 0 for i in range(len(bin_img))]
    elif mode == "thicken":
        hm = hit_or_miss(bin_img, w, h, se)
        out = [1 if (bin_img[i] == 1 or hm[i] == 1) else 0 for i in range(len(bin_img))]
    else:
        raise ValueError(f"Unknown morphology mode: {mode}")

    return binary_to_pixels(out)
