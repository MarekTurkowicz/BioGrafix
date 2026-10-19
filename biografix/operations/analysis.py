"""Segmentation and analysis operations."""

from ..algorithms import color_analysis, segmentation
from ..core.imaging import array_to_pixels
from ..core.params import FloatParam
from ..core.registry import Report, operation
from ..i18n import tr


@operation("seg.rembg", "Segmentation", "Remove background (RemBG)", supports_roi=False,
           description="A neural network separates the subject from the background (the background becomes transparent).\nDownloads the model (~170 MB) on first use.")
def rembg(image):
    return segmentation.remove_background(image)


@operation("analysis.color_coverage", "Analysis", "Colour coverage (HSV)",
           params=[FloatParam("h_min", "H min [°]", 70.0, 0.0, 360.0, step=1.0, decimals=1),
                   FloatParam("h_max", "H max [°]", 160.0, 0.0, 360.0, step=1.0, decimals=1),
                   FloatParam("s_min", "S min", 0.2, 0.0, 1.0, step=0.05, decimals=2),
                   FloatParam("v_min", "V min", 0.2, 0.0, 1.0, step=0.05, decimals=2)],
           description="Percentage of pixels within a hue range (green by default).")
def coverage(image, h_min, h_max, s_min, v_min):
    pixels, _w, _h = array_to_pixels(image)
    percent, h_min, h_max = color_analysis.color_coverage(pixels, h_min, h_max, s_min, v_min)
    return Report(
        tr("Colour coverage (H∈[{h_min}°, {h_max}°], S≥{s_min}, V≥{v_min}): {percent}%",
           h_min=f"{h_min:.1f}", h_max=f"{h_max:.1f}", s_min=f"{s_min:.2f}", v_min=f"{v_min:.2f}",
           percent=f"{percent:.2f}"),
        tr("Colour coverage"),
    )
