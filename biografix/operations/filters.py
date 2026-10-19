"""Spatial filter operations."""

from ..algorithms import filters as alg
from ..core.params import ChoiceParam, IntParam, MatrixParam
from ..core.registry import operation
from .adapters import run_on_pixels_wh

CAT = "Filters"

IMPLEMENTATIONS = [
    ("python", "Manual – replicated border (slow)"),
    ("scipy", "Fast – scipy, symmetric border"),
    ("manual", "Loop – zero border (very slow)"),
]


@operation("filter.box", CAT, "Mean (box blur)", section="Smoothing", slow=True,
           params=[IntParam("size", "Kernel size", 3, 1, 31, odd=True)])
def box(image, size):
    return run_on_pixels_wh(alg.filter_box_blur, image, size=size)


@operation("filter.sobel_magnitude", CAT, "Sobel – gradient magnitude", section="Edges and sharpening", slow=True,
           description="√(Gx² + Gy²) on the grayscale image.")
def sobel_magnitude(image):
    return run_on_pixels_wh(alg.filter_sobel, image)


@operation("filter.sharpen", CAT, "Sharpen", section="Edges and sharpening", slow=True)
def sharpen(image):
    return run_on_pixels_wh(alg.filter_sharpen, image)


@operation("filter.gaussian", CAT, "Gaussian 5×5 (blur)", section="Smoothing", slow=True)
def gaussian(image):
    return run_on_pixels_wh(alg.filter_gaussian, image)


@operation("filter.median", CAT, "Median", section="Smoothing",
           params=[IntParam("size", "Window size", 3, 1, 31, odd=True),
                   ChoiceParam("impl", "Implementation", "scipy", choices=IMPLEMENTATIONS[:2])])
def median(image, size, impl):
    if impl == "python":
        return run_on_pixels_wh(alg.filter_median, image, size=size)
    return alg.median_scipy(image, size)


@operation("filter.custom", CAT, "Custom kernel (convolution)", section="Convolution kernels",
           params=[MatrixParam("kernel", "Kernel (rows = lines)", "0 -1 0\n-1 5 -1\n0 -1 0"),
                   ChoiceParam("impl", "Implementation", "scipy", choices=IMPLEMENTATIONS)],
           description="Any rectangular kernel, preferably of odd size (3×3, 5×5…).")
def custom(image, kernel, impl):
    if impl == "python":
        return run_on_pixels_wh(alg.filter_custom, image, kernel)
    if impl == "manual":
        return alg.convolve_manual(image, kernel)
    return alg.convolve_scipy(image, kernel)


@operation("filter.mask", CAT, "Preset kernel", section="Convolution kernels",
           params=[ChoiceParam("mask", "Kernel", "Sobel", choices=[(k, k) for k in alg.PREDEFINED_MASKS]),
                   ChoiceParam("impl", "Implementation", "scipy", choices=IMPLEMENTATIONS[1:])],
           description="Low-pass, Prewitt, Sobel, Laplace, corner detection.")
def mask(image, mask, impl):
    kernel = alg.PREDEFINED_MASKS[mask]
    if impl == "manual":
        return alg.convolve_manual(image, kernel)
    return alg.convolve_scipy(image, kernel)
