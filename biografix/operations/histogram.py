"""Histogram operations."""

import numpy as np
from matplotlib.figure import Figure

from ..algorithms import histogram as alg
from ..core.imaging import array_to_pixels
from ..core.registry import operation
from ..i18n import tr
from .adapters import run_on_pixels

CAT = "Histogram"


@operation("hist.show_luma", CAT, "Show luminance histogram", section="Charts",
           description="Luminance histogram (256 bins).")
def show_luma(image):
    pixels, _w, _h = array_to_pixels(image)
    hist = alg.compute_histogram(pixels)
    fig = Figure(figsize=(7, 3.2), layout="constrained")
    ax = fig.add_subplot()
    ax.bar(range(256), hist, width=1.0, color="#444444")
    ax.set_xlim(-0.5, 255.5)
    ax.set_title(tr("Luminance histogram"))
    ax.text(0.01, 0.95, f"N = {sum(hist)}", transform=ax.transAxes, va="top")
    return fig


@operation("hist.show_rgb", CAT, "Show R, G, B and average histograms", section="Charts")
def show_rgb(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    fig = Figure(figsize=(10, 6), layout="constrained")
    for i, (data, color, title) in enumerate([
        (r, "red", tr("Histogram – R channel")),
        (g, "green", tr("Histogram – G channel")),
        (b, "blue", tr("Histogram – B channel")),
        # int32 so the sum does not overflow uint8
        ((r.astype(np.int32) + g + b) / 3, "gray", tr("Histogram – average (R+G+B)/3")),
    ]):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.hist(data.ravel(), bins=256, range=(0, 256), color=color, alpha=0.5)
        ax.set_title(title)
    return fig


@operation("hist.stretch_luma", CAT, "Stretch histogram (luminance)", section="On luminance",
           description="The result is grayscale.")
def stretch_luma(image):
    return run_on_pixels(alg.histogram_stretch, image)


@operation("hist.equalize_luma", CAT, "Equalize histogram (luminance)", section="On luminance",
           description="The result is grayscale.")
def equalize_luma(image):
    return run_on_pixels(alg.histogram_equalize, image)


@operation("hist.stretch_rgb", CAT, "Stretch histogram (each RGB channel)", section="On RGB channels")
def stretch_rgb(image):
    return alg.stretch_histogram_rgb(image)


@operation("hist.equalize_rgb", CAT, "Equalize histogram (each RGB channel)", section="On RGB channels")
def equalize_rgb(image):
    return alg.equalize_histogram_rgb(image)
