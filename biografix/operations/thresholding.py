"""Thresholding operations."""

from ..algorithms import thresholding as alg
from ..core.params import ChoiceParam, FloatParam, IntParam
from ..core.registry import Report, operation
from ..i18n import tr
from .adapters import run_on_pixels

CAT = "Thresholding"
GLOBAL = "Global thresholds"
LOCAL = "Local thresholds"

WINDOW = IntParam("window_size", "Window size (odd)", 15, 3, 301, odd=True,
                  help="Too small a window → noise; too large → behaves like a global method.")


@operation("bin.manual", CAT, "Manual threshold", section=GLOBAL,
           params=[IntParam("threshold", "Threshold T", 128, 0, 255),
                   ChoiceParam("variant", "Variant", "average", choices=[
                       ("average", "Average (R+G+B)/3 – white if > T"),
                       ("luma", "Luminance – white if ≥ T"),
                   ])])
def manual(image, threshold, variant):
    if variant == "luma":
        return run_on_pixels(alg.threshold_manual, image, threshold)
    return alg.manual_threshold(image, threshold)


@operation("bin.otsu", CAT, "Otsu", section=GLOBAL,
           description="Automatic threshold that maximises the between-class variance.")
def otsu(image):
    gray = alg._gray_average(image)
    t = alg.otsu_threshold_value(gray)
    return alg.apply_threshold(gray, t), Report(tr("Otsu threshold: T = {t}", t=t), "Otsu")


@operation("bin.percent_black", CAT, "Percent Black", section=GLOBAL,
           params=[FloatParam("percent_black", "Percentage of black pixels", 50.0, 0.0, 100.0, step=1.0, decimals=1)])
def percent_black(image, percent_black):
    return run_on_pixels(alg.threshold_percent_black, image, percent_black)


@operation("bin.mean_iterative", CAT, "Mean Iterative Selection", section=GLOBAL)
def mean_iterative(image):
    return run_on_pixels(alg.threshold_mean_iterative, image)


@operation("bin.entropy", CAT, "Entropy (Kapur)", section=GLOBAL)
def entropy(image):
    return run_on_pixels(alg.threshold_entropy, image)


@operation("bin.niblack", CAT, "Niblack", section=LOCAL,
           params=[WINDOW, FloatParam("k", "k", -0.2, -1.0, 1.0, step=0.05)],
           description="T = m + k·s")
def niblack(image, window_size, k):
    return alg.niblack_threshold(image, window_size, k)


@operation("bin.sauvola", CAT, "Sauvola", section=LOCAL,
           params=[WINDOW,
                   FloatParam("k", "k", 0.2, 0.0, 1.0, step=0.05,
                              help="Larger k = less noise, but low-contrast lines get lost."),
                   FloatParam("R", "R (deviation range)", 128.0, 1.0, 255.0, step=1.0)],
           description="T = m · (1 + k·(s/R − 1))")
def sauvola(image, window_size, k, R):
    return alg.sauvola_threshold(image, window_size, k, R)


@operation("bin.phansalkar", CAT, "Phansalkar", section=LOCAL,
           params=[WINDOW,
                   FloatParam("k", "k", 0.25, 0.0, 1.0, step=0.05),
                   FloatParam("p", "p", 2.0, 0.0, 10.0, step=0.5),
                   FloatParam("q", "q", 10.0, 0.0, 100.0, step=1.0)],
           description="T = m · (1 + p·e^(−q·m) + k·(s/R − 1)), image in [0, 1], R = 0.5")
def phansalkar(image, window_size, k, p, q):
    return alg.phansalkar_threshold(image, window_size, k, p, q, R=0.5)
