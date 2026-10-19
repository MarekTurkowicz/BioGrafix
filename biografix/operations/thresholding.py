"""Thresholding operations."""

from ..algorithms import thresholding as alg
from ..core.params import ChoiceParam, FloatParam, IntParam
from ..core.registry import Report, operation
from ..i18n import tr
from .adapters import run_on_pixels

CAT = "Thresholding"
GLOBAL = "Global thresholds"


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
