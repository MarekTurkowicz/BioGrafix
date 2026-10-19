"""Point operations."""

import numpy as np

from ..algorithms import point_ops as alg
from ..core.params import ChoiceParam, FloatParam, IntParam
from ..core.registry import operation
from .adapters import run_on_pixels

CAT = "Point operations"


@operation("point.add", CAT, "Add / subtract constant", section="Arithmetic",
           params=[IntParam("value", "Value (negative = subtract)", 20, -255, 255)],
           description="Adds a constant to each R, G, B channel (clipped to 0–255).")
def add(image, value):
    return run_on_pixels(alg.add_constant, image, value)


@operation("point.mul", CAT, "Multiply by constant", section="Arithmetic",
           params=[FloatParam("value", "Factor", 1.2, 0.0, 100.0, step=0.1)])
def mul(image, value):
    return run_on_pixels(alg.mul_constant, image, value)


@operation("point.div", CAT, "Divide by constant", section="Arithmetic",
           params=[FloatParam("value", "Divisor", 1.2, 0.001, 1000.0, step=0.1)])
def div(image, value):
    return run_on_pixels(alg.div_constant, image, value)


@operation("point.brightness", CAT, "Brightness", section="Brightness and contrast",
           params=[IntParam("delta", "Brightness", 30, -255, 255)])
def brightness(image, delta):
    return run_on_pixels(alg.change_brightness, image, delta)


@operation("point.levels", CAT, "Levels (linear scaling)", section="Brightness and contrast",
           params=[IntParam("in_min", "Input minimum", 0, 0, 255),
                   IntParam("in_max", "Input maximum", 255, 0, 255)],
           description="The range [min, max] is stretched linearly to [0, 255].")
def levels(image, in_min, in_max):
    return run_on_pixels(alg.linear_color_scale, image, in_min, in_max)


@operation("point.grayscale", CAT, "Grayscale", section="Colour",
           params=[ChoiceParam("mode", "Method", "luma", choices=[
               ("average", "Average (R+G+B)/3"),
               ("luma", "Luminance 0.299R+0.587G+0.114B"),
               ("red", "R channel"),
               ("green", "G channel"),
               ("blue", "B channel"),
           ])])
def grayscale(image, mode):
    if mode == "average":
        return run_on_pixels(alg.to_grayscale_avg, image)
    if mode == "luma":
        return run_on_pixels(alg.to_grayscale_luma, image)
    gray = alg.to_grayscale(image, mode)
    return np.stack([gray] * 3, axis=-1)


@operation("point.brighten", CAT, "Brighten (power, √x)", section="Brightness and contrast")
def brighten(image):
    return alg.brighten(image)


@operation("point.darken", CAT, "Darken (logarithmic)", section="Brightness and contrast")
def darken(image):
    return alg.darken(image)


@operation("point.fill", CAT, "Fill with colour", section="Colour",
           params=[IntParam("r", "R", 255, 0, 255), IntParam("g", "G", 0, 0, 255), IntParam("b", "B", 0, 0, 255)],
           description="Fills the selected region (Region tool) or the whole image with the given colour.")
def fill(image, r, g, b):
    return alg.fill_color(image, r, g, b)
