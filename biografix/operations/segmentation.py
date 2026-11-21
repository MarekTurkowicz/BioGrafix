"""Segmentation operations: magic-wand selection editing, filling and watershed.

The magic wand itself is a canvas tool (see ``ui/canvas/view.py``); its settings
live in the Magic wand panel. These operations act on the selection it makes.
"""

from ..algorithms import segmentation as alg
from ..algorithms.point_ops import fill_color
from ..algorithms.segmentation import StructuringElementShape
from ..core.params import ChoiceParam, FloatParam, IntParam
from ..core.registry import TARGET_SELECTION, Report, operation
from ..i18n import tr

CAT = "Segmentation"
SELECTION = "Selection"
REGIONS = "Regions"

ELEMENT_SHAPE = ChoiceParam("shape", "Structuring element", StructuringElementShape.SQUARE, choices=[
    (StructuringElementShape.SQUARE, "Square"),
    (StructuringElementShape.CROSS, "Cross"),
    (StructuringElementShape.DISK, "Disk"),
])
RADIUS = IntParam("radius", "Radius [px]", 1, 1, 50)


def palette_colour_as_rgb(_image, context) -> dict:
    """Default fill colour: the colour chosen in the palette, as R, G, B parameters."""
    hex_colour = context.get("color", "#ff0000").lstrip("#")
    return {"r": int(hex_colour[0:2], 16), "g": int(hex_colour[2:4], 16), "b": int(hex_colour[4:6], 16)}


@operation("seg.fill", CAT, "Fill selection", section=SELECTION,
           params=[IntParam("r", "R", 255, 0, 255), IntParam("g", "G", 0, 0, 255), IntParam("b", "B", 0, 0, 255)],
           dynamic_defaults=palette_colour_as_rgb,
           description="Fills the magic-wand selection (or the region, or the whole image) with a colour.\nDefault: the colour chosen in the palette.")
def fill_selection(image, r, g, b):
    return fill_color(image, r, g, b)


@operation("seg.expand", CAT, "Expand selection", section=SELECTION, target=TARGET_SELECTION,
           params=[RADIUS, ELEMENT_SHAPE],
           description="Dilation of the selection mask: the selection grows by the radius.")
def expand_selection(mask, image, radius, shape):
    return alg.dilate_selection(mask, radius, shape)


@operation("seg.shrink", CAT, "Shrink selection", section=SELECTION, target=TARGET_SELECTION,
           params=[RADIUS, ELEMENT_SHAPE],
           description="Erosion of the selection mask: the selection shrinks by the radius.")
def shrink_selection(mask, image, radius, shape):
    return alg.erode_selection(mask, radius, shape)


@operation("seg.invert", CAT, "Invert selection", section=SELECTION, target=TARGET_SELECTION,
           description="Selected pixels become unselected and the other way round.")
def invert_selection(mask, image):
    return ~mask


@operation("seg.watershed", CAT, "Watershed", section=REGIONS,
           params=[IntParam("smoothing_passes", "Smoothing (3×3 mean passes)", 2, 0, 10),
                   FloatParam("marker_percent", "Markers: flattest pixels [%]", 15.0, 1.0, 60.0, step=1.0, decimals=1),
                   IntParam("min_marker_size", "Minimum marker size [px]", 20, 1, 10000)],
           description="Marker-based watershed: flooding of the gradient relief from its flattest areas.\nRegions get their mean colour, watershed lines are white.")
def watershed(image, smoothing_passes, marker_percent, min_marker_size):
    labels, region_count = alg.watershed_segmentation(image, smoothing_passes, marker_percent, min_marker_size)
    report = Report(tr("Watershed: {n} regions", n=region_count), tr("Watershed"))
    return alg.paint_regions_with_mean_colour(image, labels), report
