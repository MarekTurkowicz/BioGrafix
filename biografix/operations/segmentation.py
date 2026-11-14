"""Segmentation operations: magic-wand selection editing, filling and watershed.

The magic wand itself is a canvas tool (see ``ui/canvas/view.py``); its settings
live in the Magic wand panel. These operations act on the selection it makes.
"""

from ..algorithms import segmentation as alg
from ..algorithms.point_ops import fill_color
from ..core.params import ChoiceParam, FloatParam, IntParam
from ..core.registry import TARGET_SELECTION, Report, operation
from ..i18n import tr

CAT = "Segmentation"
SELECTION = "Selection"
REGIONS = "Regions"

SHAPE = ChoiceParam("shape", "Structuring element", "square", choices=[
    ("square", "Square"),
    ("cross", "Cross"),
    ("disk", "Disk"),
])
RADIUS = IntParam("radius", "Radius [px]", 1, 1, 50)


def _palette_rgb(_obj, ctx) -> dict:
    """Default fill colour: the colour chosen in the palette."""
    color = ctx.get("color", "#ff0000").lstrip("#")
    return {"r": int(color[0:2], 16), "g": int(color[2:4], 16), "b": int(color[4:6], 16)}


@operation("seg.fill", CAT, "Fill selection", section=SELECTION,
           params=[IntParam("r", "R", 255, 0, 255), IntParam("g", "G", 0, 0, 255), IntParam("b", "B", 0, 0, 255)],
           dynamic_defaults=_palette_rgb,
           description="Fills the magic-wand selection (or the region, or the whole image) with a colour.\nDefault: the colour chosen in the palette.")
def fill(image, r, g, b):
    return fill_color(image, r, g, b)


@operation("seg.expand", CAT, "Expand selection", section=SELECTION, target=TARGET_SELECTION,
           params=[RADIUS, SHAPE],
           description="Dilation of the selection mask: the selection grows by the radius.")
def expand(mask, image, radius, shape):
    return alg.dilate_mask(mask, radius, shape)


@operation("seg.shrink", CAT, "Shrink selection", section=SELECTION, target=TARGET_SELECTION,
           params=[RADIUS, SHAPE],
           description="Erosion of the selection mask: the selection shrinks by the radius.")
def shrink(mask, image, radius, shape):
    return alg.erode_mask(mask, radius, shape)


@operation("seg.invert", CAT, "Invert selection", section=SELECTION, target=TARGET_SELECTION,
           description="Selected pixels become unselected and the other way round.")
def invert(mask, image):
    return ~mask


@operation("seg.watershed", CAT, "Watershed", section=REGIONS, slow=True,
           params=[IntParam("smoothing", "Smoothing (3×3 mean passes)", 2, 0, 10),
                   FloatParam("marker_percent", "Markers: flattest pixels [%]", 15.0, 1.0, 60.0, step=1.0, decimals=1),
                   IntParam("min_marker", "Minimum marker size [px]", 20, 1, 10000)],
           description="Marker-based watershed: flooding of the gradient relief from its flattest areas.\nRegions get their mean colour, watershed lines are white.")
def watershed(image, smoothing, marker_percent, min_marker):
    labels, regions = alg.watershed(image, smoothing, marker_percent, min_marker)
    return alg.render_regions(image, labels), Report(tr("Watershed: {n} regions", n=regions), tr("Watershed"))
