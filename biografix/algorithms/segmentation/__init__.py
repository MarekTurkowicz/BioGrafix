"""Segmentation, one algorithm per module.

- ``magic_wand``: selection by a value range around the clicked pixel (local flood fill or global).
- ``selection_morphology``: growing and shrinking a selection with dilation and erosion.
- ``watershed``: marker-based watershed segmentation.
- ``background``: background removal with the RemBG neural network.
"""

from .background import remove_background
from .magic_wand import (
    CompareMode,
    Connectivity,
    flood_fill_from_seed,
    magic_wand_selection,
    pixels_in_value_range,
    select_matching_pixels_globally,
)
from .selection_morphology import StructuringElementShape, dilate_selection, erode_selection, structuring_element
from .watershed import (
    WATERSHED_LINE,
    flood_from_markers,
    label_connected_regions,
    paint_regions_with_mean_colour,
    sobel_gradient_magnitude,
    watershed_segmentation,
)
