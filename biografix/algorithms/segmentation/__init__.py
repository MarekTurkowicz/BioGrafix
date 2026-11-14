"""Segmentation, one algorithm per module.

- ``magic_wand``: selection by a value range around the clicked pixel (local flood fill or global).
- ``selection_morphology``: growing and shrinking a selection with dilation and erosion.
- ``watershed``: marker-based watershed segmentation.
- ``background``: background removal with the RemBG neural network.
"""

from .background import remove_background
from .magic_wand import COMPARE_MODES, flood_select, global_select, magic_wand, range_mask
from .selection_morphology import SE_SHAPES, dilate_mask, erode_mask, structuring_element
from .watershed import LINE, flood, label_components, render_regions, sobel_magnitude, watershed
