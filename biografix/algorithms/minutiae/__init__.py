"""Fingerprint feature extraction: minutiae found on a skeleton.

- ``crossing_number``: the Crossing Number of every skeleton pixel and the minutia types.
- ``border``: the fingerprint area and the removal of false minutiae at its edge.
- ``drawing``: coloured markers for each minutia type.
"""

from .border import fill_holes, fingerprint_area, remove_border_minutiae
from .crossing_number import COUNTED_TYPES, Minutia, MinutiaType, crossing_numbers, find_minutiae, minutia_type
from .drawing import MINUTIA_COLOURS, REMOVED_COLOUR, draw_markers, draw_minutiae
