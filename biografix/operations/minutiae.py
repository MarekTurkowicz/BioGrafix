"""Feature extraction operations: minutiae found with the Crossing Number.

Both operations draw the minutiae on the skeleton (one colour and marker shape
per type) and report how many of each type were found. "Compare minutiae" does
this for every image in the scene at once, so two fingerprints can be compared.
"""

from collections import Counter
from enum import StrEnum

import numpy as np

from ..algorithms.minutiae import (
    COUNTED_TYPES,
    Minutia,
    MinutiaType,
    draw_minutiae,
    find_minutiae,
    fingerprint_area,
    remove_border_minutiae,
)
from ..algorithms.skeletonization import Binarization, ObjectColour, binarized_object_mask, mask_to_image, object_mask
from ..core.params import BoolParam, ChoiceParam
from ..core.registry import TARGET_SCENE, ImageResults, Report, operation
from ..i18n import tr
from .skeletonization import load_compiled_thinning

CAT = "Feature extraction"
SECTION = "Minutiae"

class FingerprintInput(StrEnum):
    """What the processed image contains."""

    RAW = "raw"            # a grayscale fingerprint: binarized (Sauvola) and thinned (K3M) first
    SKELETON = "skeleton"  # already a one-pixel-wide skeleton, black ridges on white


INPUT = ChoiceParam("fingerprint_input", "Input image", FingerprintInput.RAW, choices=[
    (FingerprintInput.RAW, "Fingerprint – binarize (Sauvola) and thin (K3M) first"),
    (FingerprintInput.SKELETON, "Skeleton – black one-pixel-wide ridges"),
])
REMOVE_BORDER = BoolParam("remove_border", "Remove false minutiae at the fingerprint border", True)



def legend() -> str:
    """What the marker colours and shapes mean (matches ``MINUTIA_COLOURS``)."""
    return tr("Red circle – ending, green square – bifurcation, blue diamond – crossing, "
              "orange cross – isolated point, grey – removed at the border.")


def skeleton_of(image: np.ndarray, fingerprint_input: FingerprintInput) -> np.ndarray:
    """Boolean one-pixel-wide skeleton of the fingerprint (True = ridge)."""
    if fingerprint_input == FingerprintInput.SKELETON:
        return object_mask(image, ObjectColour.BLACK)
    from ..algorithms.skeletonization.k3m import k3m_thinning  # loads numba (preloaded at start-up)

    ridges = binarized_object_mask(image, Binarization.SAUVOLA, ObjectColour.BLACK)  # local threshold
    skeleton, _iterations = load_compiled_thinning(k3m_thinning)(ridges)
    return skeleton


def extract_minutiae(image: np.ndarray, fingerprint_input: FingerprintInput, remove_border: bool):
    """Skeleton, kept minutiae and minutiae removed at the border."""
    skeleton = skeleton_of(image, fingerprint_input)
    minutiae = find_minutiae(skeleton)
    if not remove_border:
        return skeleton, minutiae, []
    kept, removed = remove_border_minutiae(minutiae, fingerprint_area(skeleton))
    return skeleton, kept, removed


def count_summary(name: str, kept: list[Minutia], removed: list[Minutia]) -> str:
    """One report line with the number of minutiae of each type."""
    counts = Counter(m.type for m in kept)
    return tr("{name}: {total} minutiae – endings {endings}, bifurcations {bifurcations}, crossings {crossings} "
              "(isolated points: {isolated}, removed at the border: {removed})",
              name=name, total=sum(counts[t] for t in COUNTED_TYPES),
              endings=counts[MinutiaType.ENDING], bifurcations=counts[MinutiaType.BIFURCATION],
              crossings=counts[MinutiaType.CROSSING], isolated=counts[MinutiaType.ISOLATED_POINT],
              removed=len(removed))


def counted_total(kept: list[Minutia]) -> int:
    """Number of minutiae without the isolated points (noise)."""
    return sum(1 for m in kept if m.type in COUNTED_TYPES)


@operation("minutiae.crossing_number", CAT, "Minutiae (Crossing Number)", section=SECTION,
           params=[INPUT, REMOVE_BORDER], supports_roi=False,
           description="Finds minutiae on the skeleton with the Crossing Number:\nCN = 1 ending, CN = 3 bifurcation, CN ≥ 4 crossing.\nEach type gets its own colour and marker.")
def crossing_number(image, fingerprint_input, remove_border):
    skeleton, kept, removed = extract_minutiae(image, fingerprint_input, remove_border)
    report = Report(count_summary(tr("Image"), kept, removed) + "\n" + legend(), tr("Minutiae"))
    return draw_minutiae(mask_to_image(skeleton), kept, removed), report


@operation("minutiae.compare", CAT, "Compare minutiae", section=SECTION, target=TARGET_SCENE,
           params=[INPUT, REMOVE_BORDER],
           description="Finds and highlights the minutiae on every image in the scene\nand lists the number of minutiae of each fingerprint side by side.")
def compare(images, names, fingerprint_input, remove_border):
    drawn, lines, totals = [], [], []
    for image, name in zip(images, names, strict=True):
        skeleton, kept, removed = extract_minutiae(image, fingerprint_input, remove_border)
        drawn.append(draw_minutiae(mask_to_image(skeleton), kept, removed))
        lines.append(count_summary(name, kept, removed))
        totals.append(counted_total(kept))
    if len(totals) == 2:
        lines.append(tr("Difference in the number of minutiae: {n}", n=abs(totals[0] - totals[1])))
    lines.append(legend())
    return ImageResults(drawn), Report("\n".join(lines), tr("Compare minutiae"))
