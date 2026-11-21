"""Skeletonization operations: K3M, Zhang-Suen and KMM thinning.

The image is binarized first, by default with the local Sauvola threshold: a
fingerprint is often lit unevenly, and a single global threshold (fixed 128 or
Otsu) turns its darker part into one black blob with a false skeleton. An image
that is already binary can use the fixed threshold. The report gives the thinning
time, which excludes the binarization and the one-off loading of the
numba-compiled code on the first K3M or KMM run.
"""

import time
from collections.abc import Callable

import numpy as np

from ..algorithms.skeletonization import (
    LOCAL_WINDOW,
    Binarization,
    ObjectColour,
    binarized_object_mask,
    mask_to_image,
    zhang_suen_thinning,
)
from ..core.params import ChoiceParam, IntParam
from ..core.registry import Report, operation
from ..i18n import tr

CAT = "Skeletonization"
SECTION = "Thinning"

BINARIZATION = ChoiceParam("binarization", "Binarization", Binarization.SAUVOLA, choices=[
    (Binarization.SAUVOLA, "Local – Sauvola (uneven lighting)"),
    (Binarization.PHANSALKAR, "Local – Phansalkar (low contrast)"),
    (Binarization.FIXED, "None – the image is already binary (threshold 128)"),
], help="A global threshold turns a darker part of the fingerprint into one blob with a false skeleton.")
WINDOW = IntParam("window_size", "Local threshold window (odd)", LOCAL_WINDOW, 3, 301, odd=True,
                  help="About two ridge spacings; ignored for an already binary image.")
OBJECT_COLOUR = ChoiceParam("object_colour", "Object pixels", ObjectColour.BLACK, choices=[
    (ObjectColour.BLACK, "Black (e.g. fingerprint ridges)"),
    (ObjectColour.WHITE, "White"),
])
PARAMS = [BINARIZATION, WINDOW, OBJECT_COLOUR]


def load_compiled_thinning(thinning: Callable) -> Callable:
    """Run ``thinning`` once on a tiny image so that numba loads or compiles it before timing."""
    warm_up = np.zeros((5, 5), dtype=bool)
    warm_up[1:4, 1:4] = True
    thinning(warm_up)
    return thinning


def preload_compiled_code() -> None:
    """Load every numba-compiled algorithm (K3M, KMM, watershed); run in a background thread at start-up."""
    from ..algorithms.segmentation import watershed_segmentation
    from ..algorithms.skeletonization.k3m import k3m_thinning
    from ..algorithms.skeletonization.kmm import kmm_thinning

    load_compiled_thinning(k3m_thinning)
    load_compiled_thinning(kmm_thinning)
    watershed_segmentation(np.zeros((8, 8, 3), dtype=np.uint8))


def _thin(image, binarization: Binarization, window_size: int, object_colour: ObjectColour,
          thinning: Callable, method: str):
    mask = binarized_object_mask(image, binarization, object_colour, window_size)
    started = time.perf_counter()
    skeleton, iterations = thinning(mask)
    seconds = time.perf_counter() - started
    report = Report(tr("{method}: {px} skeleton px from {object_px} object px, {iterations} iterations, {ms} ms",
                       method=method, px=int(skeleton.sum()), object_px=int(mask.sum()),
                       iterations=iterations, ms=f"{seconds * 1000:.1f}"), method)
    return mask_to_image(skeleton, object_colour), report


@operation("skel.k3m", CAT, "K3M", section=SECTION, params=PARAMS,
           description="Sequential thinning with six phases and lookup arrays (Saeed et al., 2010).\nGives a one-pixel-wide skeleton.")
def k3m(image, binarization, window_size, object_colour):
    from ..algorithms.skeletonization.k3m import k3m_thinning  # loads numba, which takes a few seconds

    return _thin(image, binarization, window_size, object_colour, load_compiled_thinning(k3m_thinning), "K3M")


@operation("skel.zhang_suen", CAT, "Zhang-Suen", section=SECTION, params=PARAMS,
           description="Parallel thinning with two alternating sub-iterations (Zhang, Suen, 1984).")
def zhang_suen(image, binarization, window_size, object_colour):
    return _thin(image, binarization, window_size, object_colour, zhang_suen_thinning, "Zhang-Suen")


@operation("skel.kmm", CAT, "KMM", section=SECTION, params=PARAMS,
           description="Sequential thinning with contour marking and a deletion array (Saeed et al., 2001); the predecessor of K3M.")
def kmm(image, binarization, window_size, object_colour):
    from ..algorithms.skeletonization.kmm import kmm_thinning  # loads numba, which takes a few seconds

    return _thin(image, binarization, window_size, object_colour, load_compiled_thinning(kmm_thinning), "KMM")
