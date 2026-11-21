"""Skeletonization operations: K3M, Zhang-Suen and KMM thinning.

The input is binarized at gray level 128 (threshold the image first, e.g. with
Sauvola for a fingerprint). The report gives the thinning time, which excludes the
one-off loading of the numba-compiled code on the first K3M or KMM run.
"""

import time
from collections.abc import Callable

import numpy as np

from ..algorithms.skeletonization import ObjectColour, mask_to_image, object_mask, zhang_suen_thinning
from ..core.params import ChoiceParam
from ..core.registry import Report, operation
from ..i18n import tr

CAT = "Skeletonization"
SECTION = "Thinning"

OBJECT_COLOUR = ChoiceParam("object_colour", "Object pixels", ObjectColour.BLACK, choices=[
    (ObjectColour.BLACK, "Black (e.g. fingerprint ridges)"),
    (ObjectColour.WHITE, "White"),
], help="The image is binarized at gray level 128 – threshold it first.")


def load_compiled_thinning(thinning: Callable) -> Callable:
    """Run ``thinning`` once on a tiny image so that numba loads or compiles it before timing."""
    warm_up = np.zeros((5, 5), dtype=bool)
    warm_up[1:4, 1:4] = True
    thinning(warm_up)
    return thinning


def preload_compiled_thinning() -> None:
    """Import K3M and KMM and load their compiled code; run in a background thread at start-up."""
    from ..algorithms.skeletonization.k3m import k3m_thinning
    from ..algorithms.skeletonization.kmm import kmm_thinning

    load_compiled_thinning(k3m_thinning)
    load_compiled_thinning(kmm_thinning)


def _thin(image, object_colour: ObjectColour, thinning: Callable, method: str):
    mask = object_mask(image, object_colour)
    started = time.perf_counter()
    skeleton, iterations = thinning(mask)
    seconds = time.perf_counter() - started
    report = Report(tr("{method}: {px} skeleton px from {object_px} object px, {iterations} iterations, {ms} ms",
                       method=method, px=int(skeleton.sum()), object_px=int(mask.sum()),
                       iterations=iterations, ms=f"{seconds * 1000:.1f}"), method)
    return mask_to_image(skeleton, object_colour), report


@operation("skel.k3m", CAT, "K3M", section=SECTION, params=[OBJECT_COLOUR],
           description="Sequential thinning with six phases and lookup arrays (Saeed et al., 2010).\nGives a one-pixel-wide skeleton.")
def k3m(image, object_colour):
    from ..algorithms.skeletonization.k3m import k3m_thinning  # loads numba, which takes a few seconds

    return _thin(image, object_colour, load_compiled_thinning(k3m_thinning), "K3M")


@operation("skel.zhang_suen", CAT, "Zhang-Suen", section=SECTION, params=[OBJECT_COLOUR],
           description="Parallel thinning with two alternating sub-iterations (Zhang, Suen, 1984).")
def zhang_suen(image, object_colour):
    return _thin(image, object_colour, zhang_suen_thinning, "Zhang-Suen")


@operation("skel.kmm", CAT, "KMM", section=SECTION, params=[OBJECT_COLOUR],
           description="Sequential thinning with contour marking and a deletion array (Saeed et al., 2001); the predecessor of K3M.")
def kmm(image, object_colour):
    from ..algorithms.skeletonization.kmm import kmm_thinning  # loads numba, which takes a few seconds

    return _thin(image, object_colour, load_compiled_thinning(kmm_thinning), "KMM")
