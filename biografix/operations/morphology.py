"""Binary morphology operations."""

from ..algorithms import morphology as alg
from ..core.params import MatrixParam
from ..core.registry import operation
from .adapters import run_on_pixels_wh

CAT = "Morphology"


def _se_param(default="0 1 0\n1 1 1\n0 1 0"):
    return MatrixParam(
        "se", "Structuring element", default, integer=True, allowed=(0, 1, -1),
        help="1 = object, 0 = don't care, -1 = background (hit-or-miss only). The image is thresholded at 128.",
    )


# thinning: classic hit-or-miss corner element
_THIN_DEFAULT = "-1 -1 -1\n0 1 0\n1 1 1"


def _register(mode, op_id, label, default=None):
    @operation(op_id, CAT, label, params=[_se_param(default or "0 1 0\n1 1 1\n0 1 0")],
               section="Binary image", slow=True)
    def run(image, se, _mode=mode):
        return run_on_pixels_wh(alg.apply_morphology, image, se, _mode)

    return run


dilate = _register("dilate", "morph.dilate", "Dilation")
erode = _register("erode", "morph.erode", "Erosion")
opening = _register("open", "morph.open", "Opening")
closing = _register("close", "morph.close", "Closing")
thin = _register("thin", "morph.thin", "Hit-or-miss – thinning", _THIN_DEFAULT)
thicken = _register("thicken", "morph.thicken", "Hit-or-miss – thickening", _THIN_DEFAULT)
