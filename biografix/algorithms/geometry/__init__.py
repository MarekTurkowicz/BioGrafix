"""Geometry: Bresenham rasterisation, Bézier curves, homogeneous transforms and 3D rotation, one algorithm per module."""

from ._common import Point, Matrix3  # noqa: F401
from .bresenham_line import bresenham_line  # noqa: F401
from .bresenham_circle import bresenham_circle  # noqa: F401
from .rectangle import rect_outline  # noqa: F401
from .bezier import de_casteljau, bezier_points  # noqa: F401
from .transforms_2d import apply_matrix, matrix_translate, matrix_rotate, matrix_scale  # noqa: F401
from .hit_testing import point_in_polygon, distance_to_segment, centroid  # noqa: F401
from .rotation_3d import rotate_point_3d  # noqa: F401
