"""Geometry: Bresenham rasterisation, Bézier curves, homogeneous transforms and 3D rotation, one algorithm per module."""

from ._common import Matrix3, Point  # noqa: F401
from .bezier import bezier_points, de_casteljau  # noqa: F401
from .bresenham_circle import bresenham_circle  # noqa: F401
from .bresenham_line import bresenham_line  # noqa: F401
from .hit_testing import centroid, distance_to_segment, point_in_polygon  # noqa: F401
from .rectangle import rect_outline  # noqa: F401
from .rotation_3d import rotate_point_3d  # noqa: F401
from .transforms_2d import apply_matrix, matrix_rotate, matrix_scale, matrix_translate  # noqa: F401
