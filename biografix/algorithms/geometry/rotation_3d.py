"""Rotating a point in 3D."""

import math


def rotate_point_3d(x, y, z, ax, ay, az):
    """Rotate a point about the X, Y and Z axes in turn (angles in radians)."""
    cosx, sinx = math.cos(ax), math.sin(ax)
    y1 = y * cosx - z * sinx
    z1 = y * sinx + z * cosx
    x1 = x

    cosy, siny = math.cos(ay), math.sin(ay)
    x2 = x1 * cosy + z1 * siny
    z2 = -x1 * siny + z1 * cosy
    y2 = y1

    cosz, sinz = math.cos(az), math.sin(az)
    x3 = x2 * cosz - y2 * sinz
    y3 = x2 * sinz + y2 * cosz
    z3 = z2

    return x3, y3, z3
