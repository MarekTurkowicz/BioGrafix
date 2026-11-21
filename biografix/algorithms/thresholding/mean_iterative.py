"""Mean Iterative Selection."""

from ._common import _apply_threshold_to_pixels, _to_gray_list


def threshold_mean_iterative(pixels, max_iter=100, eps=0.5):
    """Mean Iterative Selection: iterate T towards the mean of both class means."""
    vals = _to_gray_list(pixels)
    n = len(vals)
    if n == 0:
        return pixels[:]

    # start from the global mean
    T = sum(vals) / n

    for _ in range(max_iter):
        g1 = [v for v in vals if v <= T]
        g2 = [v for v in vals if v > T]
        if not g1 or not g2:
            break
        m1 = sum(g1) / len(g1)
        m2 = sum(g2) / len(g2)
        new_T = (m1 + m2) / 2.0
        if abs(new_T - T) < eps:
            T = new_T
            break
        T = new_T

    T_int = int(round(T))
    return _apply_threshold_to_pixels(pixels, T_int)
