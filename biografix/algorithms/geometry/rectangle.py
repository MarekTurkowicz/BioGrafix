"""Rectangle outline from Bresenham segments."""


from .bresenham_line import bresenham_line


def rect_outline(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """Rectangle outline made of four Bresenham segments."""
    pts = []
    pts += bresenham_line(x1, y1, x2, y1)
    pts += bresenham_line(x2, y1, x2, y2)
    pts += bresenham_line(x2, y2, x1, y2)
    pts += bresenham_line(x1, y2, x1, y1)
    return list(dict.fromkeys(pts))
