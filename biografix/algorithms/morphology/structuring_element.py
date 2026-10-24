"""Parsing a structuring element from text."""


def parse_structuring_element(raw: str):
    """Parse a structuring element: one row per line, values separated by spaces."""
    raw = raw.strip()
    if not raw:
        raise ValueError("The structuring element is empty.")

    rows = []
    width = None
    for line in raw.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            row = [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                "The structuring element may contain only integers (0, 1 and optionally -1)."
            )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("All rows of the structuring element must have the same length.")
        rows.append(row)

    if not rows or width is None:
        raise ValueError("Could not read the structuring element.")

    return rows
