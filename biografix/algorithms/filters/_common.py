"""Helpers shared by the pixel-list filters: clamping, 1D/2D reshaping, RGB convolution."""


def _clamp_byte(v: float) -> int:
    v = int(round(v))
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


def _to_2d(pixels, w, h):
    return [pixels[y * w : (y + 1) * w] for y in range(h)]


def _to_1d(rows):
    out = []
    for row in rows:
        out.extend(row)
    return out


def _conv_rgb(pixels, w, h, kernel):
    """Convolve each RGB channel with the kernel."""
    src = _to_2d(pixels, w, h)
    kh = len(kernel)
    kw = len(kernel[0])
    ky = kh // 2
    kx = kw // 2

    out = [[(0, 0, 0) for _ in range(w)] for _ in range(h)]

    for y in range(h):
        for x in range(w):
            acc_r = acc_g = acc_b = 0.0
            for j in range(kh):
                for i in range(kw):
                    yy = y + j - ky
                    xx = x + i - kx
                    if yy < 0:
                        yy = 0
                    if yy >= h:
                        yy = h - 1
                    if xx < 0:
                        xx = 0
                    if xx >= w:
                        xx = w - 1
                    wgt = kernel[j][i]
                    r, g, b = src[yy][xx]
                    acc_r += r * wgt
                    acc_g += g * wgt
                    acc_b += b * wgt
            out[y][x] = (
                _clamp_byte(acc_r),
                _clamp_byte(acc_g),
                _clamp_byte(acc_b),
            )
    return _to_1d(out)
