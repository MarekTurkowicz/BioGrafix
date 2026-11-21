"""Conversion between RGB pixels and binary 0/1 images."""


def pixels_to_binary(pixels):
    """Convert RGB pixels to a binary 0/1 image."""
    bin_img = []
    for r, g, b in pixels:
        if (r + g + b) / 3 >= 128:
            bin_img.append(1)
        else:
            bin_img.append(0)
    return bin_img


def binary_to_pixels(bin_img):
    """Convert a binary 0/1 image to black and white RGB pixels."""
    pixels = []
    for v in bin_img:
        val = 255 if v else 0
        pixels.append((val, val, val))
    return pixels
