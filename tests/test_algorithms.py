"""Algorithm and operation registry tests (no GUI)."""

import unittest

import numpy as np

from biografix.algorithms import geometry, thresholding
from biografix.algorithms.color_models import cmyk_to_rgb, rgb_to_cmyk
from biografix.core.imaging import array_to_pixels, paste_region, pixels_to_array, to_rgb
from biografix.core.params import IntParam, MatrixParam
from biografix.core.registry import REGISTRY, TARGET_IMAGE, Report
from biografix.operations import load_all

load_all()


def sample_image(h=40, w=50, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


class TestEveryOperation(unittest.TestCase):
    def test_all_image_operations_run_on_defaults(self):
        img = sample_image()
        for op in REGISTRY.values():
            if op.target != TARGET_IMAGE or op.id == "seg.rembg":  # rembg downloads a model
                continue
            with self.subTest(op=op.id):
                values = {p.name: p.validate(p.default) for p in op.params}
                result = op.func(img.copy(), **values)
                items = result if isinstance(result, tuple) else (result,)
                for item in items:
                    if isinstance(item, np.ndarray):
                        self.assertEqual(item.shape[:2], img.shape[:2])
                        self.assertEqual(item.dtype, np.uint8)


class TestThresholding(unittest.TestCase):
    def test_otsu_separates_two_levels(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, 5:] = 200
        img[:, :5] = 50
        gray = thresholding._gray_average(img)
        t = thresholding.otsu_threshold_value(gray)
        self.assertTrue(50 <= t < 200)
        out = thresholding.otsu_threshold(img)
        self.assertTrue((out[:, :5] == 0).all() and (out[:, 5:] == 255).all())


class TestGeometry(unittest.TestCase):
    def test_bresenham_endpoints(self):
        pts = geometry.bresenham_line(0, 0, 10, 4)
        self.assertEqual(pts[0], (0, 0))
        self.assertEqual(pts[-1], (10, 4))
        self.assertEqual(len(pts), 11)

    def test_rotation_matrix(self):
        (x, y), = geometry.apply_matrix([(10, 0)], geometry.matrix_rotate(0, 0, np.pi / 2))
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 10, places=6)

    def test_bezier_passes_through_endpoints(self):
        pts = [(0, 0), (50, 100), (100, 0)]
        self.assertEqual(geometry.de_casteljau(pts, 0), (0, 0))
        self.assertEqual(geometry.de_casteljau(pts, 1), (100, 0))


class TestFormatsAndParams(unittest.TestCase):
    def test_pixels_roundtrip(self):
        img = sample_image(5, 7)
        pixels, w, h = array_to_pixels(img)
        self.assertTrue(np.array_equal(pixels_to_array(pixels, w, h), img))

    def test_rgba_composited_on_white(self):
        rgba = np.zeros((1, 1, 4), np.uint8)  # black, fully transparent
        self.assertTrue((to_rgb(rgba) == 255).all())

    def test_paste_region(self):
        full = np.zeros((10, 10, 3), np.uint8)
        out = paste_region(full, np.full((2, 3, 3), 9, np.uint8), (1, 2, 4, 4))
        self.assertEqual(int(out.sum()), 2 * 3 * 3 * 9)

    def test_cmyk_roundtrip(self):
        for rgb in [(255, 0, 0), (12, 200, 99), (0, 0, 0)]:
            self.assertEqual(cmyk_to_rgb(*rgb_to_cmyk(*rgb)), rgb)

    def test_param_validation(self):
        with self.assertRaises(ValueError):
            IntParam("w", "Window", 15, 3, 99, odd=True).validate(16)
        se = MatrixParam("se", "SE", integer=True, allowed=(0, 1, -1))
        self.assertEqual(se.validate("0 1\n-1 1"), [[0, 1], [-1, 1]])
        with self.assertRaises(ValueError):
            se.validate("0 2")


if __name__ == "__main__":
    unittest.main()
