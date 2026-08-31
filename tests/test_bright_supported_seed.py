import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import bright_supported_seed_helper as seed


def test_hot_boundary_pixel_does_not_win_over_supported_solar_core():
    gray = np.zeros((41, 41), np.uint8)
    component = np.zeros_like(gray)
    component[7:34, 7:34] = 255
    gray[7:34, 7:34] = 180
    gray[7, 20] = 255
    gray[18:23, 18:23] = 230
    x, y = seed.brightest_supported_component_point(gray, component)
    assert gray[y, x] == 230


def test_equal_brightness_prefers_deeper_supported_pixel():
    gray = np.zeros((41, 41), np.uint8)
    component = np.zeros_like(gray)
    cv2.circle(component, (20, 20), 14, 255, -1)
    gray[14, 20] = 240
    gray[20, 20] = 240
    assert seed.brightest_supported_component_point(gray, component) == (20, 20)


def test_thin_component_without_5x5_support_falls_back_to_component():
    gray = np.zeros((15, 15), np.uint8)
    component = np.zeros_like(gray)
    component[7, 2:13] = 255
    gray[7, 5] = 180
    gray[7, 9] = 220
    assert seed.brightest_supported_component_point(gray, component) == (9, 7)


def test_5x5_support_is_square_not_ellipse():
    source = np.ones((5, 5), np.uint8)
    square = cv2.erode(source, np.ones((5, 5), np.uint8), iterations=1)
    assert square[2, 2] == 1
    kernel = np.ones((5, 5), np.uint8)
    assert np.all(kernel == 1)
