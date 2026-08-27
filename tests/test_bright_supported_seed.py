import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snippets"))
import bright_supported_seed_helper as seed


def test_hot_boundary_pixel_does_not_win_over_supported_solar_core():
    gray = np.zeros((31, 31), np.uint8)
    component = np.zeros_like(gray)
    component[7:24, 7:24] = 255
    gray[7:24, 7:24] = 180
    gray[7, 15] = 255
    gray[14:18, 14:18] = 230
    x, y = seed.brightest_supported_component_point(gray, component)
    assert gray[y, x] == 230


def test_equal_brightness_prefers_deeper_supported_pixel():
    gray = np.zeros((31, 31), np.uint8)
    component = np.zeros_like(gray)
    cv2.circle(component, (15, 15), 10, 255, -1)
    gray[10, 15] = 240
    gray[15, 15] = 240
    assert seed.brightest_supported_component_point(gray, component) == (15, 15)


def test_thin_component_falls_back_deterministically():
    gray = np.zeros((9, 9), np.uint8)
    component = np.zeros_like(gray)
    component[4, 2:7] = 255
    gray[4, 3] = 180
    gray[4, 5] = 220
    assert seed.brightest_supported_component_point(gray, component) == (5, 4)
