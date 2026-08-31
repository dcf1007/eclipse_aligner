import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

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


def test_thin_component_without_7x7_support_is_error_not_fallback():
    gray = np.zeros((15, 15), np.uint8)
    component = np.zeros_like(gray)
    component[7, 2:13] = 255
    gray[7, 5] = 180
    gray[7, 9] = 220
    with pytest.raises(seed.ThresholdResolutionError, match="no 7x7-supported interior seed"):
        seed.brightest_supported_component_point(gray, component)


def test_support_kernel_is_shared_shape_contract():
    assert seed.SOLAR_COMPONENT_KERNEL_SIZE == 7
    assert seed.SOLAR_COMPONENT_KERNEL.shape == (7, 7)
    assert seed.SOLAR_COMPONENT_KERNEL[3, 3] == 1
    assert seed.SOLAR_COMPONENT_KERNEL[0, 0] == 0
