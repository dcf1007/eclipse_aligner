"""Behavioral regression tests for the validated 3-bin histogram start threshold."""

import numpy as np

import circle_arc_detector as cad


def test_preceding_valley_of_rightmost_mode_is_selected():
    gray = np.concatenate((
        np.full(100, 40, dtype=np.uint8),
        np.full(200, 100, dtype=np.uint8),
    ))
    assert cad.find_histogram_start_threshold(gray) == 98


def test_saturation_mode_still_returns_its_preceding_valley():
    gray = np.concatenate((
        np.full(50, 100, dtype=np.uint8),
        np.full(200, 255, dtype=np.uint8),
    ))
    start_T = cad.find_histogram_start_threshold(gray)
    assert start_T == 253
    assert 0 <= start_T < 255
