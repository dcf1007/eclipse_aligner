"""Behavioral regression tests for the validated 3-bin histogram mode signal."""

import numpy as np

import circle_arc_detector as cad


def test_rightmost_mode_and_preceding_valley_are_selected():
    gray = np.concatenate((
        np.full(100, 40, dtype=np.uint8),
        np.full(200, 100, dtype=np.uint8),
    ))
    assert cad.find_rightmost_histogram_peak(gray) == (100, 98)


def test_saturation_can_be_the_rightmost_mode():
    gray = np.concatenate((
        np.full(50, 100, dtype=np.uint8),
        np.full(200, 255, dtype=np.uint8),
    ))
    peak, left_edge = cad.find_rightmost_histogram_peak(gray)
    assert peak == 255
    assert 0 <= left_edge < peak
