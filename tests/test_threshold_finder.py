"""Synthetic regression tests for the integrated grayscale-only Auto-T finder."""
import math

import cv2
import numpy as np
import pytest

import circle_arc_detector as tf


def synthetic_bridge_image():
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    cv2.circle(gray, (190, 105), 12, 70, -1)
    return gray


def run_auto(gray):
    state = {"settings": tf.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    threshold = tf.find_auto_threshold(gray, state)
    return threshold, state["auto_threshold_result"]


def test_threshold_semantics_and_lowest_separation():
    threshold, result = run_auto(synthetic_bridge_image())
    assert result.resolved
    assert threshold == result.threshold == 8


def test_find_rightmost_histogram_peak_has_left_edge():
    peak, left = tf.find_rightmost_histogram_peak(synthetic_bridge_image())
    assert peak >= 60
    assert 0 <= left < peak


def test_unresolved_auto_fails_instead_of_returning_histogram_left_edge():
    gray = np.full((80, 120), 100, dtype=np.uint8)
    peak, left = tf.find_rightmost_histogram_peak(gray)
    state = {"settings": tf.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    with pytest.raises(tf.ThresholdResolutionError):
        tf.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert not result.resolved
    assert result.threshold is None
    assert result.seed_threshold is None
    assert result.histogram_peak == peak
    assert result.histogram_left_edge == left


def test_sqrt_pixel_scale_preserves_linear_scaling():
    base = math.sqrt(6016 * 4000)
    half_linear = math.sqrt(3008 * 2000)
    assert abs(half_linear / base - 0.5) < 1e-12
    assert round(tf.ROI_DILATION_FRACTION * base) == 319
    assert round(tf.GUARD_DILATION_FRACTION * base) == 957


def test_working_resize_is_1200_max_and_grayscale_area():
    gray = np.arange(2000 * 1000, dtype=np.uint32).reshape(2000, 1000)
    gray = (gray % 256).astype(np.uint8)
    work = tf.resize_gray_max_dim(gray)
    assert max(work.shape) == 1200
    assert work.shape == (1200, 600)
    assert work.ndim == 2
