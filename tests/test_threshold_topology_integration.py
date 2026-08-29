"""Integration regression for topology optimization inside automatic thresholding."""

import cv2
import numpy as np

import circle_arc_detector as tf


def synthetic_bridge_with_separated_appendage():
    """Lowest separated T is 8; noisy gray=9 appendages should move final T to 9."""
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    gray[106:115, 225:285] = 9
    for x in range(225, 280, 10):
        gray[90:130, x:x+3] = 9
    cv2.circle(gray, (190, 105), 12, 70, -1)
    return gray


def test_auto_threshold_optimizes_above_lowest_separated_threshold():
    gray = synthetic_bridge_with_separated_appendage()

    work = tf.resize_gray_max_dim(gray)
    coarse = tf.coarse_threshold_search(work)
    seed = tf.establish_full_resolution_seed(
        gray, work.shape, coarse.seed_mask, coarse.seed_threshold
    )
    _roi_seed_t, roi_seed_component = tf.find_full_resolution_enclosed_seed_component(
        gray, coarse.threshold, seed
    )
    separated_t, _used_guard = tf.find_lowest_full_threshold(
        gray, seed, roi_seed_component
    )
    assert separated_t == 8

    result = tf.auto_threshold(gray)
    assert result.resolved
    assert result.threshold == 9


def test_auto_threshold_keeps_separation_t_when_no_cleanup_is_available():
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    cv2.circle(gray, (190, 105), 12, 70, -1)

    result = tf.auto_threshold(gray)
    assert result.resolved
    assert result.threshold == 8
