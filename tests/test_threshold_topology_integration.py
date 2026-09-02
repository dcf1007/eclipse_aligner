"""Integration regression for topology optimization inside automatic thresholding."""
import math

import cv2
import numpy as np

import circle_arc_detector as tf


def synthetic_bridge_with_separated_appendage():
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    gray[106:115, 225:285] = 9
    for x in range(225, 280, 10):
        gray[90:130, x:x+3] = 9
    cv2.circle(gray, (190, 105), 12, 70, -1)
    return gray


def auto(gray):
    state = {"settings": tf.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    threshold = tf.find_auto_threshold(gray, state)
    return threshold, state["auto_threshold_result"]


def base_separated_t(gray):
    full_res_height, full_res_width = gray.shape
    if max(full_res_height, full_res_width) > tf.WORK_MAX_DIM:
        scale = tf.WORK_MAX_DIM / float(max(full_res_height, full_res_width))
        work_res_size = (
            max(1, round(full_res_width * scale)),
            max(1, round(full_res_height * scale)),
        )
    else:
        work_res_size = (full_res_width, full_res_height)
    work_res_gray = tf.resize_img(gray, work_res_size)
    start_T = tf.find_histogram_start_threshold(work_res_gray)
    work_res_T, work_res_component = tf.find_work_res_solar_component(
        work_res_gray, start_T, tf.generate_kernel((tf.TRACKING_SEED_KERNEL_SIZE, tf.TRACKING_SEED_KERNEL_SIZE))
    )
    full_res_search_mask = tf.resize_img(work_res_component, (full_res_width, full_res_height))
    assert full_res_search_mask.shape == gray.shape

    mapped = tf.TRACKING_SEED_KERNEL_SIZE * max(gray.shape) / float(max(work_res_gray.shape))
    low = max(1, int(math.floor(mapped)))
    if low % 2 == 0:
        low -= 1
    high = low + 2
    kernel_size = low if abs(mapped - low) <= abs(high - mapped) else high
    full_res_seed = tf.brightest_supported_component_point(
        gray, full_res_search_mask, tf.generate_kernel((kernel_size, kernel_size))
    )
    assert full_res_seed is not None
    image_scale = math.sqrt(float(full_res_width) * float(full_res_height))
    guard = tf.dilate_component_mask(
        full_res_search_mask, tf.AUTO_T_GUARD_DILATION_FRACTION * image_scale
    )
    separated_t, _component = tf.find_lowest_full_res_threshold(
        gray, work_res_T, full_res_seed, guard
    )
    return separated_t


def test_find_auto_threshold_optimizes_above_lowest_separated_threshold():
    gray = synthetic_bridge_with_separated_appendage()
    assert base_separated_t(gray) == 8

    threshold, result = auto(gray)
    assert result.resolved
    assert threshold == result.threshold == 9


def test_find_auto_threshold_keeps_separation_t_when_no_cleanup_is_available():
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    cv2.circle(gray, (190, 105), 12, 70, -1)

    threshold, result = auto(gray)
    assert result.resolved
    assert threshold == result.threshold == 8
