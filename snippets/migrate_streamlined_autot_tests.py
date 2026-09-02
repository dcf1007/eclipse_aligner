#!/usr/bin/env python3
"""Migrate retained tests to the approved streamlined Auto-T contracts.

This script intentionally updates only tests whose assertions encode Auto-T helpers,
transient result fields, or observation-region APIs removed by the approved refactor.
Unrelated GUI, morphology, persistence, and topology assertions are left intact.
"""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"


def write(name: str, text: str) -> None:
    (TESTS / name).write_text(text, encoding="utf-8")


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path.name}: expected one {label}, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


write(
    "test_autot_seed_support_refinement.py",
    '''import numpy as np
import pytest
import circle_arc_detector as cad


def test_seed_helper_uses_explicit_kernel_and_returns_none_without_fallback():
    gray = np.zeros((21, 21), np.uint8)
    comp = np.zeros_like(gray)
    comp[10, 3:18] = 255
    gray[10, 10] = 250
    assert cad.brightest_supported_component_point(gray, comp, cad.generate_kernel((5, 5))) is None
    assert cad.brightest_supported_component_point(gray, comp, cad.generate_kernel((1, 1))) == (10, 10)


def test_work_seed_kernel_is_generated_as_fixed_5x5_square():
    kernel = cad.generate_kernel((cad.TRACKING_SEED_KERNEL_SIZE, cad.TRACKING_SEED_KERNEL_SIZE), round_kernel=False)
    assert kernel.shape == (5, 5)
    assert np.all(kernel == 1)


def test_full_seed_support_maps_6016_to_25_square():
    full_shape = (4000, 6016)
    work_shape = (798, 1200)
    mapped = cad.TRACKING_SEED_KERNEL_SIZE * max(full_shape) / float(max(work_shape))
    low = max(1, int(np.floor(mapped)))
    if low % 2 == 0:
        low -= 1
    high = low + 2
    size = low if abs(mapped - low) <= abs(high - mapped) else high
    kernel = cad.generate_kernel((size, size), round_kernel=False)
    assert kernel.shape == (25, 25)
    assert np.all(kernel == 1)


def test_full_seed_support_stays_5_if_no_downscale():
    full_shape = (800, 1200)
    work_shape = full_shape
    mapped = cad.TRACKING_SEED_KERNEL_SIZE * max(full_shape) / float(max(work_shape))
    assert round(mapped) == 5
    assert cad.generate_kernel((5, 5)).shape == (5, 5)


def test_work_search_continues_below_unsupported_candidate():
    gray = np.zeros((31, 31), np.uint8)
    gray[11:20, 11:20] = 15
    gray[14:17, 14:17] = 30
    work_T, component = cad.find_work_res_solar_component(
        gray, 20, cad.generate_kernel((5, 5))
    )
    assert work_T == 0
    assert int(component.sum()) == 81


def test_work_search_errors_if_nothing_supported_through_zero():
    gray = np.zeros((21, 21), np.uint8)
    gray[9:12, 9:12] = 30
    with pytest.raises(cad.ThresholdResolutionError, match="through T=0"):
        cad.find_work_res_solar_component(gray, 20, cad.generate_kernel((5, 5)))


def test_unresolved_auto_is_stored_but_not_returned_as_histogram_fallback():
    gray = np.full((40, 50), 100, np.uint8)
    state = {"settings": cad.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    with pytest.raises(cad.ThresholdResolutionError):
        cad.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert not result.resolved
    assert result.threshold is None
    assert result.work_res_threshold is None
    assert result.full_res_seed_point is None
''',
)


write(
    "test_histogram_peak_3bin.py",
    '''"""Behavioral regression tests for the validated 3-bin histogram start threshold."""

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
''',
)


write(
    "test_threshold_finder.py",
    '''"""Synthetic regression tests for the streamlined grayscale-only Auto-T finder."""
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


def test_histogram_start_threshold_is_left_of_rightmost_mode():
    start_T = tf.find_histogram_start_threshold(synthetic_bridge_image())
    assert 0 <= start_T < 70


def test_unresolved_auto_fails_instead_of_returning_histogram_start_threshold():
    gray = np.full((80, 120), 100, dtype=np.uint8)
    start_T = tf.find_histogram_start_threshold(gray)
    state = {"settings": tf.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    with pytest.raises(tf.ThresholdResolutionError):
        tf.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert not result.resolved
    assert result.threshold is None
    assert result.histogram_start_threshold == start_T
    assert result.work_res_threshold is None


def test_sqrt_pixel_scale_preserves_linear_scaling_and_10_percent_guard():
    base = math.sqrt(6016 * 4000)
    half_linear = math.sqrt(3008 * 2000)
    assert abs(half_linear / base - 0.5) < 1e-12
    assert round(tf.AUTO_T_GUARD_DILATION_FRACTION * base) == 491
    assert round(tf.ROI_DILATION_FRACTION * base) == 319
    assert round(tf.GUARD_DILATION_FRACTION * base) == 957


def test_working_resize_is_1200_max_and_grayscale_area():
    gray = np.arange(2000 * 1000, dtype=np.uint32).reshape(2000, 1000)
    gray = (gray % 256).astype(np.uint8)
    scale = tf.WORK_MAX_DIM / float(max(gray.shape))
    work_size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
    work = tf.resize_img(gray, work_size)
    assert max(work.shape) == 1200
    assert work.shape == (1200, 600)
    assert work.ndim == 2
''',
)


write(
    "test_threshold_topology_integration.py",
    '''"""Integration regression for topology optimization inside automatic thresholding."""
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
''',
)


# Mechanical kernel-generator rename in B's retained morphology tests.
path = TESTS / "test_euclidean_cleanup_candidates.py"
text = path.read_text(encoding="utf-8")
text = text.replace("cad.euclidean_disk_kernel(3)", "cad.generate_kernel((3, 3), round_kernel=True)")
text = text.replace("cad.euclidean_disk_kernel(5)", "cad.generate_kernel((5, 5), round_kernel=True)")
text = text.replace("cad.euclidean_disk_kernel(7)", "cad.generate_kernel((7, 7), round_kernel=True)")
text = text.replace("cad.euclidean_disk_kernel(4)", "cad.generate_kernel((4, 4), round_kernel=True)")
text = text.replace("cad.euclidean_disk_kernel(size)", "cad.generate_kernel((size, size), round_kernel=True)")
text = text.replace("cad.euclidean_disk_kernel(s)", "cad.generate_kernel((s, s), round_kernel=True)")
path.write_text(text, encoding="utf-8")


# Update the agreed-unification integration test without disturbing unrelated GUI tests.
path = TESTS / "test_agreed_threshold_unification.py"
text = path.read_text(encoding="utf-8")
old = '''def test_seed_support_is_explicit_5x5_square_with_no_fallback():
    assert cad.TRACKING_SEED_KERNEL.shape == (5, 5)
    assert np.all(cad.TRACKING_SEED_KERNEL == 1)
    gray = np.zeros((21, 21), np.uint8)
    thin = np.zeros_like(gray)
    thin[10, 4:17] = 255
    gray[10, 10] = 250
    with pytest.raises(cad.ThresholdResolutionError, match="5x5-supported"):
        cad.brightest_supported_component_point(gray, thin, cad.TRACKING_SEED_KERNEL)
'''
new = '''def test_seed_support_is_explicit_5x5_square_with_no_fallback():
    kernel = cad.generate_kernel((cad.TRACKING_SEED_KERNEL_SIZE, cad.TRACKING_SEED_KERNEL_SIZE), round_kernel=False)
    assert kernel.shape == (5, 5)
    assert np.all(kernel == 1)
    gray = np.zeros((21, 21), np.uint8)
    thin = np.zeros_like(gray)
    thin[10, 4:17] = 255
    gray[10, 10] = 250
    assert cad.brightest_supported_component_point(gray, thin, kernel) is None
'''
if old not in text:
    raise RuntimeError("test_agreed_threshold_unification.py: seed-support block changed unexpectedly")
text = text.replace(old, new, 1)
text = text.replace(
    "cad.brightest_supported_component_point(gray, component, cad.TRACKING_SEED_KERNEL)",
    "cad.brightest_supported_component_point(gray, component, cad.generate_kernel((5, 5)))",
)
text = text.replace("result.full_seed_point", "result.full_res_seed_point")
old_ctor = '''    auto = cad.AutoThresholdResult(
        threshold=17,
        histogram_peak=200,
        histogram_left_edge=17,
        seed_threshold=17,
        coarse_threshold=17,
        roi_seed_threshold=17,
        full_seed_point=(2, 2),
        used_guard=False,
        resolved=True,
    )'''
new_ctor = '''    auto = cad.AutoThresholdResult(
        threshold=17,
        histogram_start_threshold=17,
        work_res_threshold=17,
        full_res_seed_point=(2, 2),
        resolved=True,
    )'''
if old_ctor not in text:
    raise RuntimeError("test_agreed_threshold_unification.py: AutoThresholdResult constructor changed unexpectedly")
text = text.replace(old_ctor, new_ctor, 1)
text = text.replace(
    '    assert "full_seed_support = equivalent_full_resolution_seed_kernel(full_gray.shape)" in source\n',
    '    assert "full_res_seed_kernel = generate_kernel" in source\n',
)
path.write_text(text, encoding="utf-8")


# Update the GUI integration fixture's cached AutoThresholdResult shape.
path = TESTS / "test_gui_threshold_integration.py"
text = path.read_text(encoding="utf-8")
old_ctor = '''    result = appmod.AutoThresholdResult(
        threshold=10,
        histogram_peak=20,
        histogram_left_edge=10,
        seed_threshold=15,
        coarse_threshold=12,
        roi_seed_threshold=12,
        full_seed_point=(3, 0),
        used_guard=False,
        resolved=True,
    )'''
new_ctor = '''    result = appmod.AutoThresholdResult(
        threshold=10,
        histogram_start_threshold=10,
        work_res_threshold=12,
        full_res_seed_point=(3, 0),
        resolved=True,
    )'''
if old_ctor not in text:
    raise RuntimeError("test_gui_threshold_integration.py: AutoThresholdResult constructor changed unexpectedly")
path.write_text(text.replace(old_ctor, new_ctor, 1), encoding="utf-8")


# SolarData's 6.5%/19.5% geometry remains identical, but now comes directly from
# dilate_component_mask rather than an ObservationRegion object.
for name in ("test_solar_data_persistence.py", "test_solar_mask_refinement_integration.py"):
    path = TESTS / name
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'''\s*region = candidate\._build_observation_region\(.*?\n'''
        r'''\s*expected = np\.zeros\(gray\.shape, bool\)\n'''
        r'''\s*x0, y0, x1, y1 = region\.bbox\n'''
        r'''\s*expected\[y0:y1, x0:x1\] = region\.allowed_u8 != 0\n''',
        re.DOTALL,
    )
    replacement = "\n        expected = candidate.dilate_component_mask(refined, fraction * image_scale)\n"
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"{name}: expected one ObservationRegion geometry block, found {count}")
    path.write_text(text, encoding="utf-8")


# Explicitly verify that collected tests no longer require removed production names.
forbidden = (
    "TRACKING_SEED_KERNEL",
    "resize_gray_max_dim",
    "find_rightmost_histogram_peak",
    "coarse_threshold_search",
    "equivalent_full_resolution_seed_kernel",
    "establish_full_resolution_seed",
    "find_full_resolution_enclosed_seed_component",
    "_build_observation_region",
    "_evaluate_observation_region",
    "find_lowest_full_threshold",
    "euclidean_disk_kernel",
    ".seed_threshold",
    ".coarse_threshold",
    ".roi_seed_threshold",
    ".full_seed_point",
    ".used_guard",
)
leftovers: list[str] = []
for path in sorted(TESTS.glob("test_*.py")):
    text = path.read_text(encoding="utf-8")
    for token in forbidden:
        # TRACKING_SEED_KERNEL_SIZE is the retained scalar size constant, not the removed kernel object.
        if token == "TRACKING_SEED_KERNEL" and "TRACKING_SEED_KERNEL" in text.replace(
            "TRACKING_SEED_KERNEL_SIZE", ""
        ):
            leftovers.append(f"{path.name}: {token}")
        elif token != "TRACKING_SEED_KERNEL" and token in text:
            leftovers.append(f"{path.name}: {token}")
if leftovers:
    raise RuntimeError("collected tests still reference removed Auto-T APIs:\n" + "\n".join(leftovers))
