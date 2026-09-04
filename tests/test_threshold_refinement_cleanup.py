import inspect

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_extract_component_returns_only_seeded_eight_component():
    mask = np.zeros((20, 30), bool)
    mask[2:9, 2:9] = True
    mask[10:18, 18:27] = True
    first = cad.extract_component(mask, (4, 4))
    second = cad.extract_component(mask, (22, 14))
    assert first is not None and second is not None
    assert np.count_nonzero(first) == 49
    assert np.count_nonzero(second) == 72
    assert not np.any(first & second)
    assert cad.extract_component(mask, (15, 5)) is None
    with pytest.raises(ValueError):
        cad.extract_component(mask, (30, 5))


def test_clean_solar_component_runs_progressive_cleanup_guard_then_extract():
    mask = np.zeros((80, 100), bool)
    mask[15:55, 10:50] = True
    mask[20:50, 70:95] = True
    mask[5, 5] = True
    seed = (30, 35)
    guard = np.ones_like(mask, bool)
    guard[:, 90:] = False

    cleaned = cad.clean_solar_component(mask, seed, guard)
    assert cleaned is not None

    expected = np.where(mask, 255, 0).astype(np.uint8)
    for kernel in cad.SOLAR_CLEANUP_KERNELS:
        expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN, kernel, iterations=1)
        expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, kernel, iterations=1)
    expected[~guard] = 0
    expected = cad.extract_component(expected, seed)
    assert expected is not None
    assert np.array_equal(cleaned, expected)


def test_measurements_use_filled_external_silhouette_not_holes():
    solid = np.zeros((11, 11), bool)
    solid[3:8, 3:8] = True
    contour = cad.find_external_contour(solid)
    filled = cad.measure_filled_area(contour)
    rough = cad.measure_roughness(contour, filled)
    solidity = cad.measure_solidity(contour, filled)
    assert filled == 25
    assert rough == pytest.approx(16 / (2 * np.sqrt(np.pi * 25)))
    assert solidity == pytest.approx(1.0)

    holed = solid.copy()
    holed[5, 5] = False
    holed_contour = cad.find_external_contour(holed)
    holed_filled = cad.measure_filled_area(holed_contour)
    assert holed_filled == filled
    assert cad.measure_roughness(holed_contour, holed_filled) == pytest.approx(rough)
    assert cad.measure_solidity(holed_contour, holed_filled) == pytest.approx(solidity)
    assert cad.measure_internal_dark_fraction(int(np.count_nonzero(holed)), holed_filled) == pytest.approx(1 / 25)


def test_refinement_uses_full_masks_and_exactly_one_raw_reference(monkeypatch):
    gray = np.zeros((96, 112), np.uint8)
    cv2.circle(gray, (56, 48), 24, 100, -1)
    guard = np.zeros_like(gray, bool)
    cv2.circle(guard.view(np.uint8), (56, 48), 35, 1, -1)
    seed = (56, 48)

    original_extract = cad.extract_component
    original_clean = cad.clean_solar_component
    extract_shapes = []
    clean_shapes = []

    def count_extract(mask, point):
        extract_shapes.append(np.asarray(mask).shape)
        return original_extract(mask, point)

    def record_clean(mask, point, fixed_guard):
        clean_shapes.append(np.asarray(mask).shape)
        return original_clean(mask, point, fixed_guard)

    monkeypatch.setattr(cad, "extract_component", count_extract)
    monkeypatch.setattr(cad, "clean_solar_component", record_clean)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_args: (0.5, 1.0, 1.0))

    threshold, payload = cad.refine_threshold(gray, 10, seed, guard)

    # T0..T0+10 are all evaluated at full resolution. clean_solar_component calls
    # extract_component once per candidate, plus one additional extraction establishes
    # the first separated raw reference and no further raw components are extracted.
    assert clean_shapes == [gray.shape] * (cad.MAX_T_REFINEMENT_STEPS + 1)
    assert extract_shapes == [gray.shape] * (cad.MAX_T_REFINEMENT_STEPS + 2)

    restored = cad.decompress_full_mask(payload, gray.shape)
    expected = original_clean(cv2.compare(gray, threshold, cv2.CMP_GT), seed, guard)
    assert expected is not None
    assert np.array_equal(restored, expected)


def test_coarse_and_refinement_apply_guard_before_common_extraction(monkeypatch):
    gray = np.zeros((70, 90), np.uint8)
    cv2.circle(gray, (45, 35), 18, 80, -1)
    guard = np.zeros_like(gray, bool)
    cv2.circle(guard.view(np.uint8), (45, 35), 28, 1, -1)
    seed = (45, 35)

    original = cad.extract_component
    calls = []

    def record(mask, point):
        calls.append(np.asarray(mask).copy())
        return original(mask, point)

    monkeypatch.setattr(cad, "extract_component", record)
    cad.find_separation_threshold(gray, 10, seed, guard)
    assert calls and all(not np.any(mask[~guard]) for mask in calls)

    calls.clear()
    assert cad.clean_solar_component(cv2.compare(gray, 10, cv2.CMP_GT), seed, guard) is not None
    assert len(calls) == 1
    assert not np.any(calls[0][~guard])


def test_default_search_has_ten_upward_steps_and_exact_score_decision():
    assert cad.MAX_T_REFINEMENT_STEPS == 10
    source = inspect.getsource(cad.refine_threshold)
    assert "max_steps" not in source
    assert "round(" not in source
    assert "decision_score" not in source
    assert "epsilon" not in source.lower()
    assert "plateau" not in source.lower()
