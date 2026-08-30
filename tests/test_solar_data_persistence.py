from __future__ import annotations

import inspect

import cv2
import numpy as np
import pytest

import circle_arc_detector as candidate


def disk_component(shape=(161, 181), center=(90, 80), radius=24, value=220):
    gray = np.zeros(shape, np.uint8)
    component_u8 = np.zeros(shape, np.uint8)
    cv2.circle(component_u8, center, radius, 255, -1)
    component = component_u8 != 0
    gray[component] = value
    return gray, component


def make_state(threshold=100, seed=(90, 80), resolved=True):
    auto = candidate.AutoThresholdResult(
        threshold=threshold,
        histogram_peak=200,
        histogram_left_edge=threshold,
        seed_threshold=threshold,
        coarse_threshold=threshold if resolved else None,
        roi_seed_threshold=threshold if resolved else None,
        full_seed_point=seed if resolved else None,
        used_guard=False,
        resolved=resolved,
        reason="" if resolved else "synthetic unresolved",
    )
    return {
        "settings": candidate.ImageSettings(),
        "auto_threshold_result": auto,
        "solar_data": None,
    }


def test_auto_threshold_stage_has_no_solar_data_dependency():
    source = inspect.getsource(candidate.auto_threshold)
    assert "optimize_separated_threshold" in source
    assert "refine_solar_component_mask" not in source
    assert "SolarData" not in source
    assert "build_solar_data_at_threshold" not in source


def test_packbits_zlib_round_trip_non_multiple_of_eight():
    rng = np.random.default_rng(7)
    mask = rng.random((17, 19)) > 0.73
    payload = candidate.compress_full_mask(mask)
    restored = candidate.decompress_full_mask(payload, mask.shape)
    assert restored.dtype == bool
    assert np.array_equal(restored, mask)


def test_mask_decoder_rejects_wrong_shape():
    mask = np.zeros((7, 11), bool)
    payload = candidate.compress_full_mask(mask)
    with pytest.raises(ValueError, match="packed bytes"):
        candidate.decompress_full_mask(payload, (8, 11))


def test_exact_auto_t_uses_stored_seed_and_persists_refined_component():
    gray, raw = disk_component()
    state = make_state()

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    solar = state["solar_data"]

    assert isinstance(solar, candidate.SolarData)
    assert solar.threshold == 100
    assert solar.seed_point == (90, 80)
    assert np.array_equal(refined, candidate.refine_solar_component_mask(raw))
    stored = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(stored, refined)


def test_invalid_exact_auto_seed_is_error_not_reidentification(monkeypatch):
    gray, _ = disk_component()
    state = make_state(seed=(5, 5))

    monkeypatch.setattr(
        candidate,
        "largest_enclosed_bright_component",
        lambda _binary: (_ for _ in ()).throw(
            AssertionError("exact Auto T must not re-identify the solar seed")
        ),
    )

    with pytest.raises(candidate.ThresholdResolutionError, match="Stored Auto-T solar seed is not light"):
        candidate.build_solar_data_at_threshold(gray, 100, state)
    assert state["solar_data"] is None


def test_resolved_auto_without_full_seed_is_invariant_error():
    gray, _ = disk_component()
    state = make_state()
    state["auto_threshold_result"] = candidate.AutoThresholdResult(
        threshold=100,
        histogram_peak=200,
        histogram_left_edge=100,
        seed_threshold=100,
        coarse_threshold=100,
        roi_seed_threshold=100,
        full_seed_point=None,
        used_guard=False,
        resolved=True,
    )

    with pytest.raises(candidate.ThresholdResolutionError, match="has no full-resolution solar seed"):
        candidate.build_solar_data_at_threshold(gray, 100, state)


def test_non_auto_t_calculates_seed_at_that_exact_threshold():
    gray, _ = disk_component(shape=(241, 301), center=(150, 120), radius=31)
    state = make_state(threshold=90, seed=(150, 120))

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    solar = state["solar_data"]
    x, y = solar.seed_point

    assert solar.threshold == 100
    assert gray[y, x] > 100
    assert refined[y, x]


def test_unresolved_auto_t_is_established_directly_at_current_t():
    gray, _ = disk_component()
    state = make_state(threshold=100, resolved=False)

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    solar = state["solar_data"]
    x, y = solar.seed_point

    assert solar.threshold == 100
    assert refined[y, x]


def test_current_t_without_enclosed_component_fails_without_partial_write():
    gray = np.zeros((120, 140), np.uint8)
    gray[:, 0:20] = 220
    state = make_state(threshold=90, seed=(5, 5))

    with pytest.raises(candidate.ThresholdResolutionError, match="No enclosed solar component"):
        candidate.build_solar_data_at_threshold(gray, 100, state)
    assert state["solar_data"] is None


def test_refinement_seed_survival_is_hard_invariant():
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[25:56, 25:56] = True
    raw[24, 24] = True
    gray[raw] = 220
    state = make_state(threshold=100, seed=(24, 24))

    with pytest.raises(candidate.ThresholdResolutionError, match="no longer contains the solar seed"):
        candidate.build_solar_data_at_threshold(gray, 100, state)
    assert state["solar_data"] is None


def test_roi_guard_and_contour_derive_from_returned_refined_component():
    gray, raw = disk_component(shape=(201, 241), center=(120, 100), radius=26)
    raw = raw.copy()
    raw[73, 93] = True
    raw[100, 120] = False
    gray[:] = 0
    gray[raw] = 220
    state = make_state(threshold=100, seed=(121, 100))

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    solar = state["solar_data"]
    assert np.array_equal(refined, candidate.refine_solar_component_mask(raw))

    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5
    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        restored = candidate.decompress_full_mask(payload, gray.shape)
        region = candidate._build_observation_region(
            gray,
            refined,
            fraction * image_scale,
            solar.seed_point,
        )
        expected = np.zeros(gray.shape, bool)
        x0, y0, x1, y1 = region.bbox
        expected[y0:y1, x0:x1] = region.allowed_u8 != 0
        assert np.array_equal(restored, expected)

    contour = solar.component_contour
    assert contour.dtype == np.uint16
    assert contour.ndim == 2 and contour.shape[1] == 2
    assert np.all(refined[contour[:, 1], contour[:, 0]])


def test_current_solar_data_is_reused_without_reestablishing_identity(monkeypatch):
    gray, _ = disk_component()
    state = make_state()
    first = candidate.build_solar_data_at_threshold(gray, 100, state)
    existing = state["solar_data"]

    monkeypatch.setattr(
        candidate,
        "resize_gray_max_dim",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current SolarData should be reused")
        ),
    )

    second = candidate.build_solar_data_at_threshold(gray, 100, state)
    assert state["solar_data"] is existing
    assert np.array_equal(second, first)


def test_old_split_component_and_builder_apis_are_removed():
    assert not hasattr(candidate, "solar_component_from_seed_at_threshold")
    assert not hasattr(candidate, "establish_solar_component_at_threshold")
    assert not hasattr(candidate, "build_solar_data")
