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


def make_state(threshold=100):
    return {
        "settings": candidate.ImageSettings(threshold=threshold),
        "auto_threshold_result": None,
        "solar_data": None,
    }


def test_auto_threshold_stage_has_no_solar_data_dependency():
    source = inspect.getsource(candidate.find_auto_threshold)
    assert "optimize_separated_threshold" in source
    assert "refine_solar_component_mask" not in source
    assert "SolarData(" not in source
    assert "resolve_threshold" in source  # docstring only: separation is explicit.


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


def test_resolver_persists_refined_component_and_authoritative_seed():
    gray, raw = disk_component()
    gray[80, 90] = 250
    state = make_state()

    refined = candidate.resolve_threshold(gray, 100, state)
    solar = state["solar_data"]

    assert isinstance(solar, candidate.SolarData)
    assert solar.threshold == 100
    assert solar.seed_point == (90, 80)
    assert np.array_equal(refined, candidate.refine_solar_component_mask(raw))
    stored = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(stored, refined)


def test_auto_search_seed_is_not_special_cased_by_final_t_resolver():
    source = inspect.getsource(candidate.resolve_threshold)
    assert "auto_threshold_result" not in source
    assert "full_seed_point" not in source
    assert "brightest_supported_component_point" in source


def test_current_t_without_enclosed_component_fails_without_partial_write():
    gray = np.zeros((120, 140), np.uint8)
    gray[:, 0:20] = 220
    state = make_state(100)

    with pytest.raises(candidate.ThresholdResolutionError, match="No enclosed solar component"):
        candidate.resolve_threshold(gray, 100, state)
    assert state["solar_data"] is None


def test_unrefined_seed_survival_is_hard_invariant(monkeypatch):
    gray, _ = disk_component()
    gray[80, 90] = 250
    state = make_state()
    real_refine = candidate.refine_solar_component_mask

    def remove_seed(component):
        refined = real_refine(component)
        refined[80, 90] = False
        return refined

    monkeypatch.setattr(candidate, "refine_solar_component_mask", remove_seed)
    with pytest.raises(candidate.ThresholdResolutionError, match="did not survive"):
        candidate.resolve_threshold(gray, 100, state)
    assert state["solar_data"] is None


def test_roi_guard_and_contour_derive_from_returned_refined_component():
    gray, raw = disk_component(shape=(201, 241), center=(120, 100), radius=26)
    raw = raw.copy()
    raw[73, 93] = True
    raw[100, 120] = False
    gray[:] = 0
    gray[raw] = 220
    gray[100, 121] = 250
    state = make_state()

    refined = candidate.resolve_threshold(gray, 100, state)
    solar = state["solar_data"]

    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5
    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        restored = candidate.decompress_full_mask(payload, gray.shape)
        expected = candidate.dilate_component_mask(refined, fraction * image_scale)
        assert np.array_equal(restored, expected)

    contour = solar.component_contour
    assert contour.dtype == np.uint16
    assert contour.ndim == 2 and contour.shape[1] == 2
    assert np.all(refined[contour[:, 1], contour[:, 0]])


def test_current_solar_data_is_reused_without_reestablishing_identity(monkeypatch):
    gray, _ = disk_component()
    gray[80, 90] = 250
    state = make_state()
    first = candidate.resolve_threshold(gray, 100, state)
    existing = state["solar_data"]

    monkeypatch.setattr(
        candidate,
        "largest_enclosed_bright_component",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current SolarData should be reused")
        ),
    )

    second = candidate.resolve_threshold(gray, 100, state)
    assert state["solar_data"] is existing
    assert np.array_equal(second, first)


def test_old_split_builder_apis_are_removed():
    assert not hasattr(candidate, "solar_component_from_seed_at_threshold")
    assert not hasattr(candidate, "establish_solar_component_at_threshold")
    assert not hasattr(candidate, "build_solar_data")
    assert not hasattr(candidate, "build_solar_data_at_threshold")
