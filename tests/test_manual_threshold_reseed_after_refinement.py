from __future__ import annotations

import cv2
import numpy as np

import circle_arc_detector as candidate


def make_state(auto_threshold=100, auto_seed=(40, 40)):
    return {
        "settings": candidate.ImageSettings(),
        "auto_threshold_result": candidate.AutoThresholdResult(
            threshold=auto_threshold,
            histogram_peak=220,
            histogram_left_edge=auto_threshold,
            seed_threshold=auto_threshold,
            coarse_threshold=auto_threshold,
            roi_seed_threshold=auto_threshold,
            full_seed_point=auto_seed,
            used_guard=False,
            resolved=True,
        ),
        "solar_data": None,
    }


def test_manual_t_reseeds_from_refined_component_when_raw_flood_seed_is_removed(monkeypatch):
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[25:56, 25:56] = True
    raw[24, 24] = True  # 8-connected one-pixel spur removed by 7x7 opening.
    gray[raw] = 220

    state = make_state(auto_threshold=100, auto_seed=(40, 40))
    manual_threshold = 120

    monkeypatch.setattr(candidate, "resize_gray_max_dim", lambda image: image)
    monkeypatch.setattr(
        candidate,
        "largest_enclosed_bright_component",
        lambda binary: np.asarray(binary, dtype=bool),
    )
    monkeypatch.setattr(
        candidate,
        "establish_full_resolution_seed",
        lambda *_args: (24, 24),
    )

    refined = candidate.build_solar_data_at_threshold(
        gray,
        manual_threshold,
        state,
    )
    solar = state["solar_data"]

    assert isinstance(solar, candidate.SolarData)
    assert solar.threshold == manual_threshold
    assert not refined[24, 24]
    assert solar.seed_point != (24, 24)
    seed_x, seed_y = solar.seed_point
    assert refined[seed_y, seed_x]
    assert gray[seed_y, seed_x] > manual_threshold


def test_exact_auto_t_still_requires_authoritative_auto_seed_to_survive_refinement():
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[25:56, 25:56] = True
    raw[24, 24] = True
    gray[raw] = 220

    state = make_state(auto_threshold=100, auto_seed=(24, 24))

    try:
        candidate.build_solar_data_at_threshold(gray, 100, state)
    except candidate.ThresholdResolutionError as exc:
        assert "Auto-T solar seed" in str(exc)
    else:
        raise AssertionError("exact Auto-T seed-removal invariant should still fail")


def test_manual_t_final_seed_is_inside_persisted_refined_mask(monkeypatch):
    gray = np.zeros((101, 111), np.uint8)
    cv2.circle(gray, (55, 50), 22, 220, -1)
    state = make_state(auto_threshold=90, auto_seed=(55, 50))

    refined = candidate.build_solar_data_at_threshold(gray, 130, state)
    solar = state["solar_data"]
    stored = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    x, y = solar.seed_point

    assert np.array_equal(stored, refined)
    assert refined[y, x]
    assert gray[y, x] > 130
