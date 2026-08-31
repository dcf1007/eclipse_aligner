from __future__ import annotations

import cv2
import numpy as np
import pytest

import circle_arc_detector as candidate


def make_state(threshold=None):
    return {
        "settings": candidate.ImageSettings(threshold=threshold),
        "auto_threshold_result": None,
        "solar_data": None,
    }


def test_current_t_authoritative_seed_is_chosen_before_refinement_and_survives():
    gray = np.zeros((81, 81), np.uint8)
    cv2.circle(gray, (40, 40), 18, 180, -1)
    gray[40, 40] = 240
    gray[40, 58:66] = 255  # brighter thin spur, deliberately unsupported.

    state = make_state(120)
    refined = candidate.resolve_threshold(gray, 120, state)
    solar = state["solar_data"]

    assert isinstance(solar, candidate.SolarData)
    assert solar.seed_point == (40, 40)
    x, y = solar.seed_point
    assert refined[y, x]
    assert gray[y, x] > 120
    assert not refined[40, 65]


def test_resolver_does_not_reseed_if_authoritative_unrefined_seed_is_removed(monkeypatch):
    gray = np.zeros((81, 81), np.uint8)
    cv2.circle(gray, (40, 40), 18, 180, -1)
    gray[40, 40] = 240
    state = make_state(120)

    real_refine = candidate.refine_solar_component_mask

    def remove_seed(component):
        refined = real_refine(component)
        refined[40, 40] = False
        return refined

    monkeypatch.setattr(candidate, "refine_solar_component_mask", remove_seed)
    with pytest.raises(candidate.ThresholdResolutionError, match="did not survive"):
        candidate.resolve_threshold(gray, 120, state)
    assert state["solar_data"] is None


def test_same_t_reuses_persisted_seed_and_refined_mask(monkeypatch):
    gray = np.zeros((101, 111), np.uint8)
    cv2.circle(gray, (55, 50), 22, 220, -1)
    gray[50, 55] = 250
    state = make_state(130)

    first = candidate.resolve_threshold(gray, 130, state)
    solar = state["solar_data"]
    seed = solar.seed_point

    monkeypatch.setattr(
        candidate,
        "largest_enclosed_bright_component",
        lambda *_args: (_ for _ in ()).throw(AssertionError("same T must reuse SolarData")),
    )
    second = candidate.resolve_threshold(gray, 130, state)
    assert state["solar_data"] is solar
    assert solar.seed_point == seed
    assert np.array_equal(second, first)
