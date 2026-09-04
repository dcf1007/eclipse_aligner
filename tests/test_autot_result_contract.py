import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _small_component(shape=(81, 81), center=(40, 40), radius=20):
    component = np.zeros(shape, dtype=bool)
    cv2.circle(component.view(np.uint8), center, radius, 1, -1)
    return component


def test_auto_threshold_result_defaults_unresolved_and_rejects_contradictory_states():
    unresolved = cad.AutoThresholdResult(
        threshold=None,
        histogram_start_threshold=20,
        work_res_threshold=12,
        full_res_seed_point=(40, 40),
        cleaned_component_mask=None,
        reason="fine refinement: no candidate",
    )
    assert unresolved.resolved is False
    assert unresolved.work_res_threshold == 12
    assert unresolved.full_res_seed_point == (40, 40)

    with pytest.raises(ValueError, match="resolved Auto-T result requires a cleaned component mask"):
        cad.AutoThresholdResult(
            threshold=10,
            histogram_start_threshold=20,
            work_res_threshold=12,
            full_res_seed_point=(40, 40),
            cleaned_component_mask=None,
            resolved=True,
        )

    with pytest.raises(ValueError, match="unresolved Auto-T result cannot contain a final threshold"):
        cad.AutoThresholdResult(
            threshold=10,
            histogram_start_threshold=20,
            work_res_threshold=12,
            full_res_seed_point=(40, 40),
            cleaned_component_mask=None,
        )


def test_resolved_result_requires_complete_auto_t_state_but_reason_is_not_forbidden():
    payload = cad.compress_full_mask(_small_component())
    result = cad.AutoThresholdResult(
        threshold=10,
        histogram_start_threshold=20,
        work_res_threshold=12,
        full_res_seed_point=(40, 40),
        cleaned_component_mask=payload,
        resolved=True,
        reason="diagnostic text is allowed",
    )
    assert result.resolved is True

    for field, value, message in (
        ("threshold", None, "requires a threshold"),
        ("work_res_threshold", None, "requires a work-resolution threshold"),
        ("full_res_seed_point", None, "requires a full-resolution seed point"),
    ):
        kwargs = dict(
            threshold=10,
            histogram_start_threshold=20,
            work_res_threshold=12,
            full_res_seed_point=(40, 40),
            cleaned_component_mask=payload,
            resolved=True,
        )
        kwargs[field] = value
        with pytest.raises(ValueError, match=message):
            cad.AutoThresholdResult(**kwargs)


def _patch_successful_coarse_setup(monkeypatch):
    component = _small_component()
    monkeypatch.setattr(cad, "find_histogram_start_threshold", lambda _gray: 20)
    monkeypatch.setattr(
        cad,
        "find_work_res_solar_component",
        lambda _gray, _start, _kernel: (12, component),
    )
    monkeypatch.setattr(cad, "brightest_supported_component_point", lambda *_args: (40, 40))
    monkeypatch.setattr(cad, "dilate_component_mask", lambda mask, _margin: np.ones_like(mask, dtype=bool))


def test_coarse_separation_failure_stores_unresolved_result_with_stage_and_partial_state(monkeypatch):
    gray = np.zeros((81, 81), dtype=np.uint8)
    state = {}
    _patch_successful_coarse_setup(monkeypatch)

    def fail_separation(*_args):
        raise cad.ThresholdResolutionError("tracked component never separated")

    monkeypatch.setattr(cad, "find_separation_threshold", fail_separation)

    with pytest.raises(cad.ThresholdResolutionError, match="never separated"):
        cad.find_auto_threshold(gray, state)

    result = state["auto_threshold_result"]
    assert result.resolved is False
    assert result.threshold is None
    assert result.cleaned_component_mask is None
    assert result.work_res_threshold == 12
    assert result.full_res_seed_point == (40, 40)
    assert result.reason == "coarse separation: tracked component never separated"


def test_fine_refinement_failure_stores_unresolved_result_with_stage_and_partial_state(monkeypatch):
    gray = np.zeros((81, 81), dtype=np.uint8)
    state = {}
    _patch_successful_coarse_setup(monkeypatch)
    monkeypatch.setattr(cad, "find_separation_threshold", lambda *_args: 8)

    def fail_refinement(*_args):
        raise cad.ThresholdResolutionError("no separated cleaned candidate")

    monkeypatch.setattr(cad, "refine_threshold", fail_refinement)

    with pytest.raises(cad.ThresholdResolutionError, match="no separated cleaned candidate"):
        cad.find_auto_threshold(gray, state)

    result = state["auto_threshold_result"]
    assert result.resolved is False
    assert result.threshold is None
    assert result.cleaned_component_mask is None
    assert result.work_res_threshold == 12
    assert result.full_res_seed_point == (40, 40)
    assert result.reason == "fine refinement: no separated cleaned candidate"


def test_work_resolution_failure_stores_unresolved_result_with_earliest_stage(monkeypatch):
    gray = np.zeros((81, 81), dtype=np.uint8)
    state = {}
    monkeypatch.setattr(cad, "find_histogram_start_threshold", lambda _gray: 20)

    def fail_work(*_args):
        raise cad.ThresholdResolutionError("no supported component")

    monkeypatch.setattr(cad, "find_work_res_solar_component", fail_work)

    with pytest.raises(cad.ThresholdResolutionError, match="no supported component"):
        cad.find_auto_threshold(gray, state)

    result = state["auto_threshold_result"]
    assert result.resolved is False
    assert result.threshold is None
    assert result.work_res_threshold is None
    assert result.full_res_seed_point is None
    assert result.cleaned_component_mask is None
    assert result.reason == "work-resolution component search: no supported component"
