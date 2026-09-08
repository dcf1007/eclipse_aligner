import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _gray():
    gray = np.zeros((81, 93), np.uint8)
    cv2.circle(gray, (46, 40), 18, 220, -1)
    gray[40, 46] = 240
    return gray


def test_find_auto_threshold_exclusively_creates_missing_autot_result():
    state = {"auto_threshold_result": None}
    selected = cad.find_auto_threshold(_gray(), state)
    assert selected is not None
    assert isinstance(state["auto_threshold_result"], cad.AutoThresholdResult)
    assert state["auto_threshold_result"].threshold_refinement_complete


def test_find_auto_threshold_reuses_same_incomplete_nonfailed_object():
    result = cad.AutoThresholdResult()
    state = {"auto_threshold_result": result}
    selected = cad.find_auto_threshold(_gray(), state)
    assert selected is not None
    assert state["auto_threshold_result"] is result
    assert result.threshold_refinement_complete


def test_find_auto_threshold_rejects_wrong_non_none_state_without_replacing_it():
    wrong = object()
    state = {"auto_threshold_result": wrong}
    with pytest.raises(ValueError, match="AutoThresholdResult or None"):
        cad.find_auto_threshold(_gray(), state)
    assert state["auto_threshold_result"] is wrong


def test_find_auto_threshold_runs_stage_a_then_stage_b_on_same_result(monkeypatch):
    state = {"auto_threshold_result": None}
    calls = []
    real_stage_a = cad.find_separation_threshold
    real_stage_b = cad.refine_threshold

    def stage_a(gray, target_state):
        calls.append(("a", id(target_state["auto_threshold_result"])))
        return real_stage_a(gray, target_state)

    def stage_b(gray, target_state):
        calls.append(("b", id(target_state["auto_threshold_result"])))
        return real_stage_b(gray, target_state)

    monkeypatch.setattr(cad, "find_separation_threshold", stage_a)
    monkeypatch.setattr(cad, "refine_threshold", stage_b)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    selected = cad.find_auto_threshold(_gray(), state)
    result = state["auto_threshold_result"]
    assert selected is not None
    assert calls == [("a", id(result)), ("b", id(result))]
    assert result.threshold_refinement_complete
