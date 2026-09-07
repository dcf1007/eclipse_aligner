import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _gray():
    gray = np.zeros((81, 93), np.uint8)
    cv2.circle(gray, (46, 40), 18, 220, -1)
    gray[40, 46] = 240
    return gray


def test_stage_a_exclusively_creates_missing_autot_result():
    state = {"auto_threshold_result": None}
    cad.find_separation_threshold(_gray(), state)
    assert isinstance(state["auto_threshold_result"], cad.AutoThresholdResult)
    assert state["auto_threshold_result"].separation_threshold_complete


def test_stage_a_reuses_same_incomplete_nonfailed_object():
    result = cad.AutoThresholdResult()
    state = {"auto_threshold_result": result}
    cad.find_separation_threshold(_gray(), state)
    assert state["auto_threshold_result"] is result
    assert result.separation_threshold_complete


def test_stage_a_rejects_wrong_non_none_state_without_replacing_it():
    wrong = object()
    state = {"auto_threshold_result": wrong}
    with pytest.raises(ValueError, match="AutoThresholdResult or None"):
        cad.find_separation_threshold(_gray(), state)
    assert state["auto_threshold_result"] is wrong


def test_stage_b_requests_stage_a_instead_of_creating_result_itself(monkeypatch):
    state = {"auto_threshold_result": None}
    calls = []
    real_stage_a = cad.find_separation_threshold

    def stage_a(gray, target_state):
        assert target_state["auto_threshold_result"] is None
        calls.append(target_state)
        return real_stage_a(gray, target_state)

    monkeypatch.setattr(cad, "find_separation_threshold", stage_a)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    selected = cad.refine_threshold(_gray(), state)
    assert calls == [state]
    assert selected is not None
    assert state["auto_threshold_result"].threshold_refinement_complete
