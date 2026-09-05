import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _disk_gray(size=81):
    gray = np.zeros((size, size), np.uint8)
    c=size//2
    cv2.circle(gray,(c,c),20,220,-1)
    gray[c,c]=240
    return gray


def test_autot_result_is_mutable_progressive_and_defaults_empty():
    result = cad.AutoThresholdResult()
    assert result.failure_reason is None
    assert result.histogram_start_threshold is None
    assert result.full_res_refined_threshold is None
    result.histogram_start_threshold = 12
    assert result.histogram_start_threshold == 12


def test_stage_a_populates_progressive_result():
    gray=_disk_gray()
    result=cad.AutoThresholdResult()
    cad.find_separation_threshold(gray,result)
    assert result.failure_reason is None
    assert result.histogram_start_threshold is not None
    assert result.work_res_seed_point is not None
    assert result.work_res_separation_threshold is not None
    assert result.work_res_separation_component_mask is not None
    assert result.full_res_seed_point is not None
    assert result.full_res_separation_threshold is not None
    assert result.full_res_separation_component_mask is not None
    assert result.full_res_separation_guard_mask is not None
    assert result.full_res_refined_threshold is None


def test_stage_b_mutates_same_result_and_returns_final_t():
    gray=_disk_gray()
    result=cad.AutoThresholdResult()
    cad.find_separation_threshold(gray,result)
    identity=id(result)
    final_t=cad.refine_threshold(gray,result)
    assert id(result)==identity
    assert final_t==result.full_res_refined_threshold
    assert result.full_res_refined_component_mask is not None


def test_stage_a_failure_persists_failure_reason(monkeypatch):
    gray=_disk_gray()
    result=cad.AutoThresholdResult()
    monkeypatch.setattr(cad,'find_work_res_separation_threshold',lambda *_: (_ for _ in ()).throw(cad.ThresholdResolutionError('no component')))
    with pytest.raises(cad.ThresholdResolutionError,match='no component'):
        cad.find_separation_threshold(gray,result)
    assert result.failure_reason == 'work-resolution separation: no component'


def test_stage_b_rejects_already_failed_result():
    result=cad.AutoThresholdResult(failure_reason='failed')
    with pytest.raises(ValueError,match='already failed'):
        cad.refine_threshold(_disk_gray(),result)
