import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_measure_filled_area_includes_lattice_boundary_without_helper():
    mask=np.zeros((11,11),bool); mask[3:8,3:8]=True
    contour=cad.find_external_contour(mask)
    assert cad.measure_filled_area(contour)==25


def test_refinement_window_is_base_through_base_plus_ten(monkeypatch):
    gray=np.zeros((61,61),np.uint8); cv2.circle(gray,(30,30),18,180,-1); gray[30,30]=240
    result=cad.AutoThresholdResult(full_res_seed_point=(30,30), full_res_separation_threshold=100)
    guard=np.ones_like(gray,bool)
    result.full_res_separation_guard_mask=cad.compress_array(guard)
    seen=[]
    real=cad.morphological_cleanup
    def record(source,kernel,threshold=None):
        if threshold is not None: seen.append(threshold)
        return real(source,kernel,threshold)
    monkeypatch.setattr(cad,'morphological_cleanup',record)
    monkeypatch.setattr(cad,'measure_edge_alignment',lambda *_:(0.5,1.0,1.0))
    cad.refine_threshold(gray,result)
    # Stage B thresholds once then applies P357 to the existing mask, so no threshold-mode morphology here.
    assert seen == []
    assert 100 <= result.full_res_refined_threshold <= 110


def test_refinement_inlines_progressive_p357_cleanup(monkeypatch):
    gray=np.zeros((81,81),np.uint8); cv2.circle(gray,(40,40),20,180,-1); gray[40,40]=240
    result=cad.AutoThresholdResult(full_res_seed_point=(40,40),full_res_separation_threshold=100,full_res_separation_guard_mask=cad.compress_array(np.ones_like(gray,bool)))
    calls=[]
    real=cad.morphological_cleanup
    def record(source,kernel,threshold=None):
        calls.append(kernel.shape)
        return real(source,kernel,threshold)
    monkeypatch.setattr(cad,'morphological_cleanup',record)
    monkeypatch.setattr(cad,'measure_edge_alignment',lambda *_:(0.5,1.0,1.0))
    cad.refine_threshold(gray,result)
    assert calls[:3] == [(3,3),(5,5),(7,7)]
    assert len(calls) % 3 == 0
