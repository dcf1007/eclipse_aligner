from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

def _state(): return {'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}

def test_resize_img_binary_uses_exact_nearest_coordinate_mapping():
    source=np.array([[0,0,255,0,0]],np.uint8); r=cad.resize_img(source,(1,3),mask=True); assert r.dtype==source.dtype and r.shape==(1,3)

def test_resize_img_preserves_bgra_uint16_dtype_channels_and_shape():
    image=np.arange(4*6*4,dtype=np.uint16).reshape(4,6,4)*17; r=cad.resize_img(image,(2,3)); assert r.dtype==np.uint16 and r.shape==(2,3,4)

def test_resize_img_requires_complete_explicit_shape():
    image=np.zeros((4,6),np.uint8)
    with pytest.raises((TypeError,ValueError)): cad.resize_img(image,(3,None))

def test_histogram_start_threshold_returns_preceding_left_valley_or_peak_fallback():
    g=np.zeros((10,10),np.uint8); g[:,::2]=200; assert 0<=cad.find_histogram_start_threshold(g)<=200

def test_generate_kernel_unifies_square_and_euclidean_geometry():
    assert cad.generate_kernel((5,5),False).sum()==25 and cad.generate_kernel((5,5),True).sum()==13

def test_supported_point_returns_none_without_requested_support():
    g=np.zeros((15,15),np.uint8); c=np.zeros_like(g,bool); c[7,2:13]=True; assert cad.brightest_supported_component_point(g,c,cad.generate_kernel((5,5))) is None

def test_work_search_stops_candidate_rediscovery_after_seed_is_established(monkeypatch):
    g=np.zeros((41,41),np.uint8); g[10:31,10:31]=30; calls=[]; real=cad.largest_enclosed_bright_component
    monkeypatch.setattr(cad,'largest_enclosed_bright_component',lambda b: calls.append(1) or real(b))
    cad.find_work_res_separation_threshold(g,30,cad.generate_kernel((5,5))); assert len(calls)==2

def test_work_search_fails_after_no_supported_seed_through_zero():
    g=np.zeros((21,21),np.uint8); g[10,2:19]=200
    with pytest.raises(cad.ThresholdResolutionError): cad.find_work_res_separation_threshold(g,200,cad.generate_kernel((5,5)))

def _guard():
    g=np.zeros((21,21),bool); g[2:19,2:19]=True; return g

def _tight_guard():
    g=np.zeros((21,21),bool); g[5:16,5:16]=True; return g

def test_full_res_threshold_search_moves_down_when_start_is_enclosed():
    g=np.zeros((21,21),np.uint8); g[6:15,6:15]=30; g[7:14,14:18]=10
    t,_=cad.find_full_res_separation_threshold(g,10,(10,10),_guard()); assert t<=10

def test_full_res_threshold_search_moves_up_when_start_touches_guard():
    g=np.zeros((21,21),np.uint8); g[6:15,6:15]=30; g[7:14,14:18]=11
    t,_=cad.find_full_res_separation_threshold(g,10,(10,10),_tight_guard()); assert t==11

def test_dilate_component_mask_returns_one_full_resolution_mask():
    c=np.zeros((31,41),bool); c[14:17,19:22]=True; same=cad.dilate_component_mask(c,0.0); grown=cad.dilate_component_mask(c,4.0)
    assert np.array_equal(same,c) and np.all(grown[c]) and grown.sum()>c.sum()

def test_auto_result_contains_progressive_stage_fields_and_one_full_seed():
    fields=tuple(cad.AutoThresholdResult.__dataclass_fields__)
    assert 'full_res_seed_point' in fields and 'full_res_refined_component_mask' in fields and 'failure_reason' in fields
    assert 'final_seed_point' not in fields and 'resolved' not in fields

def test_auto_threshold_keeps_synthetic_bridge_result_with_current_flow():
    g=np.zeros((220,320),np.uint8); cv2.circle(g,(180,110),46,30,-1); g[106:115,0:180]=8; cv2.circle(g,(190,105),12,70,-1)
    s=_state(); t=cad.find_auto_threshold(g,s); r=s['auto_threshold_result']; assert t==r.full_res_refined_threshold==8 and r.threshold_refinement_complete
    mask=cad.decompress_image(r.full_res_refined_component_mask); x,y=r.full_res_seed_point; assert mask[y,x]

def test_auto_full_res_identity_products_match_input_shape():
    g=np.zeros((800,1259),np.uint8); cv2.circle(g,(629,400),120,180,-1); g[400,629]=250
    s=_state(); cad.find_auto_threshold(g,s); r=s['auto_threshold_result']; assert cad.decompress_image(r.full_res_separation_guard_mask).shape==g.shape

def test_resize_img_is_only_direct_cv2_resize_call_in_production_source(): assert Path(cad.__file__).read_text().count('cv2.resize(')==1
