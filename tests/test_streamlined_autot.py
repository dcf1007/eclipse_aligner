from __future__ import annotations
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad


def _state(): return {'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}

def test_resize_img_binary_uses_exact_nearest_coordinate_mapping():
    source=np.array([[0,0,255,0,0]],np.uint8); resized=cad.resize_img(source,(3,1),mask=True)
    assert resized.dtype==source.dtype and resized.shape==(1,3) and resized.tolist()==[[0,255,0]]

def test_resize_img_preserves_bgra_uint16_dtype_channels_and_aspect_ratio():
    image=np.arange(4*6*4,dtype=np.uint16).reshape(4,6,4)*17; resized=cad.resize_img(image,(3,2)); assert resized.dtype==np.uint16 and resized.shape==(2,3,4)

def test_resize_img_requires_complete_explicit_size():
    image=np.arange(24,dtype=np.uint8).reshape(4,6)
    with pytest.raises((TypeError,ValueError)): cad.resize_img(image,(3,None))
    with pytest.raises((TypeError,ValueError)): cad.resize_img(image,(None,2))

def test_histogram_start_threshold_returns_only_preceding_left_valley():
    gray=np.concatenate((np.full(100,40,np.uint8),np.full(200,100,np.uint8))); assert cad.find_histogram_start_threshold(gray)==98

def test_generate_kernel_unifies_square_and_exact_euclidean_disk_geometry():
    square=cad.generate_kernel((5,5)); assert np.all(square==1)
    for size,pixels in {3:5,5:13,7:29}.items():
        disk=cad.generate_kernel((size,size),round_kernel=True); assert int(disk.sum())==pixels and np.array_equal(disk,np.rot90(disk))
    with pytest.raises(ValueError): cad.generate_kernel((4,4))

def test_supported_point_returns_none_without_requested_support():
    gray=np.zeros((21,21),np.uint8); comp=np.zeros_like(gray); comp[10,3:18]=255; gray[10,10]=250
    assert cad.brightest_supported_component_point(gray,comp,cad.generate_kernel((5,5))) is None
    assert cad.brightest_supported_component_point(gray,comp,cad.generate_kernel((1,1)))==(10,10)

def test_work_search_stops_rediscovering_candidates_after_seed_is_established(monkeypatch):
    gray=np.zeros((31,31),np.uint8); gray[11:20,11:20]=15; gray[14:17,14:17]=30
    real=cad.largest_enclosed_bright_component; calls=[]
    def counted(binary): calls.append(1); return real(binary)
    monkeypatch.setattr(cad,'largest_enclosed_bright_component',counted)
    work_T,component=cad.find_work_res_solar_component(gray,20,cad.generate_kernel((5,5)))
    assert len(calls)==7 and work_T==0 and int(component.sum())==81

def test_work_search_fails_only_after_no_supported_seed_exists_through_zero():
    gray=np.zeros((21,21),np.uint8); gray[9:12,9:12]=30
    with pytest.raises(cad.ThresholdResolutionError,match='through T=0'): cad.find_work_res_solar_component(gray,20,cad.generate_kernel((5,5)))

def _guard(shape=(21,21)):
    g=np.zeros(shape,bool); g[3:18,3:18]=True; return g

def test_full_res_threshold_search_moves_down_only_when_start_is_enclosed():
    gray=np.zeros((21,21),np.uint8); gray[6:15,6:15]=30; gray[7:14,14:18]=10
    assert cad.find_lowest_full_res_threshold(gray,10,(10,10),_guard())==10

def test_full_res_threshold_search_moves_up_only_when_start_touches_guard():
    gray=np.zeros((21,21),np.uint8); gray[6:15,6:15]=30; gray[7:14,14:18]=11
    assert cad.find_lowest_full_res_threshold(gray,10,(10,10),_guard())==11

def test_dilate_component_mask_returns_only_one_full_resolution_mask():
    c=np.zeros((31,41),bool); c[14:17,19:22]=True; same=cad.dilate_component_mask(c,0.0); grown=cad.dilate_component_mask(c,4.0)
    assert np.array_equal(same,c) and np.all(grown[c]) and int(grown.sum())>int(c.sum())

def test_auto_result_contains_refined_mask_and_no_duplicate_seed():
    fields=tuple(cad.AutoThresholdResult.__dataclass_fields__)
    assert fields==('threshold','histogram_start_threshold','work_res_threshold','full_res_seed_point','resolved','cleaned_component_mask','reason')
    assert 'final_seed_point' not in fields

def test_auto_threshold_keeps_synthetic_bridge_result_with_streamlined_flow():
    gray=np.zeros((220,320),np.uint8); cv2.circle(gray,(180,110),46,30,-1); gray[106:115,0:180]=8; cv2.circle(gray,(190,105),12,70,-1)
    state=_state(); threshold=cad.find_auto_threshold(gray,state); result=state['auto_threshold_result']
    assert result.resolved and threshold==result.threshold==8 and result.work_res_threshold==8 and result.full_res_seed_point is not None
    assert result.cleaned_component_mask is not None
    mask=cad.decompress_full_mask(result.cleaned_component_mask,gray.shape); x,y=result.full_res_seed_point; assert mask[y,x]

def test_auto_maps_work_component_directly_to_exact_full_res_size(monkeypatch):
    gray=np.zeros((800,1259),np.uint8); monkeypatch.setattr(cad,'find_histogram_start_threshold',lambda _g:20)
    def fake_work(work,_t,_k): c=np.zeros(work.shape,bool); c[300:450,500:700]=True; return 12,c
    monkeypatch.setattr(cad,'find_work_res_solar_component',fake_work); seen={}
    def fake_seed(_g,mask,_k): seen['shape']=mask.shape; return (629,400)
    monkeypatch.setattr(cad,'brightest_supported_component_point',fake_seed)
    monkeypatch.setattr(cad,'dilate_component_mask',lambda mask,_m:np.ones(mask.shape,bool))
    monkeypatch.setattr(cad,'find_lowest_full_res_threshold',lambda _g,t,_s,_guard:t)
    payload=cad.compress_full_mask(np.ones(gray.shape,bool))
    monkeypatch.setattr(cad,'refine_threshold',lambda _g,t,_s,_guard:SimpleNamespace(threshold=t,cleaned_component_mask=payload))
    assert cad.find_auto_threshold(gray,_state())==12 and seen['shape']==gray.shape

def test_resize_img_is_the_only_direct_cv2_resize_call_in_production_source():
    from pathlib import Path
    assert Path(cad.__file__).read_text().count('cv2.resize(')==1
