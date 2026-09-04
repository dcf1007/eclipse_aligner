from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import circle_arc_detector as cad

SOURCE = Path(cad.__file__).read_text(encoding='utf-8')


def _state():
    return {'settings': cad.ImageSettings(), 'auto_threshold_result': None, 'solar_data': None}


def test_source_seed_is_checked_after_d7_not_before():
    gray=np.zeros((31,31),np.uint8); gray[5:26,5:26]=30; gray[15,15]=10
    guard=np.zeros(gray.shape,bool); guard[2:29,2:29]=True
    assert cad.find_lowest_full_res_threshold(gray,10,(15,15),guard)==0


def test_source_seed_must_still_survive_d7_cleanup():
    gray=np.zeros((31,31),np.uint8); gray[13:18,13:18]=30
    guard=np.zeros(gray.shape,bool); guard[2:29,2:29]=True
    with pytest.raises(cad.ThresholdResolutionError,match='does not survive D7 cleanup'):
        cad.find_lowest_full_res_threshold(gray,10,(15,15),guard)


def test_find_auto_threshold_requires_authoritative_uint8_gray():
    with pytest.raises(ValueError,match='requires authoritative uint8 grayscale'):
        cad.find_auto_threshold(np.zeros((31,31),np.uint16),_state())


def test_source_support_mapping_uses_actual_work_kernel(monkeypatch):
    gray=np.zeros((120,240),np.uint8); monkeypatch.setattr(cad,'WORK_RES_MAX_DIM',120)
    monkeypatch.setattr(cad,'find_histogram_start_threshold',lambda _gray:20)
    real_generate=cad.generate_kernel; first=True
    def generated(size,round_kernel=False):
        nonlocal first
        if first and size==(5,5) and not round_kernel:
            first=False; return np.ones((7,7),np.uint8)
        return real_generate(size,round_kernel=round_kernel)
    monkeypatch.setattr(cad,'generate_kernel',generated)
    def work(work_gray,_start,received):
        assert received.shape==(7,7); c=np.zeros(work_gray.shape,bool); c[10:50,30:90]=True; return 12,c
    monkeypatch.setattr(cad,'find_work_res_solar_component',work)
    seen={}
    def seed(_gray,_mask,kernel): seen['shape']=kernel.shape; return (120,60)
    monkeypatch.setattr(cad,'brightest_supported_component_point',seed)
    monkeypatch.setattr(cad,'dilate_component_mask',lambda mask,_margin:np.ones(mask.shape,bool))
    monkeypatch.setattr(cad,'find_lowest_full_res_threshold',lambda _g,t,_s,_m:t)
    payload=cad.compress_full_mask(np.ones(gray.shape,bool))
    monkeypatch.setattr(cad,'refine_threshold',lambda _g,t,_s,_m:SimpleNamespace(threshold=t,cleaned_component_mask=payload))
    assert cad.find_auto_threshold(gray,_state())==12
    assert seen['shape']==(13,13)


def test_work_failure_reports_actual_support_kernel_geometry():
    gray=np.zeros((21,21),np.uint8); gray[9:12,9:12]=30
    with pytest.raises(cad.ThresholdResolutionError,match='7x7-supported'):
        cad.find_work_res_solar_component(gray,20,cad.generate_kernel((7,7),round_kernel=False))


def test_threshold_source_documents_coarse_and_refinement_contract():
    assert 'def extract_component(' in SOURCE
    assert SOURCE.count('cv2.floodFill(')==1
    assert 'def find_guard_boundary(' in SOURCE
    assert 'MAX_T_REFINEMENT_STEPS = 10' in SOURCE
    assert 'quality plateau' not in SOURCE.lower()
    assert 'ThresholdTopology' not in SOURCE
