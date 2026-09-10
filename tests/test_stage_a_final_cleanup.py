from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad
SOURCE=Path(cad.__file__).read_text()

def test_resize_same_shape_returns_existing_raster_without_opencv(monkeypatch):
    image=np.arange(7*3,dtype=np.uint8).reshape(7,3); monkeypatch.setattr(cad.cv2,'resize',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('opencv called')))
    r=cad.resize_img(image,(7,3)); assert r is image

def test_resize_uses_area_when_height_shrinks(monkeypatch):
    image=np.zeros((7,3),np.uint8); seen={}
    def record(src,size,interpolation): seen.update(size=size,interpolation=interpolation); return np.zeros((3,3),np.uint8)
    monkeypatch.setattr(cad.cv2,'resize',record); cad.resize_img(image,(3,3)); assert seen=={'size':(3,3),'interpolation':cv2.INTER_AREA}

def test_supported_seed_requires_full_kernel_inside_raster_boundary():
    gray=np.zeros((9,9),np.uint8); component=np.ones_like(gray,bool); gray[0,0]=255; gray[4,4]=200
    assert cad.brightest_supported_component_point(gray,component,cad.generate_kernel((5,5)))==(4,4)

def test_stage_b_valueerror_propagates_without_reclassifying_as_coarse_failure(monkeypatch):
    gray=np.zeros((61,61),np.uint8); cv2.circle(gray,(30,30),18,180,-1); gray[30,30]=240; state={'auto_threshold_result':cad.AutoThresholdResult()}
    cad.find_separation_threshold(gray,state); monkeypatch.setattr(cad,'measure_hole_quality',lambda *_: (_ for _ in ()).throw(ValueError('bad descriptor')))
    with pytest.raises(ValueError,match='bad descriptor'): cad.refine_threshold(gray,state)
    assert state['auto_threshold_result'].failure_reason is None

def test_source_orders_complete_autot_before_solardata_resolver():
    assert SOURCE.index('def find_auto_threshold(') < SOURCE.index('def resolve_threshold(')
    resolver=SOURCE.split('def resolve_threshold(',1)[1].split('class DetectorApp',1)[0]
    assert 'find_auto_threshold(' in resolver
