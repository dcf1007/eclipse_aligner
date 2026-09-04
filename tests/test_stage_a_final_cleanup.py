from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

SOURCE=Path(cad.__file__).read_text(encoding='utf-8')


def test_resize_same_size_is_an_exact_copy_without_opencv(monkeypatch):
    image=np.arange(42,dtype=np.uint8).reshape(6,7)
    monkeypatch.setattr(cad.cv2,'resize',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError()))
    resized=cad.resize_img(image,(7,6)); assert np.array_equal(resized,image) and resized is not image


def test_resize_uses_area_when_only_height_shrinks(monkeypatch):
    image=np.arange(42,dtype=np.uint8).reshape(6,7); seen={}
    def record(source,size,interpolation): seen.update(size=size,interpolation=interpolation); return np.zeros((size[1],size[0]),source.dtype)
    monkeypatch.setattr(cad.cv2,'resize',record); cad.resize_img(image,(7,3))
    assert seen=={'size':(7,3),'interpolation':cv2.INTER_AREA}


def test_supported_seed_requires_full_kernel_inside_raster_boundary():
    gray=np.zeros((5,5),np.uint8); gray[0,0]=255; gray[2,2]=200; comp=np.ones((5,5),bool)
    assert cad.brightest_supported_component_point(gray,comp,cad.generate_kernel((5,5)))==(2,2)


def test_refinement_error_propagates_without_reclassifying_as_coarse_failure(monkeypatch):
    gray=np.zeros((31,31),np.uint8); state={'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}
    monkeypatch.setattr(cad,'find_histogram_start_threshold',lambda _g:10)
    def work(w,_t,_k): c=np.zeros(w.shape,bool); c[8:23,8:23]=True; return 7,c
    monkeypatch.setattr(cad,'find_work_res_solar_component',work)
    monkeypatch.setattr(cad,'brightest_supported_component_point',lambda *_a:(15,15))
    monkeypatch.setattr(cad,'dilate_component_mask',lambda c,_m:np.ones(c.shape,bool))
    monkeypatch.setattr(cad,'find_lowest_full_res_threshold',lambda *_a:6)
    def fail(*_a,**_k): raise cad.ThresholdResolutionError('refinement sentinel')
    monkeypatch.setattr(cad,'refine_threshold',fail)
    with pytest.raises(cad.ThresholdResolutionError,match='refinement sentinel'):
        cad.find_auto_threshold(gray,state)
    assert state['auto_threshold_result'] is None


def test_source_orders_coarse_refinement_before_deferred_solardata():
    generic=SOURCE.index('# Generic image and kernel utilities')
    settings=SOURCE.index('# Per-image processing settings')
    threshold=SOURCE.index('# Automatic threshold: coarse separation and deterministic refinement')
    refine=SOURCE.index('def refine_threshold(')
    auto=SOURCE.index('def find_auto_threshold(')
    final_t=SOURCE.index('# Final-T full-resolution solar resolution and persistence')
    gui=SOURCE.index('class DetectorApp:')
    assert generic < settings < threshold < refine < auto < final_t < gui
    search=SOURCE[threshold:final_t]
    assert 'AUTO_T_GUARD_DILATION_FRACTION' in search
    assert 'MAX_T_REFINEMENT_STEPS' in search
    assert 'ThresholdMeasurement' in search
    assert 'ThresholdTopology' not in search
    assert 'CLEANUP_CANDIDATE_ORDER' not in search
    assert 'ROI_DILATION_FRACTION' not in search
