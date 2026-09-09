import cv2
import numpy as np
import pytest
import circle_arc_detector as cad


def test_seed_helper_returns_none_without_requested_support():
    gray=np.zeros((15,15),np.uint8); component=np.zeros_like(gray,bool); component[7,2:13]=True; gray[7,9]=220
    assert cad.brightest_supported_component_point(gray,component,cad.generate_kernel((5,5))) is None


def test_work_seed_kernel_is_fixed_5x5_square_in_stage_a_source():
    import inspect
    source=inspect.getsource(cad.find_separation_threshold)
    assert 'generate_kernel((5, 5), round_kernel=False)' in source


def test_full_seed_support_maps_from_realized_work_scale(monkeypatch):
    gray=np.zeros((6016,4000),np.uint8); work=np.zeros(cad.calculate_work_res_shape(gray.shape),bool); work[300:700,300:700]=True
    seen=[]
    monkeypatch.setattr(cad,'brightest_supported_component_point',lambda g,c,k,**kw: seen.append((k.shape,kw.get('use_knn_depth'))) or (1000,1000))
    monkeypatch.setattr(cad,'dilate_component_mask',lambda c,m: np.ones(gray.shape,bool))
    cad.derive_full_res_seed_and_guard(gray,work,cad.generate_kernel((5,5)))
    assert seen == [((25,25),True)]


def test_full_seed_support_stays_5_without_downscale(monkeypatch):
    gray=np.zeros((301,401),np.uint8); work=np.zeros_like(gray,bool); work[50:250,80:320]=True
    seen=[]
    monkeypatch.setattr(cad,'brightest_supported_component_point',lambda g,c,k,**kw: seen.append((k.shape,kw.get('use_knn_depth'))) or (200,150))
    monkeypatch.setattr(cad,'dilate_component_mask',lambda c,m: np.ones(gray.shape,bool))
    cad.derive_full_res_seed_and_guard(gray,work,cad.generate_kernel((5,5)))
    assert seen == [((5,5),True)]


def test_work_search_continues_below_unsupported_candidate(monkeypatch):
    gray=np.zeros((41,41),np.uint8); gray[10:31,10:31]=30
    calls=[]
    real=cad.brightest_supported_component_point
    def supported(g,c,k):
        calls.append(int(g[c].max()) if np.any(c) else -1)
        if len(calls)==1: return None
        return real(g,c,k)
    monkeypatch.setattr(cad,'brightest_supported_component_point',supported)
    seed,t,component=cad.find_work_res_separation_threshold(gray,30,cad.generate_kernel((5,5)))
    assert len(calls)>=2 and seed is not None and t < 30 and np.any(component)


def test_work_search_errors_if_nothing_supported_through_zero():
    gray=np.zeros((21,21),np.uint8); gray[10,2:19]=200
    with pytest.raises(cad.ThresholdResolutionError,match='5x5-supported'):
        cad.find_work_res_separation_threshold(gray,200,cad.generate_kernel((5,5)))


def test_failed_auto_is_stored_and_parent_returns_none_not_histogram_fallback():
    gray=np.full((80,120),100,np.uint8); state={'auto_threshold_result':None,'solar_data':None}
    selected=cad.find_auto_threshold(gray,state); result=state['auto_threshold_result']
    assert selected is None and result.failure_reason is not None
    assert result.histogram_start_threshold is not None and result.full_res_refined_threshold is None
