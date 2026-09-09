import inspect
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

def _guard(shape=(41,51)):
    g=np.zeros(shape,bool); g[4:-4,4:-4]=True; return g

def test_full_res_seed_is_checked_after_d7_cleanup():
    source=inspect.getsource(cad.find_full_res_separation_threshold)
    assert source.index('morphological_cleanup(') < source.index('if binary[seed_y, seed_x] == 0:')

def test_full_res_seed_must_survive_d7_cleanup():
    gray=np.zeros((41,51),np.uint8); gray[20,25]=200
    with pytest.raises(cad.ThresholdResolutionError,match='does not survive D7 cleanup'):
        cad.find_full_res_separation_threshold(gray,100,(25,20),_guard(gray.shape))

def test_find_auto_threshold_requires_authoritative_uint8_gray():
    with pytest.raises(ValueError,match='authoritative 2D uint8 grayscale'):
        cad.find_auto_threshold(np.zeros((10,10),np.uint16),{'auto_threshold_result':None})

def test_support_mapping_uses_actual_work_kernel(monkeypatch):
    gray=np.zeros((2400,1600),np.uint8); work=np.zeros((1200,800),bool); work[300:900,200:600]=True; seen=[]
    monkeypatch.setattr(cad,'brightest_supported_component_point',lambda g,c,k,**kw: seen.append((k.shape,kw.get('use_knn_depth'))) or (800,1200))
    monkeypatch.setattr(cad,'dilate_component_mask',lambda c,m: np.ones(gray.shape,bool))
    cad.derive_full_res_seed_and_guard(gray,work,cad.generate_kernel((7,7)))
    assert seen==[((13,13),True)]

def test_work_failure_reports_actual_support_kernel_geometry():
    gray=np.zeros((21,21),np.uint8); gray[10,2:19]=200
    with pytest.raises(cad.ThresholdResolutionError,match='7x7-supported'):
        cad.find_work_res_separation_threshold(gray,200,cad.generate_kernel((7,7)))

def test_threshold_source_documents_coarse_and_refinement_contract():
    source=inspect.getsource(cad.find_full_res_separation_threshold)+inspect.getsource(cad.refine_threshold)
    assert 'D7' in source and 'MAX_T_REFINEMENT_STEPS' in source
