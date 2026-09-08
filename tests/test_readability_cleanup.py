import ast, math
from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad
ROOT=Path(__file__).parents[1]; SOURCE=ROOT/'circle_arc_detector.py'

def _legacy(component,margin):
    outside=np.where(component,0,255).astype(np.uint8); return cv2.distanceTransform(outside,cv2.DIST_L2,5)<=margin

def test_mapped_seed_support_keeps_lower_odd_tie_rule_inline():
    source=SOURCE.read_text()
    assert 'def nearest_positive_odd(' not in source
    assert '2 * math.ceil(mapped_kernel_size / 2) - 1' in source

def test_uint16_binary_resize_preserves_exact_values_when_declared_mask():
    mask=np.zeros((5,7),np.uint16); mask[1:4,2:6]=65535; r=cad.resize_img(mask,(13,17),mask=True)
    assert r.dtype==np.uint16 and set(np.unique(r))=={0,65535}

def test_full_frame_distance_dilation_matches_distance_semantics():
    c=np.zeros((91,123),bool); c[31:58,47:76]=True; c[42:49,76:83]=True
    for m in (0.,1.,6.5,17.25): assert np.array_equal(cad.dilate_component_mask(c,m),_legacy(c,m))

def test_refinement_has_no_silent_valueerror_fallback(monkeypatch):
    gray=np.zeros((61,61),np.uint8); cv2.circle(gray,(30,30),18,180,-1); gray[30,30]=240
    result=cad.AutoThresholdResult(); state={'auto_threshold_result':result}; cad.find_separation_threshold(gray,state)
    monkeypatch.setattr(cad,'measure_hole_quality',lambda *_: (_ for _ in ()).throw(ValueError('descriptor invariant failed')))
    with pytest.raises(ValueError,match='descriptor invariant failed'): cad.refine_threshold(gray,state)

def test_production_contains_no_lambda_obsolete_candidate_or_crop_constants():
    source=SOURCE.read_text(); tree=ast.parse(source); assert not any(isinstance(n,ast.Lambda) for n in ast.walk(tree))
    for name in ('WORK_MAX_DIM','TRACKING_SEED_KERNEL_SIZE','CLEANUP_KERNEL_SIZES','REFINEMENT_ITERATIONS','CLEANUP_CANDIDATE_ORDER','TOPOLOGY_OPTIMIZATION_STEPS'): assert name not in source
    assert 'component_crop' not in source

def test_refinement_kernels_use_shared_euclidean_generator():
    assert [int(k.sum()) for k in cad.SOLAR_CLEANUP_KERNELS]==[5,13,29]
    assert np.array_equal(cad.SOLAR_CLEANUP_KERNELS[-1],cad.SEPARATION_KERNEL)

def test_current_source_has_single_progressive_p357_cleanup_path():
    source=SOURCE.read_text(); assert 'def extract_separated_seed_component(' in source
    assert 'for kernel in SOLAR_CLEANUP_KERNELS:' in source
