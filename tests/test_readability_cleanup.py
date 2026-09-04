import ast
import math
from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

ROOT=Path(__file__).parents[1]; SOURCE=ROOT/'circle_arc_detector.py'; RETAINED_PATCH=ROOT/'snippets'/'apply_streamlined_autot.py'

def _legacy_cropped_dilation(component,margin):
    ys,xs=np.nonzero(component); h,w=component.shape; padding=math.ceil(margin)
    x0=max(0,int(xs.min())-padding-2); x1=min(w,int(xs.max())+padding+3); y0=max(0,int(ys.min())-padding-2); y1=min(h,int(ys.max())+padding+3)
    crop=component[y0:y1,x0:x1]; outside=np.where(crop,0,255).astype(np.uint8); distance=cv2.distanceTransform(outside,cv2.DIST_L2,5)
    result=np.zeros(component.shape,bool); result[y0:y1,x0:x1]=distance<=margin; return result

def test_nearest_positive_odd_preserves_lower_tie_rule():
    assert cad.nearest_positive_odd(23.9)==23; assert cad.nearest_positive_odd(24.0)==23; assert cad.nearest_positive_odd(24.1)==25
    with pytest.raises(ValueError): cad.nearest_positive_odd(0)

def test_uint16_binary_resize_preserves_exact_values_when_declared_mask():
    mask=np.zeros((5,7),np.uint16); mask[1:4,2:6]=65535; resized=cad.resize_img(mask,(17,13),mask=True)
    assert resized.dtype==np.uint16 and set(np.unique(resized))=={0,65535}

def test_full_frame_distance_dilation_matches_previous_cropped_semantics():
    component=np.zeros((91,123),bool); component[31:58,47:76]=True; component[42:49,76:83]=True
    for margin in (0.0,1.0,6.5,17.25): assert np.array_equal(cad.dilate_component_mask(component,margin),_legacy_cropped_dilation(component,margin))

def test_refinement_has_no_silent_fallback(monkeypatch):
    gray=np.full((31,31),100,np.uint8); guard=np.ones(gray.shape,bool)
    def fail(*_a,**_k): raise ValueError('descriptor invariant failed')
    monkeypatch.setattr(cad,'clean_solar_component',fail)
    with pytest.raises(ValueError,match='descriptor invariant failed'): cad.refine_threshold(gray,3,(15,15),guard,max_steps=0)

def test_production_contains_no_lambda_obsolete_candidate_or_crop_constants():
    source=SOURCE.read_text(); tree=ast.parse(source); assert not any(isinstance(n,ast.Lambda) for n in ast.walk(tree))
    for name in ('WORK_MAX_DIM','TRACKING_SEED_KERNEL_SIZE','CLEANUP_KERNEL_SIZES','REFINEMENT_ITERATIONS','CLEANUP_CANDIDATE_ORDER','TOPOLOGY_OPTIMIZATION_STEPS'): assert name not in source
    assert 'component_crop' not in source

def test_refinement_kernels_use_shared_euclidean_generator():
    assert [int(k.sum()) for k in cad.SOLAR_CLEANUP_KERNELS]==[5,13,29]
    assert np.array_equal(cad.SOLAR_CLEANUP_KERNELS[-1],cad.SEPARATION_KERNEL)

def test_retained_streamlined_patch_delegates_current_cleanup():
    text=RETAINED_PATCH.read_text(); assert 'apply_readability_cleanup.py' in text and 'apply_production_cleanup' in text
