import inspect
import cv2
import numpy as np
import circle_arc_detector as cad

def _gray():
    g=np.zeros((91,101),np.uint8); cv2.circle(g,(50,45),20,190,-1); g[45,50]=250; return g

def test_progressive_refinement_is_authoritative_production_helper():
    source=inspect.getsource(cad.extract_separated_seed_component)
    assert 'for kernel in SOLAR_CLEANUP_KERNELS:' in source
    assert 'return component, find_external_contour(component)' in source

def test_exact_t_resolver_uses_progressive_component_helper():
    source=inspect.getsource(cad.resolve_threshold)
    assert 'extract_separated_seed_component(' in source
    assert 'refine_solar_component_mask' not in source

def test_resolver_returns_and_stores_same_component():
    gray=_gray(); state={'settings':cad.ImageSettings(threshold=100),'auto_threshold_result':None,'solar_data':None}
    resolved=cad.resolve_threshold(gray,100,state)
    assert np.array_equal(resolved,cad.decompress_image(state['solar_data'].component_mask))

def test_guard_and_contour_derive_from_same_returned_component():
    gray=_gray(); state={'settings':cad.ImageSettings(threshold=100),'auto_threshold_result':None,'solar_data':None}
    resolved=cad.resolve_threshold(gray,100,state); solar=state['solar_data']; guard=cad.decompress_image(solar.guard_mask)
    assert np.all(resolved<=guard)
    assert np.array_equal(cad.decompress_contour(solar.component_contour),cad.find_external_contour(resolved))
