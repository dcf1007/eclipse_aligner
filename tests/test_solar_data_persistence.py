import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

def _gray():
    g=np.zeros((81,91),np.uint8); cv2.circle(g,(45,40),18,190,-1); g[40,45]=250; return g

def _state(t=100): return {'settings':cad.ImageSettings(threshold=t),'auto_threshold_result':None,'solar_data':None}

def test_auto_threshold_stage_has_no_solar_data_dependency():
    state={'auto_threshold_result':None}; selected=cad.find_auto_threshold(_gray(),state)
    assert selected is not None and 'solar_data' not in state

def test_shared_image_codec_round_trip_non_multiple_of_eight_bool_mask():
    mask=np.zeros((7,13),bool); mask[1:6,2:11]=True
    assert np.array_equal(cad.decompress_image(cad.compress_image(mask)),mask)

def test_image_decoder_rejects_corrupt_payload():
    with pytest.raises((ValueError,Exception)):
        cad.decompress_image(b'not-zlib')

def test_resolver_persists_returned_component_seed_guard_and_contour():
    gray=_gray(); state=_state(); resolved=cad.resolve_threshold(gray,100,state); solar=state['solar_data']
    assert np.array_equal(cad.decompress_image(solar.component_mask),resolved)
    guard=cad.decompress_image(solar.guard_mask); contour=cad.decompress_contour(solar.component_contour)
    x,y=solar.seed_point; assert resolved[y,x] and guard[y,x] and np.all(resolved<=guard)
    assert np.array_equal(contour,cad.find_external_contour(resolved))

def test_auto_winner_is_reused_without_selected_t_reprocessing(monkeypatch):
    gray=_gray(); state=_state(); selected=cad.find_auto_threshold(gray,state)
    monkeypatch.setattr(cad,'extract_separated_seed_component',lambda *_: (_ for _ in ()).throw(AssertionError('winner recomputed')))
    resolved=cad.resolve_threshold(gray,selected,state)
    assert np.array_equal(resolved,cad.decompress_image(state['auto_threshold_result'].full_res_refined_component_mask))

def test_current_t_without_enclosed_component_fails_without_partial_write():
    gray=np.zeros((41,51),np.uint8); gray[:,:8]=220
    state={'settings':cad.ImageSettings(threshold=100),'auto_threshold_result':cad.AutoThresholdResult(failure_reason='no auto'),'solar_data':None}
    with pytest.raises(cad.ThresholdResolutionError): cad.resolve_threshold(gray,100,state)
    assert state['solar_data'] is None

def test_same_t_solardata_is_reused_without_reestablishing_identity(monkeypatch):
    gray=_gray(); state=_state(); first=cad.resolve_threshold(gray,100,state); solar=state['solar_data']
    monkeypatch.setattr(cad,'find_auto_threshold',lambda *_: (_ for _ in ()).throw(AssertionError('identity rerun')))
    assert np.array_equal(cad.resolve_threshold(gray,100,state),first) and state['solar_data'] is solar

def test_old_split_builder_and_roi_guard_apis_are_removed():
    for name in ('refine_solar_component_mask','compress_full_mask','decompress_full_mask','ROI_DILATION_FRACTION','GUARD_DILATION_FRACTION'):
        assert not hasattr(cad,name)
    assert tuple(cad.SolarData.__dataclass_fields__)==('threshold','seed_point','component_mask','guard_mask','component_contour')
