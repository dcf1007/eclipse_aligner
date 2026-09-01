import numpy as np
import pytest
import circle_arc_detector as cad


def test_seed_helper_uses_explicit_kernel_and_has_no_fallback():
    gray=np.zeros((21,21),np.uint8); comp=np.zeros_like(gray); comp[10,3:18]=255; gray[10,10]=250
    with pytest.raises(cad.ThresholdResolutionError, match='5x5-supported'):
        cad.brightest_supported_component_point(gray, comp, np.ones((5,5),np.uint8))
    assert cad.brightest_supported_component_point(gray, comp, np.ones((1,1),np.uint8)) == (10,10)


def test_work_seed_kernel_is_fixed_5x5_square():
    assert cad.TRACKING_SEED_KERNEL.shape == (5,5)
    assert np.all(cad.TRACKING_SEED_KERNEL == 1)


def test_full_seed_support_maps_6016_to_25_square():
    k=cad.equivalent_full_resolution_seed_kernel((4000,6016),(798,1200))
    assert k.shape == (25,25)
    assert np.all(k == 1)


def test_full_seed_support_stays_5_if_no_downscale():
    assert cad.equivalent_full_resolution_seed_kernel((800,1200)).shape == (5,5)


def test_coarse_search_continues_below_unsupported_candidate(monkeypatch):
    gray=np.zeros((31,31),np.uint8)
    gray[11:20,11:20]=15
    gray[14:17,14:17]=30
    monkeypatch.setattr(cad,'find_rightmost_histogram_peak',lambda _g:(30,20))
    result=cad.coarse_threshold_search(gray)
    assert result.seed_threshold == 14


def test_coarse_search_errors_if_nothing_supported_through_zero(monkeypatch):
    gray=np.zeros((21,21),np.uint8); gray[9:12,9:12]=30
    monkeypatch.setattr(cad,'find_rightmost_histogram_peak',lambda _g:(30,20))
    with pytest.raises(cad.ThresholdResolutionError, match='through T=0'):
        cad.coarse_threshold_search(gray)


def test_unresolved_auto_is_stored_but_not_returned_as_histogram_fallback():
    gray=np.full((40,50),100,np.uint8)
    state={'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}
    with pytest.raises(cad.ThresholdResolutionError):
        cad.find_auto_threshold(gray,state)
    r=state['auto_threshold_result']
    assert not r.resolved
    assert r.threshold is None
    assert r.seed_threshold is None
