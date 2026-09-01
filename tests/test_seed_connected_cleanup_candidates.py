import cv2
import numpy as np
import pytest
import circle_arc_detector as cad


def test_cleanup_retains_only_component_containing_authoritative_seed():
    raw=np.zeros((101,151),bool)
    cv2.circle(raw.view(np.uint8),(45,50),20,1,-1)
    cv2.circle(raw.view(np.uint8),(115,50),14,1,-1)
    raw[49:52,65:102]=True
    out=cad.seed_connected_cleanup_candidates(raw,(45,50))
    assert out['raw'][50,115]
    severed=[name for name in out if not out[name][50,115]]
    assert severed
    for name in severed:
        assert out[name][50,45]


def test_candidate_that_removes_seed_is_rejected(monkeypatch):
    raw=np.zeros((41,41),bool); raw[10:31,10:31]=True; seed=(20,20)
    original=cad.cleanup_morphology_candidates
    def fake(component):
        d=original(component)
        d['D7']=d['D7'].copy(); d['D7'][seed[1],seed[0]]=False
        return d
    monkeypatch.setattr(cad,'cleanup_morphology_candidates',fake)
    out=cad.seed_connected_cleanup_candidates(raw,seed)
    assert 'raw' in out
    assert 'D7' not in out


def test_raw_component_without_seed_is_an_invariant_error():
    raw=np.zeros((21,21),bool); raw[5:10,5:10]=True
    with pytest.raises(cad.ThresholdResolutionError,match='outside the raw'):
        cad.seed_connected_cleanup_candidates(raw,(15,15))
