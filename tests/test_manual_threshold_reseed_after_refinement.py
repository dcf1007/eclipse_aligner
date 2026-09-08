import cv2
import numpy as np
import circle_arc_detector as cad

def _state(t): return {'settings':cad.ImageSettings(threshold=t),'auto_threshold_result':None,'solar_data':None}

def test_current_t_authoritative_seed_is_chosen_before_cleanup_and_survives():
    gray=np.zeros((81,81),np.uint8); cv2.circle(gray,(40,40),18,180,-1); gray[40,40]=240; gray[40,58:66]=255
    state=_state(120); refined=cad.resolve_threshold(gray,120,state); solar=state['solar_data']
    assert solar.seed_point==(40,40) and refined[40,40] and not refined[40,65]

def test_manual_t_with_existing_full_identity_does_not_rederive_seed(monkeypatch):
    gray=np.zeros((101,111),np.uint8); cv2.circle(gray,(55,50),22,220,-1); gray[50,55]=250
    state=_state(130); auto=cad.find_auto_threshold(gray,state); manual=min(255,auto+1)
    monkeypatch.setattr(cad,'derive_full_res_seed_and_guard',lambda *_: (_ for _ in ()).throw(AssertionError('seed rederived')))
    resolved=cad.resolve_threshold(gray,manual,state)
    assert np.any(resolved) and state['solar_data'].seed_point==state['auto_threshold_result'].full_res_seed_point

def test_same_t_reuses_persisted_seed_and_component(monkeypatch):
    gray=np.zeros((101,111),np.uint8); cv2.circle(gray,(55,50),22,220,-1); gray[50,55]=250
    state=_state(130); first=cad.resolve_threshold(gray,130,state); solar=state['solar_data']
    monkeypatch.setattr(cad,'find_auto_threshold',lambda *_: (_ for _ in ()).throw(AssertionError('same T reran')))
    second=cad.resolve_threshold(gray,130,state)
    assert state['solar_data'] is solar and np.array_equal(first,second)
