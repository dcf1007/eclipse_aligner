from pathlib import Path
import cv2
import numpy as np
import circle_arc_detector as cad
SOURCE=Path(cad.__file__).read_text()

def test_resize_mask_flag_owns_nearest_neighbor_selection(monkeypatch):
    seen=[]
    monkeypatch.setattr(cad.cv2,'resize',lambda src,size,interpolation: seen.append(interpolation) or np.zeros((3,1),src.dtype))
    cad.resize_img(np.zeros((5,5),np.uint8),(3,1),mask=True); assert seen==[cv2.INTER_NEAREST_EXACT]

def test_coarse_d7_removes_thin_background_bridge_and_returns_component():
    gray=np.zeros((101,141),np.uint8); cv2.circle(gray,(75,50),22,30,-1); gray[49:52,:76]=20
    guard=np.zeros_like(gray,bool); guard[10:91,30:121]=True
    t,component=cad.find_full_res_separation_threshold(gray,20,(75,50),guard)
    assert t<=20 and component[50,75] and not np.any(component[:,0])

def test_stage_b_consumes_same_stage_a_seed_and_guard(monkeypatch):
    gray=np.zeros((81,81),np.uint8); cv2.circle(gray,(40,40),20,180,-1); gray[40,40]=240; state={'auto_threshold_result':cad.AutoThresholdResult()}
    cad.find_separation_threshold(gray,state); result=state['auto_threshold_result']; seed=result.full_res_seed_point; guard=cad.decompress_image(result.full_res_separation_guard_mask); seen=[]
    real=cad.extract_separated_seed_component
    def record(mask,s,g,b): seen.append((s,np.array_equal(g,guard))); return real(mask,s,g,b)
    monkeypatch.setattr(cad,'extract_separated_seed_component',record); cad.refine_threshold(gray,state)
    assert seen and all(s==seed and same for s,same in seen)

def test_uint8_bgr_load_path_expands_exactly_to_uint16_master():
    block=SOURCE.split('def load_image_at(self, index: int):',1)[1].split('def previous_button_clicked',1)[0]
    assert 'master_image.astype(np.uint16) * 257' in block and 'cv2.COLOR_BGR2BGRA' in block

def test_uint16_bgra_master_is_retained_by_shared_codec():
    master=np.zeros((5,7,4),np.uint16); master[...,0]=1234; master[...,3]=65535
    assert np.array_equal(cad.decompress_image(cad.compress_image(master)),master)

def test_display_mapping_is_fixed_full_range_and_preserves_alpha_scale_in_load_source():
    block=SOURCE.split('def load_image_at(self, index: int):',1)[1].split('def previous_button_clicked',1)[0]
    assert '(master_image.astype(np.uint32) + 128) // 257' in block

def test_production_load_path_uses_unchanged_master_and_numpy_shape_convention():
    block=SOURCE.split('def load_image_at(self, index: int):',1)[1].split('def previous_button_clicked',1)[0]
    assert 'cv2.IMREAD_UNCHANGED' in block and 'compress_image(master_image)' in block
    assert 'master_image_shape' not in block
