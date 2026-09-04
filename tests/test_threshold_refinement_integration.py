import cv2
import numpy as np
import circle_arc_detector as cad


def _state(): return {'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}

def synthetic_bridge_with_separated_appendage():
    gray=np.zeros((220,320),np.uint8); cv2.circle(gray,(180,110),46,30,-1); gray[106:115,0:180]=8; gray[106:115,225:285]=9
    for x in range(225,280,10): gray[90:130,x:x+3]=9
    cv2.circle(gray,(190,105),12,70,-1); return gray

def test_find_auto_threshold_refines_above_lowest_separated_threshold():
    gray=synthetic_bridge_with_separated_appendage(); state=_state(); threshold=cad.find_auto_threshold(gray,state); result=state['auto_threshold_result']
    assert threshold==result.threshold==9 and result.resolved and result.cleaned_component_mask is not None
    mask=cad.decompress_full_mask(result.cleaned_component_mask,gray.shape); x,y=result.full_res_seed_point; assert mask[y,x]

def test_find_auto_threshold_keeps_separation_t_when_no_quality_improvement_wins():
    gray=np.zeros((220,320),np.uint8); cv2.circle(gray,(180,110),46,30,-1); gray[106:115,0:180]=8; cv2.circle(gray,(190,105),12,70,-1)
    state=_state(); threshold=cad.find_auto_threshold(gray,state); assert threshold==state['auto_threshold_result'].threshold==8

def test_common_component_extractor_is_the_only_flood_fill_implementation():
    from pathlib import Path
    source=Path(cad.__file__).read_text(); assert source.count('cv2.floodFill(')==1
    assert source.count('def extract_component(')==1 and source.count('def find_guard_boundary(')==1
