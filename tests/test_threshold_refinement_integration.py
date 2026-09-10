import cv2
import numpy as np
import circle_arc_detector as cad

def _state(): return {'settings':cad.ImageSettings(),'auto_threshold_result':None,'solar_data':None}
def synthetic_bridge_with_separated_appendage():
    g=np.zeros((220,320),np.uint8); cv2.circle(g,(180,110),46,30,-1); g[106:115,0:180]=8; g[106:115,225:285]=9
    for x in range(225,280,10): g[90:130,x:x+3]=9
    cv2.circle(g,(190,105),12,70,-1); return g

def test_find_auto_threshold_refines_above_lowest_separated_threshold():
    g=synthetic_bridge_with_separated_appendage(); s=_state(); t=cad.find_auto_threshold(g,s); r=s['auto_threshold_result']
    assert r.full_res_separation_threshold==8 and t==r.full_res_refined_threshold==9 and r.threshold_refinement_complete
    mask=cad.decompress_image(r.full_res_refined_component_mask); x,y=r.full_res_seed_point; assert mask[y,x]

def test_find_auto_threshold_keeps_separation_t_when_no_quality_improvement_wins():
    g=np.zeros((220,320),np.uint8); cv2.circle(g,(180,110),46,30,-1); g[106:115,0:180]=8; cv2.circle(g,(190,105),12,70,-1)
    s=_state(); t=cad.find_auto_threshold(g,s); r=s['auto_threshold_result']; assert t==r.full_res_refined_threshold==r.full_res_separation_threshold==8

def test_common_component_extractor_is_only_flood_fill_implementation():
    from pathlib import Path
    source=Path(cad.__file__).read_text(); assert source.count('cv2.floodFill(')==1 and source.count('def extract_component(')==1 and source.count('def find_guard_boundary_indices(')==1 and 'def find_guard_boundary(' not in source
