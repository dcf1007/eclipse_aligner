from pathlib import Path
import ast
SOURCE=Path(__file__).resolve().parents[1]/'circle_arc_detector.py'; TEXT=SOURCE.read_text()

def test_corrected_source_parses(): ast.parse(TEXT)
def test_single_agreed_refinement_dataclass(): assert TEXT.count('class ThresholdMeasurement:')==1 and 'ThresholdRefinementResult' not in TEXT
def test_stage_b_contract_consumes_stage_a_result_state():
    block=TEXT.split('def refine_threshold(',1)[1].split('\n\ndef ',1)[0]
    assert 'auto_threshold_result = image_state["auto_threshold_result"]' in block and 'full_res_seed_point' in block and 'full_res_guard_mask' in block
    assert 'find_separation_threshold(' not in block

def test_shared_component_extraction_and_separation_naming():
    assert TEXT.count('cv2.floodFill(')==1 and 'def extract_component(' in TEXT and 'def find_guard_boundary(' in TEXT and 'def find_separation_threshold(' in TEXT
    assert 'find_lowest_full_res_threshold' not in TEXT

def test_edge_control_uses_current_profile_radius_without_old_scaffolding():
    assert 'EDGE_PROFILE_RADIUS_PX = 25' in TEXT
    for old in ('EDGE_RADIUS','EDGE_RECOVERY_FRACTION','EDGE_TARGET_PROFILE_COUNT','EDGE_TANGENT_SPAN'): assert old not in TEXT
    assert 'max(0.02' not in TEXT

def test_auto_result_contract_is_progressive_and_parent_persists_expected_failures():
    tree=ast.parse(TEXT); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AutoThresholdResult')
    fields={n.target.id for n in cls.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
    assert {'histogram_start_threshold','work_res_seed_point','full_res_seed_point','full_res_refined_threshold','failure_reason'}<=fields
    assert 'resolved' not in fields
    parent=TEXT.split('def find_auto_threshold(',1)[1].split('def find_separation_threshold(',1)[0]
    assert 'find_separation_threshold(' in parent and 'refine_threshold(' in parent and 'AutoThresholdResult()' in parent

def test_solardata_resolver_path_exists_after_autot(): assert TEXT.index('def find_auto_threshold(')<TEXT.index('class SolarData:')<TEXT.index('def resolve_threshold(')
