from pathlib import Path
import ast

SOURCE = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_corrected_source_parses():
    ast.parse(TEXT)


def test_single_agreed_refinement_dataclass():
    assert TEXT.count("class ThresholdMeasurement:") == 1
    assert "ThresholdRefinementResult" not in TEXT


def test_resolution_explicit_refinement_contract():
    block = TEXT.split("def refine_threshold(", 1)[1].split("\n\ndef ", 1)[0]
    assert "full_res_gray" in block
    assert "full_res_seed_point" in block
    assert "full_res_guard_mask" in block
    assert "gray = np.asarray(full_res_gray)" not in block
    assert "guard = np.asarray(full_res_guard_mask" not in block
    assert "max_steps" not in block


def test_shared_component_extraction_and_separation_naming():
    assert TEXT.count("cv2.floodFill(") == 1
    assert "def extract_component(" in TEXT
    assert "def find_guard_boundary(" in TEXT
    assert "def find_separation_threshold(" in TEXT
    assert "find_lowest_full_res_threshold" not in TEXT


def test_edge_controls_have_explicit_names_and_dead_recovery_floor_is_removed():
    for name in (
        "EDGE_PROFILE_RADIUS_PX",
        "EDGE_SLOPE_RECOVERY_FRACTION",
        "EDGE_SLOPE_RECOVERY_PERSISTENCE_PX",
        "EDGE_PROFILE_MIN_SAMPLE_STRIDE",
        "EDGE_PROFILE_MAX_SAMPLE_COUNT",
        "EDGE_NORMAL_TANGENT_HALF_SPAN",
    ):
        assert name in TEXT
    for old_name in (
        "EDGE_RADIUS",
        "EDGE_RECOVERY_FRACTION",
        "EDGE_RECOVERY_PERSISTENCE",
        "EDGE_MIN_SAMPLE_STRIDE",
        "EDGE_TARGET_PROFILE_COUNT",
        "EDGE_TANGENT_SPAN",
    ):
        assert old_name not in TEXT
    assert "max(0.02" not in TEXT
    assert "TODO: Revisit and benchmark this edge-confidence calculation" in TEXT


def test_auto_result_contract_defaults_unresolved_and_refinement_failure_is_persisted():
    tree = ast.parse(TEXT)
    result_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AutoThresholdResult")
    fields = {
        node.target.id: node.value
        for node in result_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert isinstance(fields["resolved"], ast.Constant) and fields["resolved"].value is False

    auto_block = TEXT.split("def find_auto_threshold(", 1)[1].split(
        "# ---------------------------------------------------------------------------\n# Final-T full-resolution solar resolution",
        1,
    )[0]
    assert 'resolution_step = "coarse separation"' in auto_block
    assert 'resolution_step = "fine refinement"' in auto_block
    assert 'reason=f"{resolution_step}: {exc}"' in auto_block
    assert "final_T, cleaned_component_mask = refine_threshold(" in auto_block
    assert auto_block.index("final_T, cleaned_component_mask = refine_threshold(") < auto_block.index("except ThresholdResolutionError as exc:")


def test_deferred_solardata_path_still_exists_unmodified_in_scope():
    assert "def resolve_threshold(" in TEXT
    assert "class SolarData:" in TEXT
