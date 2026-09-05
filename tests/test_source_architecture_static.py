from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_removed_obsolete_autot_and_gui_scaffolding():
    for name in (
        "find_auto_threshold",
        "separation_result_callback",
        "_display_auto_separation_result",
        "display_raw_threshold",
        "_schedule_canvas_redraw",
        "_redraw_cached_canvases",
        "CANVAS_REDRAW_DELAY_MS",
    ):
        assert name not in TEXT


def test_removed_cumbersome_helpers_and_old_codecs():
    for name in (
        "clean_solar_component",
        "_lattice_boundary_points",
        "compress_full_mask",
        "decompress_full_mask",
        "compress_master_bgra16",
        "decompress_master_bgra16",
        "master_bgra16_to_gray8",
        "master_bgra16_to_display_bgra8",
        "normalize_master_bgra16",
        "master_image_shape",
    ):
        assert name not in TEXT


def test_expected_stage_names_and_shared_helpers_exist():
    for name in (
        "def morphological_cleanup(",
        "def compress_array(",
        "def decompress_array(",
        "def find_work_res_separation_threshold(",
        "def find_full_res_separation_threshold(",
        "def find_separation_threshold(",
        "def refine_threshold(",
    ):
        assert name in TEXT


def test_edge_descriptor_uses_one_profile_helper_without_old_scaffolding():
    assert "def _sample_grayscale_profiles(" in TEXT
    for removed in (
        "def _sample_edge_profiles(",
        "def _contour_normals(",
        "def _nearest_mask(",
        "def _linear_fit(",
    ):
        assert removed not in TEXT
    block = TEXT.split("def _sample_grayscale_profiles(", 1)[1].split("\n\ndef ", 1)[0]
    assert "math.hypot(0.5, 0.5)" in block
    assert "cv2.GaussianBlur" not in block


def test_stageb_descriptor_set_is_roughness_holes_area_and_edge_only():
    assert "def measure_hole_quality(" in TEXT
    assert "def measure_solidity(" not in TEXT
    assert "def measure_internal_dark_fraction(" not in TEXT
    score_block = TEXT.split("best_threshold: int | None = None", 1)[1].split(
        "if best_threshold is None:", 1
    )[0]
    assert "measurement.hole_quality" in score_block
    assert "0.5 * q_area" in score_block
    assert "edge_reliability * q_edge" in score_block
    assert "q_solidity" not in score_block
