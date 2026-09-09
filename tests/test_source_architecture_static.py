from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_removed_obsolete_autot_and_gui_scaffolding():
    for name in (
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
        "compress_array",
        "decompress_array",
        "decompress_master_bgra16",
        "_ordered_external_component_contour",
        "master_bgra16_to_gray8",
        "master_bgra16_to_display_bgra8",
        "normalize_master_bgra16",
        "master_image_shape",
        "refine_solar_component_mask",
        "ROI_DILATION_FRACTION",
        "\nGUARD_DILATION_FRACTION =",
        "SOLAR_COMPONENT_KERNEL",
        "def opaque_bgra(",
        "def nearest_positive_odd(",
        "def build_parser(",
        "def validate_args(",
        "def close(",
    ):
        assert name not in TEXT


def test_expected_stage_names_and_shared_helpers_exist():
    for name in (
        "def morphological_cleanup(",
        "def compress_image(",
        "def decompress_image(",
        "def compress_contour(",
        "def decompress_contour(",
        "def find_work_res_separation_threshold(",
        "def find_full_res_separation_threshold(",
        "def find_auto_threshold(",
        "def find_separation_threshold(",
        "def refine_threshold(",
        "def find_external_contour(",
        "def calculate_work_res_shape(",
        "def derive_full_res_seed_and_guard(",
        "def extract_separated_seed_component(",
    ):
        assert name in TEXT


def test_edge_descriptor_uses_one_profile_helper_without_old_scaffolding():
    assert "def sample_grayscale_profiles(" in TEXT
    for removed in (
        "def _sample_edge_profiles(",
        "def _contour_normals(",
        "def _nearest_mask(",
        "def _linear_fit(",
    ):
        assert removed not in TEXT
    block = TEXT.split("def sample_grayscale_profiles(", 1)[1].split("\n\ndef ", 1)[0]
    assert "simplify_raster_contour(contour)" in block
    simplify_block = TEXT.split("def simplify_raster_contour(", 1)[1].split("\n\ndef ", 1)[0]
    assert "math.hypot(0.5, 0.5)" in simplify_block
    assert "cv2.approxPolyDP" in simplify_block
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


def test_gui_auto_select_button_delegates_parent_then_setting_application():
    block = TEXT.split("    def threshold_auto_button_clicked(self):", 1)[1].split(
        "\n    def radius_auto_button_clicked", 1
    )[0]
    assert "find_auto_threshold(" in block
    assert "find_separation_threshold(" not in block
    assert "refine_threshold(" not in block
    assert 'self.apply_changed_setting("threshold", selected_threshold)' in block
    assert "resolve_threshold(" not in block


def test_gui_button_callbacks_name_the_widget_boundary():
    for name in (
        "load_images_button_clicked",
        "save_centered_button_clicked",
        "previous_button_clicked",
        "next_button_clicked",
        "threshold_auto_button_clicked",
        "radius_auto_button_clicked",
        "preview_button_clicked",
        "full_button_clicked",
    ):
        assert f"def {name}(" in TEXT


def test_stageb_converts_full_resolution_gray_to_float32_once_before_candidate_loop():
    refine_block = TEXT.split("def refine_threshold(", 1)[1].split("\n\ndef ", 1)[0]
    profile_block = TEXT.split("def sample_grayscale_profiles(", 1)[1].split("\n\ndef ", 1)[0]
    assert refine_block.count("full_res_gray.astype(np.float32)") == 1
    assert "full_res_gray.astype(np.float32" not in profile_block


def test_bool_mask_conversion_is_centralized_without_np_where_integer_intermediates():
    assert "def bool_mask_to_uint8(" in TEXT
    assert "mask_u8 = mask.astype(np.uint8)" in TEXT
    assert "mask_u8 *= 255" in TEXT
    assert "np.where(source != 0, 255, 0)" not in TEXT
    assert "np.where(binary_mask != 0, 255, 0)" not in TEXT
    assert "np.where(component != 0, 255, 0)" not in TEXT


def test_canvas_renderer_uses_rendered_content_and_raw_png_bytes_without_base64():
    renderer = TEXT.split("    def render_canvas_content(self, canvas, content):", 1)[1].split(
        "\n\ndef main():", 1
    )[0]
    assert "_rendered_content" in renderer
    assert "_unscaled_render_raster" not in TEXT
    assert "import base64" not in TEXT
    assert "data=encoded_png.tobytes()" in renderer
    assert 'anchor="nw"' in renderer
