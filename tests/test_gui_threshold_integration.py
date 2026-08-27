"""Regression checks for the final single-file threshold/UI integration.

These preserve the assertions exercised by the successful integration CI run,
updated only to reflect that the tested finder is now in circle_arc_detector.py.
"""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_tested_threshold_algorithm_is_integrated_inline():
    assert "from threshold_finder import" not in TEXT
    assert "def rightmost_histogram_peak(" in TEXT
    assert "def auto_threshold_from_gray(" in TEXT
    assert "WORK_MAX_DIM = 1200" in TEXT
    assert "ROI_DILATION_FRACTION = 0.065" in TEXT
    assert "GUARD_DILATION_FRACTION = 0.195" in TEXT


def test_first_image_load_runs_auto_and_generates_preview():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_image", 1
    )[0]
    assert "self.gray_image = to_gray(image)" in block
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.render_threshold_preview()" in block


def test_auto_select_recomputes_threshold_without_regenerating_preview():
    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius", 1
    )[0]
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.threshold.set(selected_threshold)" in block
    assert "self.render_threshold_preview()" not in block
    assert "Preview not regenerated." in block


def test_manual_threshold_is_stored_per_image():
    assert "self.image_thresholds: dict[str, int] = {}" in TEXT
    pending = TEXT.split("def pending(self, *_args):", 1)[1].split(
        "def center_target_changed", 1
    )[0]
    assert "self.image_thresholds[self.current_path] = int(self.threshold.get())" in pending


def test_explicit_preview_actions_render_threshold_output():
    refresh = TEXT.split("def refresh_preview(self):", 1)[1].split(
        "def apply_full_resolution", 1
    )[0]
    apply = TEXT.split("def apply_full_resolution(self):", 1)[1].split(
        "def schedule_redraw", 1
    )[0]
    assert "self.render_threshold_preview()" in refresh
    assert "self.render_threshold_preview()" in apply


def test_preview_uses_exact_threshold_semantics():
    renderer = TEXT.split("def render_threshold_preview(self):", 1)[1].split(
        "def clear_threshold_preview", 1
    )[0]
    assert "cv2.CMP_GT" in renderer
    assert "dark = gray <= T, light = gray > T" in renderer
