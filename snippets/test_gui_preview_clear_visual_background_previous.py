# This file preserves the rejected visual-background-matching regression test from the interrupted GUI pass.
# The active alpha-channel regression lives in tests/test_gui_preview_clear.py.

"""Regression checks for blank, backgroundless previews and stale-preview clearing."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_preview_canvases_have_no_placeholder_text_or_dark_fill():
    assert "def placeholder(" not in TEXT
    assert "#202020" not in TEXT
    assert 'preview_background = frame.cget("background")' in TEXT
    assert TEXT.count("highlightthickness=0") >= 2
    assert TEXT.count("borderwidth=0") >= 2


def test_setting_changes_clear_threshold_preview():
    assert "def clear_threshold_preview(self):" in TEXT
    assert 'self.threshold_canvas.delete("all")' in TEXT
    assert "self.threshold_photo = None" in TEXT
    pending = TEXT.split("def pending(self, *_args):", 1)[1].split("def center_target_changed", 1)[0]
    assert "self.clear_threshold_preview()" in pending


def test_center_target_change_also_invalidates_threshold_preview():
    block = TEXT.split("def center_target_changed(self):", 1)[1].split("def update_center_preview_label", 1)[0]
    assert "self.clear_threshold_preview()" in block


def test_refresh_and_apply_remain_explicit_recompute_actions():
    assert "def refresh_preview(self):" in TEXT
    assert "def apply_full_resolution(self):" in TEXT
