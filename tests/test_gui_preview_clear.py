"""Regression checks for alpha-backed previews and refresh event semantics."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_preview_frames_have_no_placeholder_text():
    assert "def placeholder(" not in TEXT
    assert "create_text(" not in TEXT


def test_empty_preview_is_real_bgra_alpha_zero():
    assert "def transparent_bgra(" in TEXT
    assert "np.zeros((height, width, 4)" in TEXT
    assert "alpha = 0" in TEXT or "alpha=0" in TEXT


def test_transparency_is_not_faked_by_tk_background_matching():
    assert 'preview_background = frame.cget("background")' not in TEXT
    assert "Transparency is retained in the" in TEXT


def test_discrete_setting_changes_request_preview_refresh():
    pending = TEXT.split("def pending(self, *_args):", 1)[1].split("def center_target_changed", 1)[0]
    assert "self.refresh_preview()" in pending
    assert "self.clear_threshold_preview()" not in pending


def test_center_target_change_requests_preview_refresh():
    block = TEXT.split("def center_target_changed(self):", 1)[1].split("def update_center_preview_label", 1)[0]
    assert "self.refresh_preview()" in block
    assert "self.clear_threshold_preview()" not in block


def test_refresh_and_apply_actions_remain_present():
    assert "def refresh_preview(self):" in TEXT
    assert "def apply_full_resolution(self):" in TEXT
