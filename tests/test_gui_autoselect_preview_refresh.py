"""Regression checks for threshold Auto-select preview-refresh event behavior."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def block(start_marker: str, end_marker: str) -> str:
    return TEXT.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_source_parses():
    ast.parse(TEXT)


def test_threshold_autoselect_uses_preview_refresh_hook():
    auto = block("    def auto_select_threshold(self):", "    def auto_select_radius")
    assert "self.refresh_preview()" in auto


def test_gui_only_branch_stays_backend_free():
    assert "cv2.COLOR_BGR2GRAY" not in TEXT
    assert "cv2.COLOR_BGRA2GRAY" not in TEXT
    assert "cv2.threshold(" not in TEXT
    assert "def auto_threshold_from_gray(" not in TEXT
