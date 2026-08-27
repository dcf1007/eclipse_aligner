"""Regression checks for removal of the obsolete threshold-candidate palette."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_threshold_candidate_palette_is_absent():
    assert "Useful threshold candidates:" not in TEXT
    assert "self.palette_frame" not in TEXT
    assert "threshold candidate generation" not in TEXT


def test_control_rows_close_gap_after_palette_removal():
    assert 'button_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))' in TEXT
    assert ').grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))' in TEXT
