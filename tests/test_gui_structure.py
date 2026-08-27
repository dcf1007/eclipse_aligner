"""Static regression checks for the GUI-only milestone.

These tests avoid creating a Tk window, so they are safe in headless CI.
"""

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_gui_source_parses():
    ast.parse(TEXT)


def test_center_target_is_light_by_default():
    assert 'self.center_target = tk.StringVar(value="light")' in TEXT


def test_center_controls_are_mutually_exclusive_radiobuttons():
    assert TEXT.count("tk.Radiobutton(") == 2
    assert 'variable=self.center_target' in TEXT
    assert 'value="light"' in TEXT
    assert 'value="dark"' in TEXT


def test_detection_is_not_implemented_in_gui_milestone():
    assert "def detect(" not in TEXT
    assert "def fit_ellipse" not in TEXT
    assert "detector backend not implemented" in TEXT
