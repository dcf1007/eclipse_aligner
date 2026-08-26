"""Static regression checks for slider focus and Auto select GUI controls."""

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_sliders_explicitly_keep_keyboard_focus_after_click():
    assert 'takefocus=True' in TEXT
    assert 'scale.bind("<ButtonPress-1>", self.focus_scale, add="+")' in TEXT
    assert 'scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")' in TEXT
    assert 'event.widget.focus_set()' in TEXT


def test_threshold_has_auto_select_button_on_its_row():
    assert 'self.threshold_auto_button = tk.Button(' in TEXT
    assert 'command=self.auto_select_threshold' in TEXT
    assert 'row=0, column=3' in TEXT


def test_radius_auto_select_button_spans_min_and_max_rows():
    assert 'self.radius_auto_button = tk.Button(' in TEXT
    assert 'command=self.auto_select_radius' in TEXT
    assert 'row=1, column=3, rowspan=2' in TEXT


def test_auto_select_buttons_are_ui_only_placeholders():
    assert 'Auto select threshold: algorithm not implemented' in TEXT
    assert 'Auto select radius range: algorithm not implemented' in TEXT
    assert 'def detect(' not in TEXT
