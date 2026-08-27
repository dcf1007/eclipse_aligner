"""Regression checks for releasing slider focus on outside mouse clicks."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")

def test_source_parses():
    ast.parse(TEXT)

def test_root_observes_mouse_clicks_for_focus_release():
    assert 'root.bind("<ButtonPress-1>", self._release_slider_focus_if_clicked_elsewhere, add="+")' in TEXT

def test_outside_click_releases_only_scale_focus():
    assert 'def _release_slider_focus_if_clicked_elsewhere(self, event):' in TEXT
    assert 'focused = self.root.focus_get()' in TEXT
    assert 'isinstance(focused, tk.Scale)' in TEXT
    assert 'event.widget is not focused' in TEXT
    assert 'self.root.focus_set()' in TEXT

def test_scale_click_still_claims_focus():
    assert 'slider.bind("<ButtonPress-1>", self._focus_slider, add="+")' in TEXT
    assert 'event.widget.focus_set()' in TEXT
