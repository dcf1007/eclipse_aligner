from pathlib import Path
import ast
import circle_arc_detector as cad
ROOT=Path(__file__).resolve().parents[1]; TEXT=(ROOT/'circle_arc_detector.py').read_text()

def test_gui_source_parses(): ast.parse(TEXT)
def test_center_target_is_light_by_default():
    assert cad.ImageSettings().center_target == "light"
    assert "dataclasses.fields(ImageSettings)" in TEXT
def test_center_controls_are_mutually_exclusive_radiobuttons():
    assert TEXT.count('tk.Radiobutton(')==2 and 'variable=self.center_target' in TEXT and 'value="light"' in TEXT and 'value="dark"' in TEXT
def test_heavy_detection_stages_are_explicit_placeholders_in_threshold_branch():
    assert 'def detect(' not in TEXT and 'def fit_ellipse' not in TEXT
    assert '# TODO: horizon finding consumes solar_data.' in TEXT
    assert 'downstream preview ' in TEXT and 'processing is not implemented in the threshold-finder branch.' in TEXT
