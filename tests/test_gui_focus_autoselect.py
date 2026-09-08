from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_auto_buttons_bind_to_explicit_button_callbacks():
    assert "command=self.auto_select_threshold_button" in TEXT
    assert "command=self.auto_select_radius_button" in TEXT


def test_auto_threshold_button_uses_parent_then_lightweight_setting_path_only():
    block = TEXT.split("def auto_select_threshold_button(self):", 1)[1].split(
        "def auto_select_radius_button", 1
    )[0]
    assert "find_auto_threshold(" in block
    assert 'self.apply_changed_setting("threshold", selected_threshold)' in block
    assert "find_separation_threshold(" not in block
    assert "refine_threshold(" not in block
    assert "refresh_preview_button(" not in block
    assert "render_canvas_content(" not in block
