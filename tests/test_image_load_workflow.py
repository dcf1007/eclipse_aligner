from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_load_does_not_use_auto_button_as_processing_operation():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_button_clicked", 1
    )[0]
    assert "find_auto_threshold(" in block
    assert "threshold_auto_button_clicked(" not in block


def test_load_restores_settings_through_apply_changed_setting_then_heavy_refresh():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_button_clicked", 1
    )[0]
    apply_pos = block.index("self.apply_changed_setting(setting_name, value)")
    refresh_pos = block.index("self.preview_button_clicked()")
    assert apply_pos < refresh_pos
