from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_load_new_image_delegates_threshold_selection_to_auto_button():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_button_clicked", 1
    )[0]
    assert "self.threshold_auto_button_clicked()" in block
    assert "find_auto_threshold(" not in block


def test_load_restores_settings_through_apply_changed_setting_then_heavy_refresh():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_button_clicked", 1
    )[0]
    apply_pos = block.index("self.apply_changed_setting(setting_name, value)")
    refresh_pos = block.index("self.preview_button_clicked()")
    assert apply_pos < refresh_pos
