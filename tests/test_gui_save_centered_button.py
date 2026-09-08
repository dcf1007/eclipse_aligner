from pathlib import Path

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def test_save_centered_button_has_explicit_button_callback_name():
    assert "command=self.save_centered_images_button" in TEXT
    assert "def save_centered_images_button(self):" in TEXT
