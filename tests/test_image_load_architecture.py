from pathlib import Path

import numpy as np

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def _load_block():
    return TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "\n    def previous_image_button", 1
    )[0]


def test_load_path_inlines_master_normalization_and_uses_shared_codec():
    block = _load_block()
    assert "cv2.IMREAD_UNCHANGED" in block
    assert "np.uint16" in block
    assert "compress_image(master_image)" in block
    assert "master_image_shape" not in block


def test_load_runs_complete_autot_before_restoring_settings():
    block = _load_block()
    auto = block.index("automatic_threshold = find_auto_threshold(")
    restore = block.index("for setting_name in self.setting_variables:")
    assert auto < restore
    assert "self.apply_changed_setting(setting_name, value)" in block


def test_load_restored_threshold_overrides_auto_and_missing_threshold_uses_auto():
    block = _load_block()
    assert "settings.threshold" in block and "if settings.threshold is not None" in block
    assert "automatic_threshold" in block


def test_load_finishes_with_heavy_refresh_after_lightweight_restoration():
    block = _load_block()
    assert block.index("self.apply_changed_setting(setting_name, value)") < block.index(
        "self.refresh_preview_button()"
    )


def test_previous_and_next_only_select_index_then_use_load_image_at():
    previous = TEXT.split("def previous_image_button(self):", 1)[1].split(
        "def next_image_button", 1
    )[0]
    next_block = TEXT.split("def next_image_button(self):", 1)[1].split(
        "@contextmanager", 1
    )[0]
    assert "self.load_image_at(self.current_index - 1)" in previous
    assert "self.load_image_at(self.current_index + 1)" in next_block


def test_uint16_master_codec_roundtrip():
    master = np.zeros((5, 7, 4), np.uint16)
    master[..., 0] = 1234
    master[..., 3] = 65535
    assert np.array_equal(cad.decompress_image(cad.compress_image(master)), master)
