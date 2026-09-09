from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import circle_arc_detector as cad

TEXT = Path(cad.__file__).read_text(encoding="utf-8")


def _load_block():
    return TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "\n    def previous_button_clicked", 1
    )[0]


def test_load_path_inlines_master_normalization_and_uses_shared_codec():
    block = _load_block()
    assert "cv2.IMREAD_UNCHANGED" in block
    assert "np.uint16" in block
    assert "compress_image(master_image)" in block
    assert "master_image_shape" not in block


def test_load_new_images_delegate_autot_then_restore_complete_settings():
    block = _load_block()
    create = block.index('settings = ImageSettings()')
    auto = block.index("self.threshold_auto_button_clicked()")
    restore = block.index("for setting_name, value in vars(settings).items():")
    assert create < auto < restore
    assert 'state.setdefault("auto_threshold_result"' not in block
    assert 'state.setdefault("solar_data"' not in block
    assert "self.default_settings" not in block
    assert "self.setting_variables" not in block


def test_load_existing_images_restore_stored_settings_without_autot_branch_duplication():
    block = _load_block()
    assert 'settings = state.get("settings")' in block
    assert "if settings is None:" in block
    assert "elif not isinstance(settings, ImageSettings):" in block
    assert block.count("self.threshold_auto_button_clicked()") == 1


def test_load_finishes_with_heavy_refresh_after_lightweight_restoration():
    block = _load_block()
    assert block.index("self.apply_changed_setting(setting_name, value)") < block.index(
        "self.preview_button_clicked()"
    )


def test_previous_and_next_only_select_index_then_use_load_image_at():
    previous = TEXT.split("def previous_button_clicked(self):", 1)[1].split(
        "def next_button_clicked", 1
    )[0]
    next_block = TEXT.split("def next_button_clicked(self):", 1)[1].split(
        "@contextmanager", 1
    )[0]
    assert "self.load_image_at(self.current_index - 1)" in previous
    assert "self.load_image_at(self.current_index + 1)" in next_block


def test_uint16_master_codec_roundtrip():
    master = np.zeros((5, 7, 4), np.uint16)
    master[..., 0] = 1234
    master[..., 3] = 65535
    assert np.array_equal(cad.decompress_image(cad.compress_image(master)), master)


def _load_test_app(state=None):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.root = SimpleNamespace(update_idletasks=lambda: None)
    app.image_paths = ["image.tif"]
    app.current_index = -1
    app.current_path = None
    app.master_image_payload = None
    app.gray_image = None
    app.image_state = {} if state is None else {"image.tif": state}
    app.blocked_gui = nullcontext
    app._update_navigation_state = lambda: None
    app._update_center_preview_label = lambda: None
    app.preview_button_clicked = lambda: None
    return app


def test_new_image_autot_updates_settings_before_uniform_restoration(monkeypatch):
    app = _load_test_app()
    calls = []

    def auto():
        calls.append(("auto",))
        app.image_state["image.tif"]["settings"].threshold = 17

    app.threshold_auto_button_clicked = auto
    app.apply_changed_setting = lambda name, value: calls.append((name, value))
    monkeypatch.setattr(
        cad.cv2,
        "imread",
        lambda *_: np.arange(12, dtype=np.uint8).reshape(3, 4),
    )

    app.load_image_at(0)

    settings = app.image_state["image.tif"]["settings"]
    assert isinstance(settings, cad.ImageSettings)
    assert settings.threshold == 17
    assert calls[0] == ("auto",)
    assert calls[1:] == list(vars(settings).items())


def test_existing_images_restore_stored_settings_without_invoking_autot(monkeypatch):
    settings = cad.ImageSettings(threshold=23, min_radius=1200, center_target="dark")
    app = _load_test_app({"settings": settings})
    calls = []
    app.threshold_auto_button_clicked = lambda: (_ for _ in ()).throw(
        AssertionError("Auto-T invoked for existing settings")
    )
    app.apply_changed_setting = lambda name, value: calls.append((name, value))
    monkeypatch.setattr(
        cad.cv2,
        "imread",
        lambda *_: np.arange(12, dtype=np.uint8).reshape(3, 4),
    )

    app.load_image_at(0)

    assert app.image_state["image.tif"]["settings"] is settings
    assert calls == list(vars(settings).items())
