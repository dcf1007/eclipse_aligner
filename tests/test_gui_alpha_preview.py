from pathlib import Path
import importlib.util
import numpy as np
import cv2
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")

spec = importlib.util.spec_from_file_location("gui_alpha_preview", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_loaded_bgr_becomes_lossless_opaque_uint16_bgra(tmp_path, monkeypatch):
    source = np.zeros((3, 4, 3), np.uint8)
    source[:, :, 1] = 123
    path = tmp_path / "source.tif"
    assert cv2.imwrite(str(path), source)

    root = tk.Tk()
    app = module.DetectorApp(root, [])
    try:
        app.image_paths = [str(path)]
        app.preview_button_clicked = lambda: None

        def auto_threshold(gray, state):
            state["auto_threshold_result"] = module.AutoThresholdResult(
                failure_reason="test Auto-T stop"
            )
            return None

        monkeypatch.setattr(module, "find_auto_threshold", auto_threshold)
        monkeypatch.setattr(
            module,
            "resolve_threshold",
            lambda gray, threshold, state: gray > threshold,
        )
        app.load_image_at(0)
        master = module.decompress_image(app.master_image_payload)
    finally:
        root.destroy()

    assert master.shape == (3, 4, 4)
    assert master.dtype == np.uint16
    assert np.all(master[:, :, 3] == 65535)
    assert np.all(master[:, :, 1] == 123 * 257)
    assert app.color_canvas._rendered_content is app.master_image_payload
    assert isinstance(app.color_canvas._rendered_content, bytes)
    assert "def opaque_bgra(" not in TEXT


def test_empty_canvas_uses_none_instead_of_a_transparent_placeholder_raster():
    assert 'preview_background = frame.cget("background")' not in TEXT
    assert "def transparent_bgra(" not in TEXT
    assert "self.render_canvas_content(self.threshold_canvas, None)" in TEXT
    assert "self.render_canvas_content(self.color_canvas, None)" in TEXT


def test_no_in_pane_placeholder_text():
    assert "def placeholder(" not in TEXT
    assert "create_text(" not in TEXT


def test_canvas_is_only_display_surface():
    assert 'bg="#202020"' in TEXT
    assert "Transparency is retained in the" in TEXT


def test_new_image_uses_and_stores_default_threshold_when_auto_t_fails(
    tmp_path,
    monkeypatch,
):
    source = np.full((9, 9, 3), 200, np.uint8)
    path = tmp_path / "source.jpg"
    assert cv2.imwrite(str(path), source)

    root = tk.Tk()
    app = module.DetectorApp(root, [])
    try:
        app.image_paths = [str(path)]
        app.preview_button_clicked = lambda: None
        result = module.AutoThresholdResult(failure_reason="test Auto-T failure")

        def auto_threshold(gray, state):
            state["auto_threshold_result"] = result
            return None

        resolved_thresholds = []

        def resolve(gray, threshold, state):
            resolved_thresholds.append(threshold)
            return gray > threshold

        monkeypatch.setattr(module, "find_auto_threshold", auto_threshold)
        monkeypatch.setattr(module, "resolve_threshold", resolve)

        app.load_image_at(0)
        settings = app.image_state[str(path)]["settings"]

        assert settings.threshold == 8
        assert app.threshold.get() == 8
        assert resolved_thresholds
        assert set(resolved_thresholds) == {8}
    finally:
        root.destroy()
