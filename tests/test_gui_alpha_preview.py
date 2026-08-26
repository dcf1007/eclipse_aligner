from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")

spec = importlib.util.spec_from_file_location("gui_alpha_preview", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_transparent_bgra_has_real_alpha_channel():
    frame = module.transparent_bgra(7, 5)
    assert frame.shape == (5, 7, 4)
    assert frame.dtype == np.uint8
    assert np.all(frame[:, :, 3] == 0)


def test_loaded_bgr_becomes_opaque_bgra():
    source = np.zeros((3, 4, 3), np.uint8)
    source[:, :, 1] = 123
    frame = module.opaque_bgra(source)
    assert frame.shape == (3, 4, 4)
    assert np.all(frame[:, :, 3] == 255)
    assert np.all(frame[:, :, 1] == 123)


def test_transparency_is_not_faked_with_canvas_background():
    assert 'preview_background = frame.cget("background")' not in TEXT
    assert "def transparent_bgra(" in TEXT
    assert "alpha=0" in TEXT or "alpha = 0" in TEXT


def test_no_in_pane_placeholder_text():
    assert "def placeholder(" not in TEXT
    assert "create_text(" not in TEXT


def test_setting_changes_replace_threshold_with_transparent_frame():
    start = TEXT.index("    def clear_threshold_preview(self):")
    end = TEXT.index("    def pending(", start)
    block = TEXT[start:end]
    assert "transparent_bgra(" in block
    assert "self.show_image(" in block


def test_canvas_is_only_display_surface():
    assert 'bg="#202020"' in TEXT
    assert "Transparency is retained in the" in TEXT
