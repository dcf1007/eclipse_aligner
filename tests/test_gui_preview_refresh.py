"""Behavioral checks for lightweight setting commits and full preview entry points."""

from types import SimpleNamespace

import cv2
import numpy as np

import circle_arc_detector as gui


class FakeStatus:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class FakeStringVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def test_completed_setting_change_only_refreshes_threshold_image():
    app = object.__new__(gui.DetectorApp)
    calls = []
    app._refresh_threshold_image = lambda: calls.append("threshold")
    app.refresh_preview = lambda: calls.append("full")

    app._commit_setting_change()

    assert calls == ["threshold"]


def test_threshold_auto_select_uses_completed_setting_change_path():
    app = object.__new__(gui.DetectorApp)
    app.status = FakeStatus()
    calls = []
    app._commit_setting_change = lambda: calls.append("commit")
    app.refresh_preview = lambda: calls.append("full")

    app.auto_select_threshold()

    assert calls == ["commit"]


def test_center_target_change_updates_label_and_commits_once():
    app = object.__new__(gui.DetectorApp)
    app.center_target = FakeStringVar("dark")
    app.center_preview_label = SimpleNamespace(set=lambda value: setattr(app, "label", value))
    app.status = FakeStatus()
    calls = []
    app._commit_setting_change = lambda: calls.append("commit")

    app._handle_center_target_change()

    assert calls == ["commit"]
    assert app.label == "Full-color image — center on dark ellipse"


def test_readable_image_load_invokes_full_preview_processing(tmp_path):
    path = tmp_path / "image.png"
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    app = object.__new__(gui.DetectorApp)
    app.image_paths = [str(path)]
    app.current_index = -1
    app.current_path = None
    app.color_image = None
    app.threshold_preview = gui.transparent_bgra()
    app.threshold_photo = None
    app.color_photo = None
    app.status = FakeStatus()
    app.full_calls = 0
    app.refresh_preview = lambda: setattr(app, "full_calls", app.full_calls + 1)
    app._redraw_previews = lambda: None
    app.update_navigation_state = lambda: None

    app.load_image_at(0)

    assert app.full_calls == 1
    assert app.color_image.shape == (5, 7, 4)
