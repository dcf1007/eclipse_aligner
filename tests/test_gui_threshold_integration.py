"""Behavioral integration checks for threshold state and preview routing."""

from types import SimpleNamespace

import numpy as np

import circle_arc_detector as appmod


class FakeVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def make_state_app():
    app = object.__new__(appmod.DetectorApp)
    app.current_path = "/tmp/image.jpg"
    app.gray_image = np.array([[0, 9, 10, 255]], dtype=np.uint8)
    app.threshold = FakeVariable(10)
    app.min_radius = FakeVariable(1000)
    app.max_radius = FakeVariable(1500)
    app.max_error = FakeVariable(8.0)
    app.min_coverage = FakeVariable(8)
    app.morphology = FakeVariable(False)
    app.outer_limb_assistance = FakeVariable(False)
    app.use_horizon = FakeVariable(True)
    app.center_target = FakeVariable("light")
    app.default_settings = {
        "min_radius": 1000, "max_radius": 1500, "max_error": 8.0,
        "min_coverage": 8, "morphology": False,
        "outer_limb_assistance": False, "use_horizon": True,
        "center_target": "light",
    }
    app.setting_variables = {
        "threshold": app.threshold, "min_radius": app.min_radius,
        "max_radius": app.max_radius, "max_error": app.max_error,
        "min_coverage": app.min_coverage, "morphology": app.morphology,
        "outer_limb_assistance": app.outer_limb_assistance,
        "use_horizon": app.use_horizon, "center_target": app.center_target,
    }
    result = appmod.AutoThresholdResult(
        threshold=10, histogram_peak=20, histogram_left_edge=10, seed_threshold=15,
        coarse_threshold=12, roi_seed_threshold=12, full_seed_point=(1, 1),
        used_guard=False, resolved=True,
    )
    app.image_state = {app.current_path: {"settings": {}, "auto_threshold_result": result}}
    app.refresh_count = 0
    app._refresh_threshold_image = lambda: setattr(app, "refresh_count", app.refresh_count + 1)
    return app


def test_changed_setting_is_stored_sparsely_and_refreshes_bw_once():
    app = make_state_app()
    app.min_radius.set(900)
    app._commit_setting_change("min_radius")
    assert app.image_state[app.current_path]["settings"] == {"min_radius": 900}
    assert app.refresh_count == 1


def test_setting_returned_to_default_removes_override():
    app = make_state_app()
    app.image_state[app.current_path]["settings"]["min_radius"] = 900
    app.min_radius.set(1000)
    app._commit_setting_change("min_radius")
    assert "min_radius" not in app.image_state[app.current_path]["settings"]


def test_threshold_uses_cached_auto_result_as_its_baseline():
    app = make_state_app()
    app.threshold.set(14)
    app._commit_setting_change("threshold")
    assert app.image_state[app.current_path]["settings"]["threshold"] == 14
    app.threshold.set(10)
    app._commit_setting_change("threshold")
    assert "threshold" not in app.image_state[app.current_path]["settings"]


def test_bw_renderer_uses_exact_gray_greater_than_t_semantics():
    app = make_state_app()
    app.threshold_preview = appmod.transparent_bgra()
    app.threshold_photo = None
    app._refresh_threshold_image = appmod.DetectorApp._refresh_threshold_image.__get__(app)
    app._refresh_threshold_image()
    assert app.threshold_preview.shape == (1, 4, 4)
    assert app.threshold_preview[0, 0, 0] == 0
    assert app.threshold_preview[0, 1, 0] == 0
    assert app.threshold_preview[0, 2, 0] == 0
    assert app.threshold_preview[0, 3, 0] == 255
    assert np.all(app.threshold_preview[:, :, 3] == 255)
