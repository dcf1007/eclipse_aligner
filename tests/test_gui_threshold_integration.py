"""Behavioral integration checks for per-image settings and threshold preview routing."""

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
    app.default_settings = appmod.ImageSettings(
        min_radius=1000,
        max_radius=1500,
        max_error=8.0,
        min_coverage=8,
        morphology=False,
        outer_limb_assistance=False,
        use_horizon=True,
        center_target="light",
    )
    app.setting_variables = {
        "threshold": app.threshold,
        "min_radius": app.min_radius,
        "max_radius": app.max_radius,
        "max_error": app.max_error,
        "min_coverage": app.min_coverage,
        "morphology": app.morphology,
        "outer_limb_assistance": app.outer_limb_assistance,
        "use_horizon": app.use_horizon,
        "center_target": app.center_target,
    }
    result = appmod.AutoThresholdResult(
        threshold=10,
        histogram_peak=20,
        histogram_left_edge=10,
        seed_threshold=15,
        coarse_threshold=12,
        roi_seed_threshold=12,
        full_seed_point=(1, 1),
        used_guard=False,
        resolved=True,
    )
    app.image_state = {
        app.current_path: {
            "settings": appmod.ImageSettings(),
            "auto_threshold_result": result,
        }
    }
    app.refresh_count = 0
    app._refresh_threshold_image = lambda: setattr(
        app, "refresh_count", app.refresh_count + 1
    )
    return app


def test_each_processing_setting_is_stored_only_when_it_differs_from_baseline():
    app = make_state_app()
    changes = {
        "threshold": 14,
        "min_radius": 900,
        "max_radius": 1400,
        "max_error": 9.5,
        "min_coverage": 12,
        "morphology": True,
        "outer_limb_assistance": True,
        "use_horizon": False,
        "center_target": "dark",
    }

    settings = app.image_state[app.current_path]["settings"]
    for setting_name, value in changes.items():
        app.setting_variables[setting_name].set(value)
        app._commit_setting_change(setting_name)
        assert getattr(settings, setting_name) == value

    assert app.refresh_count == len(changes)


def test_returning_settings_to_their_baselines_clears_the_sparse_overrides():
    app = make_state_app()
    settings = app.image_state[app.current_path]["settings"]

    app.min_radius.set(900)
    app._commit_setting_change("min_radius")
    assert settings.min_radius == 900
    app.min_radius.set(1000)
    app._commit_setting_change("min_radius")
    assert settings.min_radius is None

    app.threshold.set(14)
    app._commit_setting_change("threshold")
    assert settings.threshold == 14
    app.threshold.set(10)
    app._commit_setting_change("threshold")
    assert settings.threshold is None


def test_auto_select_reuses_cached_result_without_rerunning_algorithm(monkeypatch):
    app = make_state_app()
    app.status = FakeVariable("")
    app.threshold.set(14)
    commits = []
    app._commit_setting_change = lambda name: commits.append(name)

    def fail_if_called(_gray):
        raise AssertionError("cached AutoThresholdResult should have been reused")

    monkeypatch.setattr(appmod, "auto_threshold", fail_if_called)
    app.auto_select_threshold()

    assert app.threshold.get() == 10
    assert commits == ["threshold"]


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
