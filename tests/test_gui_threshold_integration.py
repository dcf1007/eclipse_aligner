"""Behavioral integration checks for per-image settings and threshold preview routing."""

import inspect

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
    app.status = FakeVariable("")
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
        full_seed_point=(3, 0),
        used_guard=False,
        resolved=True,
    )
    app.image_state = {
        app.current_path: {
            "settings": appmod.ImageSettings(),
            "auto_threshold_result": result,
            "solar_data": None,
        }
    }
    return app


def test_each_processing_setting_is_stored_only_when_it_differs_from_baseline(monkeypatch):
    app = make_state_app()
    refined = np.array([[False, False, False, True]])
    monkeypatch.setattr(
        appmod,
        "build_solar_data_at_threshold",
        lambda _gray, _threshold, _state: refined,
    )
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


def test_returning_settings_to_their_baselines_clears_sparse_overrides(monkeypatch):
    app = make_state_app()
    refined = np.array([[False, False, False, True]])
    monkeypatch.setattr(
        appmod,
        "build_solar_data_at_threshold",
        lambda _gray, _threshold, _state: refined,
    )
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


def test_threshold_commit_displays_pure_threshold_then_final_refined_component(monkeypatch):
    app = make_state_app()
    app.threshold_canvas = object()
    displayed = []
    app.display_on_canvas = lambda canvas, content: displayed.append(np.asarray(content).copy())
    refined = np.array([[False, False, True, True]])
    monkeypatch.setattr(
        appmod,
        "build_solar_data_at_threshold",
        lambda _gray, _threshold, _state: refined,
    )

    app.threshold.set(12)
    app._commit_setting_change("threshold")

    assert len(displayed) == 2
    assert np.array_equal(displayed[0], app.gray_image > 12)
    assert np.array_equal(displayed[1], refined)
    assert np.array_equal(app.threshold_preview, refined)


def test_non_threshold_commit_does_not_touch_threshold_canvas_or_solar_data(monkeypatch):
    app = make_state_app()
    app.threshold_canvas = object()
    app.display_on_canvas = lambda *_args: (_ for _ in ()).throw(
        AssertionError("non-threshold commit must not redraw threshold canvas")
    )
    monkeypatch.setattr(
        appmod,
        "build_solar_data_at_threshold",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("non-threshold commit must not rebuild SolarData")
        ),
    )

    app.min_radius.set(900)
    app._commit_setting_change("min_radius")


def test_auto_select_reuses_cached_result_without_rerunning_algorithm(monkeypatch):
    app = make_state_app()
    app.threshold.set(14)
    commits = []
    app._commit_setting_change = lambda name: commits.append(name)

    def fail_if_called(_gray):
        raise AssertionError("cached AutoThresholdResult should have been reused")

    monkeypatch.setattr(appmod, "auto_threshold", fail_if_called)
    app.auto_select_threshold()

    assert app.threshold.get() == 10
    assert commits == ["threshold"]


def test_display_on_canvas_is_generic_and_flushes_pending_repaint_work():
    source = inspect.getsource(appmod.DetectorApp.display_on_canvas)
    assert "canvas_content" in source
    assert "canvas._display_photo" in source
    assert "update_idletasks" in source
    assert not hasattr(appmod.DetectorApp, "_show_image_on_canvas")
    assert not hasattr(appmod.DetectorApp, "_refresh_threshold_image")


def test_generic_display_refresh_does_not_recompute_threshold_or_solar_data():
    source = inspect.getsource(appmod.DetectorApp._refresh_display_images)
    assert "gray_image >" not in source
    assert "cv2.compare" not in source
    assert "build_solar_data_at_threshold" not in source
