"""Behavioral integration checks for explicit settings and synchronous threshold routing."""
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


class FakeStatus(FakeVariable):
    pass


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
    app.status = FakeStatus("")
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
            "settings": appmod.ImageSettings(threshold=10),
            "auto_threshold_result": result,
            "solar_data": None,
        }
    }
    return app


def test_each_processing_setting_change_uses_explicit_value_and_refreshes_once():
    app = make_state_app()
    refreshes = []
    app.refresh_preview = lambda changed_setting=None, full_resolution=False: refreshes.append(
        (changed_setting, full_resolution)
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
        app.commit_setting_change(setting_name, value)
        assert app.setting_variables[setting_name].get() == value
        assert getattr(settings, setting_name) == value
    assert refreshes == [(name, False) for name in changes]


def test_returning_ordinary_setting_to_baseline_clears_override_but_threshold_remains_explicit():
    app = make_state_app()
    app.refresh_preview = lambda **_kwargs: None
    settings = app.image_state[app.current_path]["settings"]

    app.commit_setting_change("min_radius", 900)
    assert settings.min_radius == 900
    app.commit_setting_change("min_radius", 1000)
    assert settings.min_radius is None

    app.commit_setting_change("threshold", 14)
    assert settings.threshold == 14
    app.commit_setting_change("threshold", 10)
    assert settings.threshold == 10


def test_threshold_commit_synchronously_displays_raw_then_resolves_then_displays_refined(monkeypatch):
    app = make_state_app()
    app.threshold_canvas = object()
    events = []
    app.render_canvas_content = lambda _canvas, content: events.append(("display", np.asarray(content).copy()))
    refined = np.array([[False, False, True, True]])

    def fake_resolve(_gray, threshold, _state):
        events.append(("resolve", threshold))
        return refined

    monkeypatch.setattr(appmod, "resolve_threshold", fake_resolve)
    app.commit_setting_change("threshold", 12)

    assert events[0][0] == "display"
    assert np.array_equal(events[0][1], app.gray_image > 12)
    assert events[1] == ("resolve", 12)
    assert events[2][0] == "display"
    assert np.array_equal(events[2][1], refined)


def test_threshold_change_invalidates_old_t_solar_data_before_resolve(monkeypatch):
    app = make_state_app()
    app.image_state[app.current_path]["solar_data"] = appmod.SolarData(
        threshold=10,
        seed_point=(3, 0),
        component_mask=b"old",
        roi_6_5_mask=b"old",
        guard_19_5_mask=b"old",
        component_contour=np.empty((0, 2), dtype=np.uint16),
    )
    seen = []

    def fake_refresh(**_kwargs):
        seen.append(app.image_state[app.current_path]["solar_data"])

    app.refresh_preview = fake_refresh
    app.commit_setting_change("threshold", 12)
    assert seen == [None]


def test_auto_select_reuses_cached_result_and_commits_exact_value(monkeypatch):
    app = make_state_app()
    app.threshold.set(14)
    commits = []
    app.commit_setting_change = lambda name, value: commits.append((name, value))

    def fail_if_called(_gray, _state):
        raise AssertionError("cached AutoThresholdResult should have been reused")

    monkeypatch.setattr(appmod, "find_auto_threshold", fail_if_called)
    app.auto_select_threshold()
    assert commits == [("threshold", 10)]


def test_renderer_is_generic_flushes_idle_work_and_owns_canvas_cache():
    source = inspect.getsource(appmod.DetectorApp.render_canvas_content)
    assert "content" in source
    assert "canvas._unscaled_render_raster" in source
    assert "canvas._tk_photo_image" in source
    assert "update_idletasks" in source
    assert "resolve_threshold" not in source


def test_refresh_preview_has_no_threshold_specific_deferred_refinement():
    source = inspect.getsource(appmod.DetectorApp.refresh_preview)
    assert "resolve_threshold" in source
    assert "render_canvas_content" in source
    assert "self.root.after(" not in source
    assert "_finish_threshold_preview_refresh" not in source
    assert "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS" not in source
