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


class FakeRoot:
    def __init__(self):
        self.jobs = {}
        self.cancelled = []
        self._next_job = 0

    def after(self, delay_ms, callback, *args):
        self._next_job += 1
        job = f"job-{self._next_job}"
        self.jobs[job] = (delay_ms, callback, args)
        return job

    def after_cancel(self, job):
        self.cancelled.append(job)
        self.jobs.pop(job, None)

    def run_pending(self):
        jobs = list(self.jobs.items())
        self.jobs.clear()
        for _job, (_delay, callback, args) in jobs:
            callback(*args)


def make_state_app():
    app = object.__new__(appmod.DetectorApp)
    app.root = FakeRoot()
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
    app.threshold_refinement_job = None
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


def test_returning_settings_to_their_baselines_clears_sparse_overrides():
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


def test_threshold_commit_paints_raw_now_and_refined_on_next_gui_turn(monkeypatch):
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

    assert len(displayed) == 1
    assert np.array_equal(displayed[0], app.gray_image > 12)
    assert app.threshold_refinement_job is not None
    assert list(app.root.jobs.values())[0][0] == appmod.THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS

    app.root.run_pending()

    assert len(displayed) == 2
    assert np.array_equal(displayed[1], refined)
    assert np.array_equal(app.threshold_preview, refined)
    assert app.threshold_refinement_job is None


def test_non_threshold_commit_refreshes_current_preview_once():
    app = make_state_app()
    refreshes = []
    app.refresh_preview = lambda changed_setting=None, full_resolution=False: refreshes.append(
        (changed_setting, full_resolution)
    )

    app.min_radius.set(900)
    app._commit_setting_change("min_radius")

    assert refreshes == [("min_radius", False)]


def test_threshold_change_invalidates_old_t_solar_data_before_deferred_rebuild():
    app = make_state_app()
    app.image_state[app.current_path]["solar_data"] = appmod.SolarData(
        threshold=10,
        seed_point=(3, 0),
        component_mask=b"old",
        roi_6_5_mask=b"old",
        guard_19_5_mask=b"old",
        component_contour=np.empty((0, 2), dtype=np.uint16),
    )

    app.threshold.set(12)
    app._commit_setting_change("threshold")

    assert app.image_state[app.current_path]["solar_data"] is None
    assert app.threshold_refinement_job is not None


def test_newer_threshold_commit_cancels_pending_old_t_stage(monkeypatch):
    app = make_state_app()
    calls = []
    monkeypatch.setattr(
        appmod,
        "build_solar_data_at_threshold",
        lambda _gray, threshold, _state: calls.append(threshold) or (_gray > threshold),
    )

    app.threshold.set(12)
    app._commit_setting_change("threshold")
    first_job = app.threshold_refinement_job

    app.threshold.set(14)
    app._commit_setting_change("threshold")

    assert first_job in app.root.cancelled
    assert len(app.root.jobs) == 1
    app.root.run_pending()
    assert calls == [14]


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


def test_refresh_preview_defers_refinement_instead_of_replacing_raw_in_same_callback():
    source = inspect.getsource(appmod.DetectorApp.refresh_preview)
    assert "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS" in source
    assert "self.root.after(" in source
    assert "_finish_threshold_preview_refresh" in source


def test_generic_display_refresh_does_not_recompute_threshold_or_solar_data():
    source = inspect.getsource(appmod.DetectorApp._refresh_display_images)
    assert "gray_image >" not in source
    assert "cv2.compare" not in source
    assert "build_solar_data_at_threshold" not in source
