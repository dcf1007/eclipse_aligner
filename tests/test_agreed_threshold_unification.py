from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FakeStatus:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class FakeRoot:
    def __init__(self):
        self.idle_calls = 0

    def update_idletasks(self):
        self.idle_calls += 1


class FakeCanvas:
    def __init__(self):
        self._tk_photo_image = None

    def winfo_width(self):
        return 100

    def winfo_height(self):
        return 80

    def delete(self, *_args):
        pass

    def create_image(self, *_args, **_kwargs):
        pass


def disk_gray(size=81, radius=20, threshold=100):
    gray = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(gray, (center, center), radius, 180, -1)
    gray[center, center] = 240
    return gray, threshold, center


def test_seed_support_is_explicit_5x5_square_with_no_fallback():
    kernel = cad.generate_kernel(cad.TRACKING_SEED_KERNEL_SIZE, round_kernel=False)
    assert kernel.shape == (5, 5)
    assert np.all(kernel == 1)
    gray = np.zeros((21, 21), np.uint8)
    thin = np.zeros_like(gray)
    thin[10, 4:17] = 255
    gray[10, 10] = 250
    assert cad.brightest_supported_component_point(gray, thin, kernel) is None

def test_seed_rule_is_brightest_supported_then_innermost_tie_break():
    gray = np.zeros((51, 51), np.uint8)
    component = np.zeros_like(gray)
    cv2.circle(component, (25, 25), 15, 255, -1)

    # Unsupported boundary hot pixel must not win.
    gray[25, 40] = 255
    # Two equally bright supported pixels; center is farther from the boundary.
    gray[25, 25] = 240
    gray[25, 30] = 240

    assert cad.brightest_supported_component_point(gray, component, cad.generate_kernel(5)) == (25, 25)


def test_find_auto_threshold_writes_result_and_returns_threshold():
    gray = np.zeros((81, 81), np.uint8)
    center = 40
    cv2.circle(gray, (center, center), 20, 220, -1)
    state = {
        "settings": cad.ImageSettings(),
        "auto_threshold_result": None,
        "solar_data": None,
    }

    threshold = cad.find_auto_threshold(gray, state)

    result = state["auto_threshold_result"]
    assert isinstance(result, cad.AutoThresholdResult)
    assert threshold == result.threshold
    assert isinstance(threshold, int)
    assert result.full_res_seed_point == (center, center)


def test_resolve_threshold_chooses_authoritative_seed_before_refinement_and_persists_same_mask():
    gray = np.zeros((81, 81), np.uint8)
    component_center = (40, 40)
    cv2.circle(gray, component_center, 18, 180, -1)
    gray[component_center[1], component_center[0]] = 240

    # Add a thin connected spur with a brighter endpoint. It is part of the raw
    # component but cannot become the seed because it lacks 5x5 support.
    gray[40, 58:66] = 255

    state = {"settings": cad.ImageSettings(threshold=100), "auto_threshold_result": None, "solar_data": None}
    refined = cad.resolve_threshold(gray, 100, state)
    solar = state["solar_data"]

    assert isinstance(solar, cad.SolarData)
    assert solar.threshold == 100
    assert solar.seed_point == component_center
    assert refined[component_center[1], component_center[0]]
    assert gray[solar.seed_point[1], solar.seed_point[0]] > 100

    stored = cad.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(stored, refined)
    assert not refined[40, 65]  # spur endpoint removed by the shared 7x7 OPEN/CLOSE


def test_resolve_threshold_reuses_exact_t_solardata_and_preserves_seed(monkeypatch):
    gray, threshold, center = disk_gray()
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}
    first = cad.resolve_threshold(gray, threshold, state)
    first_solar = state["solar_data"]
    first_seed = first_solar.seed_point

    def fail(*_args, **_kwargs):
        raise AssertionError("same-T SolarData should be validated/reused, not re-resolved")

    monkeypatch.setattr(cad, "largest_enclosed_bright_component", fail)
    second = cad.resolve_threshold(gray, threshold, state)

    assert state["solar_data"] is first_solar
    assert state["solar_data"].seed_point == first_seed == (center, center)
    assert np.array_equal(second, first)


def test_commit_setting_change_always_stores_initialized_threshold_even_at_auto_value():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"
    app.gray_image = np.ones((5, 5), np.uint8)
    app.threshold = FakeVar(3)
    app.min_radius = FakeVar(100)
    app.setting_variables = {"threshold": app.threshold, "min_radius": app.min_radius}
    app.default_settings = cad.ImageSettings(min_radius=100)
    auto = cad.AutoThresholdResult(
        threshold=17,
        histogram_start_threshold=17,
        work_res_threshold=17,
        full_res_seed_point=(2, 2),
        resolved=True,
    )
    app.image_state = {
        "image": {
            "settings": cad.ImageSettings(),
            "auto_threshold_result": auto,
            "solar_data": None,
        }
    }
    calls = []
    app.refresh_preview = lambda **kwargs: calls.append(kwargs)

    app.commit_setting_change("threshold", 17)

    assert app.threshold.get() == 17
    assert app.image_state["image"]["settings"].threshold == 17
    assert calls == [{"changed_setting": "threshold"}]


def test_commit_threshold_change_invalidates_old_t_solardata():
    gray, threshold, _ = disk_gray()
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}
    cad.resolve_threshold(gray, threshold, state)
    assert state["solar_data"].threshold == threshold

    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"
    app.gray_image = gray
    app.threshold = FakeVar(threshold)
    app.setting_variables = {"threshold": app.threshold}
    app.default_settings = cad.ImageSettings()
    app.image_state = {"image": state}
    app.refresh_preview = lambda **_kwargs: None

    app.commit_setting_change("threshold", threshold + 1)
    assert state["settings"].threshold == threshold + 1
    assert state["solar_data"] is None


def test_refresh_preview_is_raw_display_then_synchronous_resolve_then_refined_display(monkeypatch):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"
    app.gray_image = np.array([[0, 20], [30, 5]], np.uint8)
    app.status = FakeStatus()
    app.threshold_canvas = object()
    state = {"settings": cad.ImageSettings(threshold=10), "auto_threshold_result": None, "solar_data": None}
    app.image_state = {"image": state}

    events = []
    app.render_canvas_content = lambda _canvas, content: events.append(("display", np.asarray(content).copy()))

    refined = np.array([[False, True], [False, False]])

    def fake_resolve(gray, threshold, image_state):
        events.append(("resolve", threshold))
        assert gray is app.gray_image
        assert image_state is state
        return refined

    monkeypatch.setattr(cad, "resolve_threshold", fake_resolve)

    app.refresh_preview()

    assert events[0][0] == "display"
    assert np.array_equal(events[0][1], app.gray_image > 10)
    assert events[1] == ("resolve", 10)
    assert events[2][0] == "display"
    assert np.array_equal(events[2][1], refined)


def test_full_resolution_flag_is_placeholder_using_same_resolver(monkeypatch):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"
    app.gray_image = np.full((9, 9), 200, np.uint8)
    app.status = FakeStatus()
    app.image_state = {"image": {"settings": cad.ImageSettings(threshold=100), "auto_threshold_result": None, "solar_data": None}}
    calls = []

    def fake_resolve(gray, threshold, state):
        calls.append((id(gray), threshold, id(state)))
        return gray > threshold

    monkeypatch.setattr(cad, "resolve_threshold", fake_resolve)
    app.refresh_preview(full_resolution=False)
    app.refresh_preview(full_resolution=True)

    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_source_has_no_threshold_specific_deferred_refinement_or_duplicate_preview_state():
    source = Path(cad.__file__).read_text(encoding="utf-8")
    assert "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS" not in source
    assert "_finish_threshold_preview_refresh" not in source
    assert "threshold_refinement_job" not in source
    assert "self.threshold_preview" not in source
    assert "def _commit_setting_change" not in source
    assert "def commit_setting_change(self, setting_name, value)" in source


def test_render_canvas_content_owns_update_idletasks_flush():
    source = inspect.getsource(cad.DetectorApp.render_canvas_content)
    assert "self.root.update_idletasks()" in source
    assert ".after(" not in source
    assert "resolve_threshold" not in source


def test_load_image_initializes_missing_threshold_through_auto_then_commit():
    source = inspect.getsource(cad.DetectorApp.load_image_at)
    none_pos = source.index("if settings.threshold is None:")
    auto_pos = source.index("threshold = find_auto_threshold(self.gray_image, state)")
    commit_pos = source.index('self.commit_setting_change("threshold", threshold)')
    assert none_pos < auto_pos < commit_pos


def test_resolve_threshold_is_atomic_solardata_writer():
    source = inspect.getsource(cad.resolve_threshold)
    assert 'image_state["solar_data"] = solar_data' in source
    assert "component_mask=compress_full_mask(refined_component)" in source
    assert "full_res_seed_kernel = generate_kernel" in source
    assert "Authoritative solar seed did not survive" in source


def test_resolve_threshold_errors_if_authoritative_unrefined_seed_does_not_survive(monkeypatch):
    gray, threshold, center = disk_gray()
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}

    real_refine = cad.refine_solar_component_mask

    def remove_seed(component):
        refined = real_refine(component)
        refined[center, center] = False
        return refined

    monkeypatch.setattr(cad, "refine_solar_component_mask", remove_seed)
    with pytest.raises(cad.ThresholdResolutionError, match="did not survive"):
        cad.resolve_threshold(gray, threshold, state)
    assert state["solar_data"] is None


def test_render_canvas_content_flushes_idle_tasks_and_keeps_only_display_layer_cache(monkeypatch):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.root = FakeRoot()
    canvas = FakeCanvas()

    class DummyPhoto:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cad.tk, "PhotoImage", DummyPhoto)
    content = np.array([[False, True], [True, False]])
    app.render_canvas_content(canvas, content)

    assert app.root.idle_calls == 1
    assert hasattr(canvas, "_unscaled_render_raster")
    assert canvas._unscaled_render_raster.shape == (2, 2, 4)
    assert hasattr(canvas, "_tk_photo_image")
    assert not hasattr(app, "threshold_preview")


def test_all_current_gui_setting_change_callers_pass_explicit_value():
    source = Path(cad.__file__).read_text(encoding="utf-8")
    # No one-argument call form remains in the application source.
    import re
    one_arg = re.findall(r"self\.commit_setting_change\([^,\n()]+\)", source)
    assert one_arg == []


def test_resize_redraw_reuses_each_canvas_own_retained_raster():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.canvas_redraw_job = "pending"
    threshold_canvas = type("Canvas", (), {})()
    color_canvas = type("Canvas", (), {})()
    threshold_canvas._unscaled_render_raster = np.full((2, 2, 4), 11, np.uint8)
    color_canvas._unscaled_render_raster = np.full((3, 3, 4), 22, np.uint8)
    app.threshold_canvas = threshold_canvas
    app.color_canvas = color_canvas

    calls = []
    app.render_canvas_content = lambda canvas, raster: calls.append(
        (canvas, np.asarray(raster).copy())
    )

    app._redraw_cached_canvases()

    assert app.canvas_redraw_job is None
    assert len(calls) == 2
    assert calls[0][0] is threshold_canvas
    assert np.all(calls[0][1] == 11)
    assert calls[1][0] is color_canvas
    assert np.all(calls[1][1] == 22)
