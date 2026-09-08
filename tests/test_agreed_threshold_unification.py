from contextlib import nullcontext
import inspect
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import circle_arc_detector as cad


class Var:
    def __init__(self, value=None): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def disk_gray(size=81, radius=20, threshold=100):
    gray = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(gray, (center, center), radius, 180, -1)
    gray[center, center] = 240
    return gray, threshold, center


def test_seed_support_is_explicit_5x5_square_with_no_fallback():
    kernel = cad.generate_kernel((5, 5), round_kernel=False)
    gray = np.zeros((21, 21), np.uint8)
    thin = np.zeros_like(gray)
    thin[10, 4:17] = 255
    gray[10, 10] = 250
    assert np.all(kernel == 1)
    assert cad.brightest_supported_component_point(gray, thin, kernel) is None


def test_seed_rule_is_brightest_supported_then_innermost_tie_break():
    gray = np.zeros((51, 51), np.uint8)
    component = np.zeros_like(gray)
    cv2.circle(component, (25, 25), 15, 255, -1)
    gray[25, 40] = 255
    gray[25, 25] = gray[25, 30] = 240
    assert cad.brightest_supported_component_point(gray, component, cad.generate_kernel((5, 5))) == (25, 25)


def test_find_auto_threshold_owns_result_and_returns_refined_threshold():
    gray, _, center = disk_gray()
    state = {"settings": cad.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    threshold = cad.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert isinstance(result, cad.AutoThresholdResult)
    assert threshold == result.full_res_refined_threshold
    assert result.threshold_refinement_complete
    assert result.full_res_seed_point == (center, center)


def test_resolver_persists_exact_returned_component_and_authoritative_seed():
    gray, threshold, center = disk_gray()
    gray[center, center + 18:center + 26] = 255
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}
    resolved = cad.resolve_threshold(gray, threshold, state)
    solar = state["solar_data"]
    assert solar.threshold == threshold
    assert solar.seed_point == (center, center)
    assert np.array_equal(cad.decompress_image(solar.component_mask), resolved)
    assert resolved[center, center]


def test_resolver_reuses_exact_t_solardata(monkeypatch):
    gray, threshold, _ = disk_gray()
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}
    first = cad.resolve_threshold(gray, threshold, state)
    solar = state["solar_data"]
    monkeypatch.setattr(cad, "find_auto_threshold", lambda *_: (_ for _ in ()).throw(AssertionError("Auto-T reran")))
    second = cad.resolve_threshold(gray, threshold, state)
    assert state["solar_data"] is solar
    assert np.array_equal(first, second)


def test_apply_changed_setting_persists_exact_threshold_and_resolves(monkeypatch):
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"
    app.gray_image = np.full((9, 9), 200, np.uint8)
    app.threshold = Var(3)
    app.setting_variables = {"threshold": app.threshold}
    app.default_settings = cad.ImageSettings()
    app.image_state = {"image": {"settings": cad.ImageSettings(), "auto_threshold_result": None, "solar_data": None}}
    app.blocked_gui = nullcontext
    app.status = Var("")
    calls = []
    monkeypatch.setattr(cad, "resolve_threshold", lambda gray, threshold, state: calls.append(threshold) or (gray > threshold))
    app.apply_changed_setting("threshold", 17)
    assert app.threshold.get() == 17
    assert app.image_state["image"]["settings"].threshold == 17
    assert calls == [17]


def test_threshold_change_rebuilds_solardata_through_resolver():
    gray, threshold, _ = disk_gray()
    state = {"settings": cad.ImageSettings(threshold=threshold), "auto_threshold_result": None, "solar_data": None}
    cad.resolve_threshold(gray, threshold, state)
    old = state["solar_data"]
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "image"; app.gray_image = gray; app.threshold = Var(threshold)
    app.setting_variables = {"threshold": app.threshold}; app.default_settings = cad.ImageSettings()
    app.image_state = {"image": state}; app.blocked_gui = nullcontext; app.status = Var("")
    app.apply_changed_setting("threshold", threshold + 1)
    assert state["settings"].threshold == threshold + 1
    assert state["solar_data"] is not old
    assert state["solar_data"].threshold == threshold + 1


def test_heavy_actions_construct_only_genuinely_missing_solardata():
    for action in (
        cad.DetectorApp.preview_button_clicked,
        cad.DetectorApp.full_button_clicked,
    ):
        source = inspect.getsource(action)
        lookup = source.index('solar_data = state.get("solar_data")')
        missing = source.index("if solar_data is None:")
        resolve = source.index("resolve_threshold(")
        assert lookup < missing < resolve
        assert "solar_data.failure_reason is not None" in source
        assert "not solar_data.complete" in source


def test_source_has_one_lightweight_setting_application_path():
    source = Path(cad.__file__).read_text(encoding="utf-8")
    assert "def apply_changed_setting(self, setting_name, value)" in source
    assert "commit_setting_change" not in source
    assert "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS" not in source
    assert "self.threshold_preview" not in source


def test_load_runs_autot_before_setting_restoration_and_refresh():
    source = inspect.getsource(cad.DetectorApp.load_image_at)
    assert source.index("find_auto_threshold(") < source.index("for setting_name in self.setting_variables:")
    assert source.index("self.apply_changed_setting(setting_name, value)") < source.index("self.preview_button_clicked()")


def test_resolve_threshold_is_atomic_solardata_writer_and_does_not_write_settings():
    source = inspect.getsource(cad.resolve_threshold)
    assert 'image_state["solar_data"] = solar_data' in source
    assert "component_mask=compress_image(component)" in source
    assert "component_contour=compress_contour(contour)" in source
    assert "settings.threshold" not in source


def test_resize_redraw_reuses_each_canvas_own_retained_raster():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    c1 = SimpleNamespace(_unscaled_render_raster=np.ones((2, 3, 4), np.uint8))
    c2 = SimpleNamespace(_unscaled_render_raster=np.zeros((4, 5, 4), np.uint8))
    calls = []
    app.render_canvas_content = lambda c, r: calls.append((c, r.copy()))
    app._handle_canvas_resize(SimpleNamespace(widget=c1)); app._handle_canvas_resize(SimpleNamespace(widget=c2))
    assert calls[0][0] is c1 and np.array_equal(calls[0][1], c1._unscaled_render_raster)
    assert calls[1][0] is c2 and np.array_equal(calls[1][1], c2._unscaled_render_raster)
