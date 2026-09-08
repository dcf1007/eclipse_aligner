from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

import circle_arc_detector as cad


class Var:
    def __init__(self, value):
        self.value = value
    def set(self, value):
        self.value = value
    def get(self):
        return self.value


def _app():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "x"
    app.gray_image = np.arange(81, dtype=np.uint8).reshape(9, 9)
    app.threshold = Var(10)
    app.min_radius = Var(1000)
    app.setting_variables = {"threshold": app.threshold, "min_radius": app.min_radius}
    app.default_settings = cad.ImageSettings(min_radius=1000)
    app.image_state = {
        "x": {
            "settings": cad.ImageSettings(threshold=10),
            "auto_threshold_result": None,
            "solar_data": None,
        }
    }
    app.threshold_canvas = object()
    app.status = Var("")
    app.processing_ui = nullcontext
    return app


def test_apply_changed_setting_persists_then_renders_grayscale_and_refined_component(monkeypatch):
    app = _app()
    refined = np.zeros_like(app.gray_image, bool)
    refined[2:7, 2:7] = True
    calls = []

    def resolve(gray, threshold, state):
        calls.append((gray.copy(), threshold, state))
        state["solar_data"] = cad.SolarData(
            threshold,
            (4, 4),
            cad.compress_image(refined),
            cad.compress_image(np.ones_like(refined)),
            cad.compress_contour(cad.find_external_contour(refined)),
        )
        return refined

    monkeypatch.setattr(cad, "resolve_threshold", resolve)
    rendered = []
    app.render_canvas_content = lambda canvas, content: rendered.append(
        (canvas, np.asarray(content).copy())
    )

    app.apply_changed_setting("threshold", 11)

    assert app.image_state["x"]["settings"].threshold == 11
    assert calls[0][1] == 11
    assert np.array_equal(rendered[0][1], app.gray_image)
    assert np.array_equal(rendered[1][1], refined)
    assert app.image_state["x"]["solar_data"].threshold == 11


def test_apply_changed_setting_failure_leaves_grayscale_displayed(monkeypatch):
    app = _app()
    rendered = []
    app.render_canvas_content = lambda canvas, content: rendered.append(np.asarray(content).copy())
    monkeypatch.setattr(
        cad,
        "resolve_threshold",
        lambda *_: (_ for _ in ()).throw(cad.ThresholdResolutionError("no component")),
    )
    app.apply_changed_setting("threshold", 12)
    assert len(rendered) == 1
    assert np.array_equal(rendered[0], app.gray_image)
    assert "Grayscale remains displayed" in app.status.get()


def test_nonthreshold_setting_uses_same_lightweight_solardata_path(monkeypatch):
    app = _app()
    calls = []
    refined = np.ones_like(app.gray_image, bool)
    monkeypatch.setattr(cad, "resolve_threshold", lambda gray, threshold, state: calls.append(threshold) or refined)
    app.render_canvas_content = lambda *_: None
    app.apply_changed_setting("min_radius", 1100)
    assert app.image_state["x"]["settings"].min_radius == 1100
    assert calls == [10]


def test_canvas_resize_renders_only_event_widget_retained_raster():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    canvas = SimpleNamespace(_unscaled_render_raster=np.ones((2, 3, 4), np.uint8))
    calls = []
    app.render_canvas_content = lambda c, r: calls.append((c, r.copy()))
    app._handle_canvas_resize(SimpleNamespace(widget=canvas))
    assert len(calls) == 1 and calls[0][0] is canvas
    assert np.array_equal(calls[0][1], canvas._unscaled_render_raster)


def test_auto_select_button_calls_parent_then_apply_changed_setting(monkeypatch):
    app = _app()
    calls = []
    result = cad.AutoThresholdResult(
        full_res_refined_threshold=17,
        failure_reason=None,
    )

    def auto(gray, state):
        state["auto_threshold_result"] = result
        calls.append(("auto", gray))
        return 17

    monkeypatch.setattr(cad, "find_auto_threshold", auto)
    app.apply_changed_setting = lambda name, value: calls.append(("apply", name, value))
    app.auto_select_threshold_button()
    assert calls[0][0] == "auto"
    assert calls[1] == ("apply", "threshold", 17)


def test_auto_select_button_failure_does_not_apply_setting(monkeypatch):
    app = _app()
    result = cad.AutoThresholdResult(failure_reason="failed")
    monkeypatch.setattr(
        cad,
        "find_auto_threshold",
        lambda gray, state: state.__setitem__("auto_threshold_result", result),
    )
    app.apply_changed_setting = lambda *_: (_ for _ in ()).throw(AssertionError("setting applied"))
    app.auto_select_threshold_button()
    assert "failed" in app.status.get()
