from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np

import circle_arc_detector as cad

class Var:
    def __init__(self,v): self.v=v
    def set(self,v): self.v=v
    def get(self): return self.v


def _app():
    app=cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path='x'
    app.threshold=Var(10)
    app.setting_variables={'threshold':app.threshold}
    app.default_settings=cad.ImageSettings()
    app.image_state={'x':{'settings':cad.ImageSettings(threshold=10),'auto_threshold_result':None,'solar_data':None}}
    app.threshold_canvas=object()
    return app


def test_commit_threshold_is_state_only_and_clears_canvas():
    app=_app(); rendered=[]
    app.render_canvas_content=lambda canvas,content: rendered.append((canvas,content))
    app.commit_setting_change('threshold',11)
    assert app.image_state['x']['settings'].threshold==11
    assert rendered == [(app.threshold_canvas,None)]


def test_commit_threshold_leaves_solardata_lifecycle_to_resolver():
    app = _app()
    solar = cad.SolarData(
        10,
        (0, 0),
        b"component",
        b"guard",
        cad.compress_contour(np.array([[0, 0]], np.int32)),
    )
    app.image_state["x"]["solar_data"] = solar
    app.render_canvas_content = lambda *_: None
    app.commit_setting_change("threshold", 11)
    assert app.image_state["x"]["solar_data"] is solar


def test_canvas_resize_renders_only_event_widget_retained_raster():
    app=cad.DetectorApp.__new__(cad.DetectorApp)
    canvas=SimpleNamespace(_unscaled_render_raster=np.ones((2,3,4),np.uint8))
    calls=[]; app.render_canvas_content=lambda c,r:calls.append((c,r.copy()))
    app._handle_canvas_resize(SimpleNamespace(widget=canvas))
    assert len(calls)==1 and calls[0][0] is canvas
    assert np.array_equal(calls[0][1],canvas._unscaled_render_raster)


def test_auto_select_calls_complete_autot_parent_and_reads_failure_state(monkeypatch):
    app = _app()
    app.gray_image = np.zeros((9, 9), np.uint8)
    app.status = Var("")
    app.processing_ui = nullcontext
    del app.threshold_canvas
    calls = []

    def complete_autot(gray, state):
        result = cad.AutoThresholdResult(
            failure_reason="work-resolution separation: no component"
        )
        state["auto_threshold_result"] = result
        calls.append(result)
        return None

    monkeypatch.setattr(cad, "find_auto_threshold", complete_autot)
    app.auto_select_threshold()

    result = app.image_state["x"]["auto_threshold_result"]
    assert calls == [result]
    assert "work-resolution separation: no component" in app.status.get()
