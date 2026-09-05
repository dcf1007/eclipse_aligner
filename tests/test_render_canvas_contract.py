import numpy as np
import pytest

import circle_arc_detector as cad

class Root:
    def update_idletasks(self): pass
class Canvas:
    def winfo_width(self): return 20
    def winfo_height(self): return 20
    def delete(self,*a): pass
    def create_image(self,*a,**k): pass


def _app(monkeypatch):
    app=cad.DetectorApp.__new__(cad.DetectorApp); app.root=Root()
    class Photo:
        def __init__(self,*a,**k): pass
    monkeypatch.setattr(cad.tk,'PhotoImage',Photo)
    return app


def test_renderer_accepts_bool_uint8_gray_and_uint8_bgra(monkeypatch):
    app=_app(monkeypatch)
    for content in (np.zeros((3,4),bool),np.zeros((3,4),np.uint8),np.zeros((3,4,4),np.uint8)):
        canvas=Canvas(); app.render_canvas_content(canvas,content)
        assert canvas._unscaled_render_raster.dtype==np.uint8
        assert canvas._unscaled_render_raster.ndim==3 and canvas._unscaled_render_raster.shape[2]==4


def test_renderer_rejects_uint16_float_and_bgr(monkeypatch):
    app=_app(monkeypatch)
    for content in (np.zeros((3,4),np.uint16),np.zeros((3,4),np.float32),np.zeros((3,4,3),np.uint8)):
        with pytest.raises(ValueError): app.render_canvas_content(Canvas(),content)
