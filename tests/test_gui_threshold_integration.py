from contextlib import nullcontext
from pathlib import Path

import numpy as np

import circle_arc_detector as cad


class Var:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value


def _app():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.current_path = "img"
    app.gray_image = np.full((11, 13), 100, np.uint8)
    app.threshold = Var(10)
    app.min_radius = Var(1000)
    app.setting_variables = {"threshold": app.threshold, "min_radius": app.min_radius}
    app.default_settings = cad.ImageSettings(min_radius=1000)
    app.image_state = {"img": {"settings": cad.ImageSettings(threshold=10), "auto_threshold_result": None, "solar_data": None}}
    app.status = Var("")
    app.blocked_gui = nullcontext
    return app


def test_apply_changed_setting_keeps_sparse_nonthreshold_persistence(monkeypatch):
    app = _app()
    monkeypatch.setattr(cad, "resolve_threshold", lambda gray, threshold, state: gray > threshold)
    app.apply_changed_setting("min_radius", 1100)
    assert app.image_state["img"]["settings"].min_radius == 1100
    app.apply_changed_setting("min_radius", 1000)
    assert app.image_state["img"]["settings"].min_radius is None


def test_threshold_is_persisted_exactly_even_when_equal_to_gui_default(monkeypatch):
    app = _app()
    monkeypatch.setattr(cad, "resolve_threshold", lambda gray, threshold, state: gray > threshold)
    app.apply_changed_setting("threshold", 10)
    assert app.image_state["img"]["settings"].threshold == 10


def test_heavy_preview_and_full_resolution_do_not_resolve_threshold(monkeypatch):
    app = _app()
    app.image_state["img"]["solar_data"] = cad.SolarData(10, (0, 0), b"c", b"g", b"q")
    monkeypatch.setattr(cad, "resolve_threshold", lambda *_: (_ for _ in ()).throw(AssertionError("resolver called")))
    app.preview_button_clicked()
    assert "downstream preview" in app.status.get()
    app.full_button_clicked()
    assert "downstream full-resolution" in app.status.get()


def test_heavy_actions_require_same_t_solardata():
    app = _app()
    app.image_state["img"]["solar_data"] = cad.SolarData(9, (0, 0), b"c", b"g", b"q")
    app.preview_button_clicked()
    assert "requires current SolarData" in app.status.get()

def test_gui_source_contains_downstream_placeholders_after_solardata_precondition():
    text = Path(cad.__file__).read_text(encoding="utf-8")
    preview = text.split("def preview_button_clicked(self):", 1)[1].split("def full_button_clicked", 1)[0]
    assert "resolve_threshold(" not in preview
    assert "# TODO: horizon finding consumes solar_data." in preview
    assert "# TODO: ellipse finding" in preview
    assert "# TODO: center the full-color image" in preview
