from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "action_name",
    ("preview_button_clicked", "full_button_clicked"),
)
def test_heavy_actions_reject_stale_different_t_solardata(action_name):
    app = _app()
    app.image_state["img"]["solar_data"] = cad.SolarData(
        9, (0, 0), b"c", b"g", b"q"
    )
    with pytest.raises(ValueError, match="different threshold"):
        getattr(app, action_name)()


def test_gui_source_constructs_only_missing_solardata_before_downstream_placeholders():
    text = Path(cad.__file__).read_text(encoding="utf-8")
    preview = text.split("def preview_button_clicked(self):", 1)[1].split(
        "def full_button_clicked", 1
    )[0]
    assert preview.index("if solar_data is None:") < preview.index("resolve_threshold(")
    assert preview.index("resolve_threshold(") < preview.index(
        "# TODO: horizon finding consumes solar_data."
    )
    assert "solar_data.failure_reason is not None" in preview
    assert "not solar_data.complete" in preview
    assert "# TODO: ellipse finding" in preview
    assert "# TODO: center the full-color image" in preview


def _complete_solar_data(threshold=10):
    return cad.SolarData(threshold, (0, 0), b"c", b"g", b"q")


@pytest.mark.parametrize(
    "action_name, status_fragment",
    (
        ("preview_button_clicked", "downstream preview"),
        ("full_button_clicked", "downstream full-resolution"),
    ),
)
def test_heavy_actions_construct_missing_solardata_once(
    monkeypatch, action_name, status_fragment
):
    app = _app()
    solar = _complete_solar_data()
    calls = []

    def resolve(gray, threshold, state):
        calls.append(threshold)
        state["solar_data"] = solar
        return gray > threshold

    monkeypatch.setattr(cad, "resolve_threshold", resolve)
    getattr(app, action_name)()
    assert calls == [10]
    assert app.image_state["img"]["solar_data"] is solar
    assert status_fragment in app.status.get()


@pytest.mark.parametrize(
    "action_name",
    ("preview_button_clicked", "full_button_clicked"),
)
def test_heavy_actions_stop_on_newly_failed_solardata(monkeypatch, action_name):
    app = _app()
    calls = []

    def resolve(gray, threshold, state):
        calls.append(threshold)
        state["solar_data"] = cad.SolarData(
            threshold=threshold,
            failure_reason="no solar component",
        )
        raise cad.ThresholdResolutionError("no solar component")

    monkeypatch.setattr(cad, "resolve_threshold", resolve)
    getattr(app, action_name)()
    assert calls == [10]
    assert "stopped" in app.status.get()
    assert "no solar component" in app.status.get()


@pytest.mark.parametrize(
    "action_name",
    ("preview_button_clicked", "full_button_clicked"),
)
def test_heavy_actions_do_not_retry_cached_failed_solardata(
    monkeypatch, action_name
):
    app = _app()
    app.image_state["img"]["solar_data"] = cad.SolarData(
        threshold=10,
        failure_reason="known failure",
    )
    monkeypatch.setattr(
        cad,
        "resolve_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("cached failure retried")),
    )
    getattr(app, action_name)()
    assert "known failure" in app.status.get()


@pytest.mark.parametrize(
    "action_name",
    ("preview_button_clicked", "full_button_clicked"),
)
def test_heavy_actions_reject_partial_solardata(action_name):
    app = _app()
    app.image_state["img"]["solar_data"] = cad.SolarData(
        threshold=10,
        seed_point=(0, 0),
    )
    with pytest.raises(ValueError, match="incomplete SolarData"):
        getattr(app, action_name)()
