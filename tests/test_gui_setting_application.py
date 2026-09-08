from types import SimpleNamespace

import circle_arc_detector as cad


def test_mouse_slider_applies_changed_setting_once_only_after_value_changes():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.calls = []
    app.apply_changed_setting = lambda name, value: app.calls.append((name, value))
    widget = SimpleNamespace(_preview_mouse_start_value=10, _setting_name="threshold", get=lambda: 11)
    app._finish_slider_mouse_change(SimpleNamespace(widget=widget))
    assert app.calls == [("threshold", 11)]


def test_mouse_slider_noop_does_not_apply_setting():
    app = cad.DetectorApp.__new__(cad.DetectorApp)
    app.calls = []
    app.apply_changed_setting = lambda name, value: app.calls.append((name, value))
    widget = SimpleNamespace(_preview_mouse_start_value=10, _setting_name="threshold", get=lambda: 10)
    app._finish_slider_mouse_change(SimpleNamespace(widget=widget))
    assert app.calls == []
