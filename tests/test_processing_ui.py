import argparse
import tkinter as tk
from contextlib import contextmanager

import numpy as np

import circle_arc_detector as cad


def build_app():
    root = tk.Tk()
    args = argparse.Namespace(
        threshold=8,
        min_radius=1000.0,
        max_radius=1500.0,
        max_error=0.08,
        min_coverage=0.08,
    )
    app = cad.DetectorApp(root, [], args)
    root.update()
    return root, app


def descendants(root):
    pending = [root]
    while pending:
        parent = pending.pop()
        for child in parent.winfo_children():
            pending.append(child)
            yield child


def interactive_controls(root):
    control_types = (tk.Button, tk.Scale, tk.Checkbutton, tk.Radiobutton)
    return [widget for widget in descendants(root) if isinstance(widget, control_types)]


def test_processing_ui_disables_controls_and_restores_prior_states():
    root, app = build_app()
    try:
        controls = interactive_controls(root)
        prior = [(widget, str(widget.cget("state"))) for widget in controls]
        assert any(state == str(tk.NORMAL) for _widget, state in prior)
        assert any(state == str(tk.DISABLED) for _widget, state in prior)

        with app.processing_ui():
            assert all(str(widget.cget("state")) == str(tk.DISABLED) for widget in controls)

        assert [(widget, str(widget.cget("state"))) for widget, _state in prior] == prior
    finally:
        root.destroy()


def test_queued_scale_click_is_discarded_before_restore():
    root, app = build_app()
    try:
        scale = next(widget for widget in descendants(root) if isinstance(widget, tk.Scale))
        scale.set(20)
        root.update()
        before = scale.get()
        width = max(scale.winfo_width(), 100)
        height = max(scale.winfo_height(), 20)

        with app.processing_ui():
            assert str(scale.cget("state")) == str(tk.DISABLED)
            scale.event_generate("<ButtonPress-1>", x=width - 2, y=height // 2, when="tail")
            scale.event_generate("<B1-Motion>", x=width - 2, y=height // 2, when="tail")
            scale.event_generate("<ButtonRelease-1>", x=width - 2, y=height // 2, when="tail")
            assert scale.get() == before

        root.update_idletasks()
        assert scale.get() == before
        assert str(scale.cget("state")) == str(tk.NORMAL)
    finally:
        root.destroy()


def test_disabled_button_click_does_not_replay_after_restore():
    root, app = build_app()
    try:
        fired = []
        button = app.preview_button
        button.config(command=lambda: fired.append(True), state=tk.NORMAL)
        root.update()

        with app.processing_ui():
            assert str(button.cget("state")) == str(tk.DISABLED)
            width = max(button.winfo_width(), 10)
            height = max(button.winfo_height(), 10)
            button.event_generate("<ButtonPress-1>", x=width // 2, y=height // 2, when="tail")
            button.event_generate("<ButtonRelease-1>", x=width // 2, y=height // 2, when="tail")
            assert fired == []

        root.update_idletasks()
        assert fired == []
        assert str(button.cget("state")) == str(tk.NORMAL)
    finally:
        root.destroy()


def test_auto_threshold_observer_runs_after_stage_a_before_stage_b(monkeypatch):
    gray = np.zeros((31, 31), np.uint8)
    state = {
        "settings": cad.ImageSettings(),
        "auto_threshold_result": None,
        "solar_data": None,
    }
    component = np.zeros(gray.shape, bool)
    component[8:23, 8:23] = True
    order = []

    monkeypatch.setattr(cad, "find_histogram_start_threshold", lambda _gray: 10)
    monkeypatch.setattr(
        cad,
        "find_work_res_solar_component",
        lambda _gray, _start, _kernel: (7, component),
    )
    monkeypatch.setattr(cad, "brightest_supported_component_point", lambda *_args: (15, 15))
    monkeypatch.setattr(cad, "dilate_component_mask", lambda mask, _margin: np.ones(mask.shape, bool))

    def coarse(*_args):
        order.append("stage_a")
        return 6, component.copy()

    monkeypatch.setattr(cad, "find_separation_threshold", coarse)
    payload = cad.compress_full_mask(component)

    def fine(*_args):
        order.append("stage_b")
        return 8, payload

    monkeypatch.setattr(cad, "refine_threshold", fine)

    def observer(threshold, observed_component):
        assert threshold == 6
        assert np.array_equal(observed_component, component)
        order.append("observer")

    assert cad.find_auto_threshold(gray, state, observer) == 8
    assert order == ["stage_a", "observer", "stage_b"]
