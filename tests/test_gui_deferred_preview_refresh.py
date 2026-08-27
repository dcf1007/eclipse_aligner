"""Behavioral regression checks for completed slider interactions."""

from types import SimpleNamespace

import circle_arc_detector as gui


class FakeRoot:
    def __init__(self):
        self.jobs = {}
        self.cancelled = []
        self.next_job = 1

    def after(self, delay, callback):
        job = self.next_job
        self.next_job += 1
        self.jobs[job] = (delay, callback)
        return job

    def after_cancel(self, job):
        self.cancelled.append(job)
        self.jobs.pop(job, None)


class FakeSlider:
    def __init__(self, value):
        self.value = value
        self.focused = False

    def get(self):
        return self.value

    def focus_set(self):
        self.focused = True


def make_app():
    app = object.__new__(gui.DetectorApp)
    app.root = FakeRoot()
    app.slider_keyboard_commit_job = None
    app.slider_keyboard_widget = None
    app.slider_keyboard_start_value = None
    app.commits = 0
    app._commit_setting_change = lambda: setattr(app, "commits", app.commits + 1)
    return app


def test_mouse_slider_commits_once_after_changed_release():
    app = make_app()
    slider = FakeSlider(8)
    event = SimpleNamespace(widget=slider)

    app._begin_slider_mouse_change(event)
    slider.value = 12
    app._finish_slider_mouse_change(event)

    assert app.commits == 1


def test_mouse_slider_does_not_commit_when_value_is_unchanged():
    app = make_app()
    slider = FakeSlider(8)
    event = SimpleNamespace(widget=slider)

    app._begin_slider_mouse_change(event)
    app._finish_slider_mouse_change(event)

    assert app.commits == 0


def test_keyboard_repeat_releases_are_cancelled_and_final_release_commits_once():
    app = make_app()
    slider = FakeSlider(8)
    event = SimpleNamespace(widget=slider)

    app._begin_slider_keyboard_change(event)
    slider.value = 9
    app._schedule_slider_keyboard_commit(event)
    first_job = app.slider_keyboard_commit_job

    app._begin_slider_keyboard_change(event)
    assert first_job in app.root.cancelled
    assert first_job not in app.root.jobs

    slider.value = 12
    app._schedule_slider_keyboard_commit(event)
    final_job = app.slider_keyboard_commit_job
    delay, callback = app.root.jobs[final_job]

    assert delay == gui.SLIDER_KEY_RELEASE_SETTLE_MS
    assert app.commits == 0

    callback()

    assert app.commits == 1
    assert app.slider_keyboard_commit_job is None
    assert app.slider_keyboard_widget is None
    assert app.slider_keyboard_start_value is None


def test_keyboard_interaction_does_not_commit_if_final_value_matches_start():
    app = make_app()
    slider = FakeSlider(8)
    event = SimpleNamespace(widget=slider)

    app._begin_slider_keyboard_change(event)
    slider.value = 9
    app._schedule_slider_keyboard_commit(event)
    app._begin_slider_keyboard_change(event)
    slider.value = 8
    app._schedule_slider_keyboard_commit(event)
    _, callback = app.root.jobs[app.slider_keyboard_commit_job]
    callback()

    assert app.commits == 0
