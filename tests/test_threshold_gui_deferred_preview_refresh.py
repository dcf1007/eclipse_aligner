"""Behavioral checks for deferred slider-interaction commits on the threshold branch."""
from types import SimpleNamespace

import circle_arc_detector as appmod


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
    def __init__(self, value, setting_name="threshold"):
        self.value = value
        self._setting_name = setting_name

    def get(self):
        return self.value


def make_app():
    app = object.__new__(appmod.DetectorApp)
    app.root = FakeRoot()
    app.slider_keyboard_commit_job = None
    app.slider_keyboard_widget = None
    app.slider_keyboard_start_value = None
    app.commits = []
    app.commit_setting_change = lambda name, value: app.commits.append((name, value))
    return app


def test_mouse_release_commits_only_the_changed_slider_setting_and_value():
    app = make_app()
    slider = FakeSlider(8, "threshold")
    event = SimpleNamespace(widget=slider)
    app._begin_slider_mouse_change(event)
    slider.value = 12
    app._finish_slider_mouse_change(event)
    assert app.commits == [("threshold", 12)]


def test_mouse_release_without_change_does_not_commit():
    app = make_app()
    slider = FakeSlider(8, "min_radius")
    event = SimpleNamespace(widget=slider)
    app._begin_slider_mouse_change(event)
    app._finish_slider_mouse_change(event)
    assert app.commits == []


def test_keyboard_repeat_release_is_cancelled_until_final_release_then_commits_value():
    app = make_app()
    slider = FakeSlider(8, "threshold")
    event = SimpleNamespace(widget=slider)
    app._begin_slider_keyboard_change(event)
    slider.value = 9
    app._schedule_slider_keyboard_commit(event)
    first_job = app.slider_keyboard_commit_job
    app._begin_slider_keyboard_change(event)
    assert first_job in app.root.cancelled
    slider.value = 12
    app._schedule_slider_keyboard_commit(event)
    final_job = app.slider_keyboard_commit_job
    delay, callback = app.root.jobs[final_job]
    assert delay == appmod.SLIDER_KEY_RELEASE_SETTLE_MS
    callback()
    assert app.commits == [("threshold", 12)]
