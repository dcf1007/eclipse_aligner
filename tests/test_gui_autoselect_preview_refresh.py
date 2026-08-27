"""Behavioral regression check for threshold Auto-select event routing."""

import circle_arc_detector as gui


class FakeStatus:
    def set(self, _value):
        pass


def test_threshold_autoselect_commits_setting_without_running_full_preview():
    app = object.__new__(gui.DetectorApp)
    app.status = FakeStatus()
    calls = []
    app._commit_setting_change = lambda: calls.append("commit")
    app.refresh_preview = lambda: calls.append("full")

    app.auto_select_threshold()

    assert calls == ["commit"]
