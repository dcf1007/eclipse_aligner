import inspect

import circle_arc_detector as appmod


def test_image_load_routes_initialized_and_auto_initialized_t_through_common_refresh_path():
    source = inspect.getsource(appmod.DetectorApp.load_image_at)
    assert "if settings.threshold is None:" in source
    assert "threshold = find_auto_threshold(self.gray_image, state)" in source
    assert 'self.commit_setting_change("threshold", threshold)' in source
    assert 'self.refresh_preview(changed_setting="image load")' in source
    assert "resolve_threshold(" not in source
    assert "self.gray_image >" not in source
