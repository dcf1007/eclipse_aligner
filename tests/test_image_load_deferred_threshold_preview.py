import inspect

import circle_arc_detector as appmod


def test_image_load_uses_same_deferred_two_stage_threshold_pipeline():
    source = inspect.getsource(appmod.DetectorApp.load_image_at)

    assert 'self.refresh_preview(changed_setting="image load")' in source
    assert "build_solar_data_at_threshold" not in source
    assert "self.gray_image > selected_threshold" not in source
