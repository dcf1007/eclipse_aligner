from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "circle_arc_detector.py"

import circle_arc_detector as candidate


def disk_component(shape=(161, 181), center=(90, 80), radius=24, value=220):
    gray = np.zeros(shape, np.uint8)
    component = np.zeros(shape, np.uint8)
    cv2.circle(component, center, radius, 255, -1)
    gray[component != 0] = value
    return gray, component != 0


def test_auto_threshold_stage_source_is_unchanged():
    import hashlib

    text = SOURCE_PATH.read_text()
    start = "# ---------------------------------------------------------------------------\n# Grayscale automatic threshold finder\n"
    stage = text.split(start, 1)[1].split(
        "\n\n# ---------------------------------------------------------------------------\n# Post-threshold full-resolution solar data\n",
        1,
    )[0]
    assert hashlib.sha256(stage.encode()).hexdigest() == (
        "0e915214de161831e7be5dec461b60b734c19d1e439c8ade24c7bdb84f29207a"
    )


def test_auto_threshold_has_no_solar_data_dependency():
    source = inspect.getsource(candidate.auto_threshold)
    assert "SolarData" not in source
    assert "build_solar_data" not in source
    assert "establish_solar_component_at_threshold" not in source


def test_packbits_zlib_round_trip_non_multiple_of_eight():
    rng = np.random.default_rng(7)
    mask = rng.random((17, 19)) > 0.73
    payload = candidate.compress_full_mask(mask)
    restored = candidate.decompress_full_mask(payload, mask.shape)
    assert restored.dtype == bool
    assert np.array_equal(restored, mask)


def test_mask_decoder_rejects_wrong_shape():
    mask = np.zeros((7, 11), bool)
    payload = candidate.compress_full_mask(mask)
    with pytest.raises(ValueError, match="packed bytes"):
        candidate.decompress_full_mask(payload, (8, 11))


def test_seeded_component_is_exact_connected_component():
    gray, component = disk_component()
    seed = (90, 80)
    found = candidate.solar_component_from_seed_at_threshold(gray, 100, seed)
    assert np.array_equal(found, component)


def test_seeded_component_rejects_dark_seed():
    gray, _ = disk_component()
    with pytest.raises(candidate.ThresholdResolutionError, match="not light"):
        candidate.solar_component_from_seed_at_threshold(gray, 250, (90, 80))


def test_fixed_t_establishment_without_seed_uses_existing_identity_machinery():
    gray, component = disk_component(shape=(241, 301), center=(150, 120), radius=31)
    seed, found = candidate.establish_solar_component_at_threshold(gray, 100)
    x, y = seed
    assert found[y, x]
    assert gray[y, x] > 100
    assert np.array_equal(found, component)


def test_fixed_t_establishment_falls_back_when_preferred_seed_is_dark():
    gray, component = disk_component(shape=(241, 301), center=(150, 120), radius=31)
    seed, found = candidate.establish_solar_component_at_threshold(
        gray, 100, preferred_seed=(5, 5)
    )
    assert seed != (5, 5)
    assert np.array_equal(found, component)


def test_fixed_t_establishment_fails_when_only_bright_component_touches_border():
    gray = np.zeros((120, 140), np.uint8)
    gray[:, 0:20] = 220
    with pytest.raises(candidate.ThresholdResolutionError, match="No enclosed solar component"):
        candidate.establish_solar_component_at_threshold(gray, 100)


def test_build_solar_data_masks_match_existing_observation_geometry():
    gray, component = disk_component(shape=(201, 241), center=(120, 100), radius=26)
    seed = (120, 100)
    solar = candidate.build_solar_data(gray, 100, seed, component)

    assert solar.threshold == 100
    assert solar.seed_point == seed
    restored_component = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(restored_component, component)

    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5
    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        restored = candidate.decompress_full_mask(payload, gray.shape)
        region = candidate._build_observation_region(
            gray, component, fraction * image_scale, seed
        )
        expected = np.zeros(gray.shape, bool)
        x0, y0, x1, y1 = region.bbox
        expected[y0:y1, x0:x1] = region.allowed_u8 != 0
        assert np.array_equal(restored, expected)


def test_contour_is_raw_ordered_external_uint16():
    gray, component = disk_component(shape=(201, 241), center=(120, 100), radius=26)
    solar = candidate.build_solar_data(gray, 100, (120, 100), component)
    contour = solar.component_contour
    assert contour.dtype == np.uint16
    assert contour.ndim == 2 and contour.shape[1] == 2
    assert len(contour) > 4

    # CHAIN_APPROX_NONE traces neighboring boundary pixels in order, allowing
    # diagonal steps but no multi-pixel jumps.
    closed = np.vstack([contour.astype(np.int32), contour[0].astype(np.int32)])
    steps = np.abs(np.diff(closed, axis=0))
    assert np.all(np.max(steps, axis=1) <= 1)

    boundary = np.zeros(component.shape, np.uint8)
    cv2.drawContours(boundary, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, 255, 1)
    assert np.all(component[contour[:, 1], contour[:, 0]])


def test_solar_data_seed_must_match_component_and_threshold():
    gray, component = disk_component()
    with pytest.raises(candidate.ThresholdResolutionError, match="outside the solar component"):
        candidate.build_solar_data(gray, 100, (5, 5), component)
    with pytest.raises(candidate.ThresholdResolutionError, match="not light"):
        candidate.build_solar_data(gray, 230, (90, 80), component)


def test_gui_control_commits_do_not_trigger_solar_processing():
    source = inspect.getsource(candidate.DetectorApp._commit_setting_change)
    assert "_ensure_solar_data" not in source
    assert "build_solar_data" not in source
    assert "establish_solar_component" not in source
    assert "_refresh_threshold_image" in source


def test_refresh_preview_is_the_explicit_solar_processing_boundary():
    source = inspect.getsource(candidate.DetectorApp.refresh_preview)
    assert "_ensure_solar_data_for_current_threshold" in source
    load_source = inspect.getsource(candidate.DetectorApp.load_image_at)
    assert "self.refresh_preview()" not in load_source
    assert "self._refresh_display_images()" in load_source


def test_auto_result_is_created_before_initial_solar_data_calls():
    source = inspect.getsource(candidate.DetectorApp.load_image_at)
    auto_pos = source.index("result = auto_threshold(self.gray_image)")
    component_pos = source.index("component = solar_component_from_seed_at_threshold")
    solar_pos = source.index("solar_data = build_solar_data")
    assert auto_pos < component_pos < solar_pos


def test_unresolved_auto_message_waits_for_explicit_refresh():
    source = inspect.getsource(candidate.DetectorApp.load_image_at)
    assert "Adjust the " in source
    assert "threshold if needed, then click Refresh Preview to continue." in source


def test_current_solar_data_is_reused_without_rebuild(monkeypatch):
    gray, component = disk_component()
    solar = candidate.build_solar_data(gray, 100, (90, 80), component)
    auto = candidate.AutoThresholdResult(
        threshold=100,
        histogram_peak=200,
        histogram_left_edge=100,
        seed_threshold=100,
        coarse_threshold=100,
        roi_seed_threshold=100,
        full_seed_point=(90, 80),
        used_guard=False,
        resolved=True,
    )

    class Var:
        def get(self):
            return 100

    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    app.gray_image = gray
    app.current_path = "synthetic"
    app.threshold = Var()
    app.image_state = {
        "synthetic": {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto,
            "solar_data": solar,
        }
    }

    def fail(*_args, **_kwargs):
        raise AssertionError("current SolarData should have been reused")

    monkeypatch.setattr(candidate, "establish_solar_component_at_threshold", fail)
    returned, rebuilt = app._ensure_solar_data_for_current_threshold()
    assert returned is solar
    assert rebuilt is False


def test_stale_solar_data_rebuild_prefers_existing_seed():
    gray, component = disk_component()
    old = candidate.build_solar_data(gray, 90, (90, 80), component)
    auto = candidate.AutoThresholdResult(
        threshold=90,
        histogram_peak=200,
        histogram_left_edge=90,
        seed_threshold=90,
        coarse_threshold=90,
        roi_seed_threshold=90,
        full_seed_point=(90, 80),
        used_guard=False,
        resolved=True,
    )

    class Var:
        def get(self):
            return 100

    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    app.gray_image = gray
    app.current_path = "synthetic"
    app.threshold = Var()
    app.image_state = {
        "synthetic": {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto,
            "solar_data": old,
        }
    }

    new, rebuilt = app._ensure_solar_data_for_current_threshold()
    assert rebuilt is True
    assert new.threshold == 100
    assert new.seed_point == old.seed_point
    assert app.image_state["synthetic"]["solar_data"] is new


def test_unresolved_auto_can_bootstrap_only_when_ensure_is_called():
    gray, component = disk_component()
    auto = candidate.AutoThresholdResult(
        threshold=100,
        histogram_peak=200,
        histogram_left_edge=100,
        seed_threshold=100,
        coarse_threshold=None,
        roi_seed_threshold=None,
        full_seed_point=None,
        used_guard=False,
        resolved=False,
        reason="synthetic unresolved",
    )

    class Var:
        def get(self):
            return 100

    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    app.gray_image = gray
    app.current_path = "synthetic"
    app.threshold = Var()
    app.image_state = {
        "synthetic": {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto,
            "solar_data": None,
        }
    }

    assert app.image_state["synthetic"]["solar_data"] is None
    solar, rebuilt = app._ensure_solar_data_for_current_threshold()
    assert rebuilt is True
    assert solar.threshold == 100
    assert np.array_equal(
        candidate.decompress_full_mask(solar.component_mask, gray.shape), component
    )
