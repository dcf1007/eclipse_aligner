from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

import circle_arc_detector as candidate


class Var:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


def disk_component(shape=(401, 421), center=(210, 200), radius=100, value=220):
    gray = np.zeros(shape, np.uint8)
    u8 = np.zeros(shape, np.uint8)
    cv2.circle(u8, center, radius, 255, -1)
    component = u8 != 0
    gray[component] = value
    return gray, component


def auto_result(threshold=100, seed=(210, 200)):
    return candidate.AutoThresholdResult(
        threshold=threshold,
        histogram_peak=220,
        histogram_left_edge=threshold,
        seed_threshold=threshold,
        coarse_threshold=threshold,
        roi_seed_threshold=threshold,
        full_seed_point=seed,
        used_guard=False,
        resolved=True,
    )


def test_solar_data_already_persists_full_component_mask():
    gray, component = disk_component()
    solar = candidate.build_solar_data(gray, 100, (210, 200), component)
    restored = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(restored, component)


def test_radius_bounds_use_contracted_component_and_expanded_guard_extrema():
    gray, component = disk_component(radius=100)
    solar = candidate.build_solar_data(gray, 100, (210, 200), component)
    bounds = candidate.derive_auto_radius_bounds(solar, gray.shape)

    margin = candidate.GUARD_DILATION_FRACTION * math.sqrt(gray.shape[0] * gray.shape[1])
    assert 79.0 < margin < 81.0
    assert bounds.threshold == 100
    # A perfect 100px disk contracted by ~80px should leave a small positive
    # minimum witness, while its stored guard should reach roughly 180px.
    assert bounds.min_radius is not None
    assert 15 <= bounds.min_radius <= 25
    assert bounds.max_radius is not None
    assert 175 <= bounds.max_radius <= 185
    assert bounds.contracted_fit_count >= 1
    assert bounds.expanded_fit_count >= 1


def test_empty_19_5_contraction_is_a_valid_no_minimum_witness():
    gray, component = disk_component(radius=25)
    solar = candidate.build_solar_data(gray, 100, (210, 200), component)
    bounds = candidate.derive_auto_radius_bounds(solar, gray.shape)
    assert bounds.min_radius is None
    assert bounds.contracted_fit_count == 0
    assert bounds.max_radius is not None


def test_selected_radii_are_reused_without_image_read(monkeypatch):
    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    path = "/does/not/exist.jpg"
    app.image_state = {
        path: {
            "settings": candidate.ImageSettings(min_radius=800, max_radius=1600),
            "auto_threshold_result": auto_result(),
            "solar_data": None,
            "auto_radius_bounds": None,
        }
    }

    def fail(*_args, **_kwargs):
        raise AssertionError("explicit radii should avoid image I/O")
    monkeypatch.setattr(cv2, "imread", fail)

    assert app._radius_auto_witness_for_path(path) == (800, 1600, "selected")


def test_cached_radius_bounds_are_reused_without_refitting(monkeypatch):
    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    path = "/does/not/exist.jpg"
    solar = candidate.SolarData(
        threshold=100,
        seed_point=(1, 1),
        component_mask=b"",
        roi_6_5_mask=b"",
        guard_19_5_mask=b"",
        component_contour=np.empty((0, 2), np.uint16),
    )
    app.image_state = {
        path: {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto_result(),
            "solar_data": solar,
            "auto_radius_bounds": candidate.AutoRadiusBounds(100, 700, 1700, 4, 4),
        }
    }

    def fail(*_args, **_kwargs):
        raise AssertionError("current cached radius bounds should avoid image I/O")
    monkeypatch.setattr(cv2, "imread", fail)

    assert app._radius_auto_witness_for_path(path) == (700, 1700, "cached")


def test_radius_auto_updates_batch_defaults_but_preserves_explicit_image_choices():
    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    selected_path = "/selected.jpg"
    default_path = "/default.jpg"
    solar = candidate.SolarData(
        threshold=100,
        seed_point=(1, 1),
        component_mask=b"",
        roi_6_5_mask=b"",
        guard_19_5_mask=b"",
        component_contour=np.empty((0, 2), np.uint16),
    )
    app.image_paths = [selected_path, default_path]
    app.current_path = default_path
    app.image_state = {
        selected_path: {
            "settings": candidate.ImageSettings(min_radius=800, max_radius=1600),
            "auto_threshold_result": auto_result(),
            "solar_data": None,
            "auto_radius_bounds": None,
        },
        default_path: {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto_result(),
            "solar_data": solar,
            "auto_radius_bounds": candidate.AutoRadiusBounds(100, 700, 1700, 4, 4),
        },
    }
    app.default_settings = candidate.ImageSettings(min_radius=1000, max_radius=1500)
    app.min_radius = Var(1000)
    app.max_radius = Var(1500)
    app.status = Var("")
    app.setting_sliders = {}

    app.auto_select_radius()

    assert app.default_settings.min_radius == 700
    assert app.default_settings.max_radius == 1700
    assert app.min_radius.get() == 700
    assert app.max_radius.get() == 1700
    chosen = app.image_state[selected_path]["settings"]
    assert chosen.min_radius == 800
    assert chosen.max_radius == 1600


def test_radius_commit_stays_explicit_when_equal_to_batch_default():
    app = candidate.DetectorApp.__new__(candidate.DetectorApp)
    path = "/image.jpg"
    app.current_path = path
    app.min_radius = Var(1000)
    app.setting_variables = {"min_radius": app.min_radius}
    app.default_settings = candidate.ImageSettings(min_radius=1000)
    app.image_state = {
        path: {
            "settings": candidate.ImageSettings(),
            "auto_threshold_result": auto_result(),
        }
    }
    app._refresh_threshold_image = lambda: None

    app._commit_setting_change("min_radius")
    assert app.image_state[path]["settings"].min_radius == 1000


def test_solar_data_rebuild_invalidates_cached_radius_bounds():
    gray, component = disk_component(radius=100)
    old = candidate.build_solar_data(gray, 90, (210, 200), component)
    state = {
        "settings": candidate.ImageSettings(),
        "auto_threshold_result": auto_result(threshold=90),
        "solar_data": old,
        "auto_radius_bounds": candidate.AutoRadiusBounds(90, 20, 180, 4, 4),
    }
    app = candidate.DetectorApp.__new__(candidate.DetectorApp)

    new, rebuilt = app._ensure_solar_data_for_state(gray, state, 100)
    assert rebuilt is True
    assert new.threshold == 100
    assert state["auto_radius_bounds"] is None


def test_threshold_branch_does_not_own_horizon_proposal_state():
    source = (Path(__file__).resolve().parents[1] / "circle_arc_detector.py").read_text()
    assert '"horizon_proposal"' not in source
