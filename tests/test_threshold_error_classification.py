import math

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_work_res_tracked_component_disappearance_is_invariant_error(monkeypatch):
    gray = np.full((11, 11), 200, np.uint8)
    component = np.zeros_like(gray, bool)
    component[3:8, 3:8] = True
    monkeypatch.setattr(cad, "largest_enclosed_bright_component", lambda *_: component)
    monkeypatch.setattr(cad, "brightest_supported_component_point", lambda *_: (5, 5))
    monkeypatch.setattr(cad, "extract_component", lambda *_: None)
    with pytest.raises(ValueError, match="disappeared while lowering T"):
        cad.find_work_res_separation_threshold(
            gray,
            2,
            cad.generate_kernel((5, 5)),
        )


def test_full_res_seed_disappearance_after_guard_clipping_is_invariant_error(monkeypatch):
    gray = np.full((9, 9), 200, np.uint8)
    guard = np.ones_like(gray, bool)
    binary = np.ones_like(gray, dtype=bool)
    monkeypatch.setattr(cad, "morphological_cleanup", lambda *_: binary.copy())
    monkeypatch.setattr(cad, "extract_component", lambda *_: None)
    with pytest.raises(ValueError, match="after clipping"):
        cad.find_full_res_separation_threshold(gray, 100, (4, 4), guard)


def test_full_res_tracked_component_disappearance_while_lowering_is_invariant_error(
    monkeypatch,
):
    gray = np.full((9, 9), 200, np.uint8)
    guard = np.ones_like(gray, bool)
    component = np.zeros_like(gray, bool)
    component[3:6, 3:6] = True
    binary = np.ones_like(gray, dtype=bool)
    components = iter((component, None))
    monkeypatch.setattr(cad, "morphological_cleanup", lambda *_: binary.copy())
    monkeypatch.setattr(cad, "extract_component", lambda *_: next(components))
    with pytest.raises(ValueError, match="disappeared while lowering T"):
        cad.find_full_res_separation_threshold(gray, 100, (4, 4), guard)


def test_full_res_seed_disappearance_after_surviving_cleanup_is_invariant_error(
    monkeypatch,
):
    gray = np.full((9, 9), 200, np.uint8)
    guard = np.ones_like(gray, bool)
    component = guard.copy()
    binary = np.ones_like(gray, dtype=bool)
    components = iter((component, None))
    monkeypatch.setattr(cad, "morphological_cleanup", lambda *_: binary.copy())
    monkeypatch.setattr(cad, "extract_component", lambda *_: next(components))
    with pytest.raises(ValueError, match="after surviving D7 cleanup"):
        cad.find_full_res_separation_threshold(gray, 100, (4, 4), guard)


def _refinement_state():
    gray = np.zeros((31, 31), np.uint8)
    gray[10:21, 10:21] = 200
    guard = np.ones_like(gray, bool)
    result = cad.AutoThresholdResult(
        full_res_seed_point=(15, 15),
        full_res_separation_threshold=100,
        full_res_separation_guard_mask=cad.compress_image(guard),
    )
    return gray, {"auto_threshold_result": result}


def _patch_refinement_candidate(monkeypatch, gray, roughness=1.0, hole_quality=1.0, edge_reliability=1.0):
    component = gray > 100
    contour = cad.find_external_contour(component)
    monkeypatch.setattr(
        cad,
        "extract_separated_seed_component",
        lambda *_: (component, contour),
    )
    monkeypatch.setattr(cad, "measure_filled_area", lambda *_: 100)
    monkeypatch.setattr(cad, "measure_roughness", lambda *_: roughness)
    monkeypatch.setattr(cad, "measure_hole_quality", lambda *_: hole_quality)
    monkeypatch.setattr(
        cad,
        "measure_edge_alignment",
        lambda *_: (0.5, edge_reliability),
    )


def test_invalid_stageb_score_scale_is_invariant_error(monkeypatch):
    gray, state = _refinement_state()
    _patch_refinement_candidate(monkeypatch, gray, roughness=0.0)
    with pytest.raises(ValueError, match="invalid within-image score scale"):
        cad.refine_threshold(gray, state)


def test_nonfinite_stageb_edge_reliability_is_invariant_error(monkeypatch):
    gray, state = _refinement_state()
    _patch_refinement_candidate(monkeypatch, gray, edge_reliability=math.nan)
    with pytest.raises(ValueError, match="median edge reliability must be finite"):
        cad.refine_threshold(gray, state)


def test_stageb_measurements_without_score_winner_are_invariant_error(monkeypatch):
    gray, state = _refinement_state()
    _patch_refinement_candidate(monkeypatch, gray, hole_quality=math.nan)
    with pytest.raises(ValueError, match="produced no score winner"):
        cad.refine_threshold(gray, state)


def test_sampled_profiles_with_zero_total_length_are_invariant_error(monkeypatch):
    profiles = np.ones((1, 2 * cad.EDGE_PROFILE_RADIUS_PX + 1), np.float64)
    monkeypatch.setattr(
        cad,
        "sample_grayscale_profiles",
        lambda *_: (profiles, np.array([0.0], np.float64)),
    )
    with pytest.raises(ValueError, match="length must be finite and positive"):
        cad.measure_edge_alignment(np.zeros((9, 9), np.uint8), np.zeros((3, 2), np.int32))
