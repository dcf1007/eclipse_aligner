import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _rectangle_contour(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((220, 220), dtype=np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0]


def _erf_edge_image(center: float, sigma: float) -> np.ndarray:
    x = np.arange(220, dtype=np.float64)
    profile = np.fromiter(
        (
            20.0
            + 200.0
            * 0.5
            * (1.0 - math.erf((value - center) / (sigma * math.sqrt(2.0))))
            for value in x
        ),
        dtype=np.float64,
        count=len(x),
    )
    return np.repeat(np.clip(np.rint(profile), 0, 255).astype(np.uint8)[None, :], 220, axis=0)


def test_hole_quality_is_one_without_missing_internal_area():
    contour = _rectangle_contour(50, 50, 149, 149)
    filled_area = cad.measure_filled_area(contour)
    assert cad.measure_hole_quality(contour, filled_area, filled_area) == pytest.approx(1.0)


def test_hole_quality_uses_isoperimetric_perimeter_scale():
    contour = _rectangle_contour(50, 50, 149, 149)
    filled_area = cad.measure_filled_area(contour)
    component_area = filled_area - 25
    perimeter = float(cv2.arcLength(contour, True))
    expected = perimeter / (perimeter + 2.0 * math.sqrt(math.pi * 25.0))
    assert cad.measure_hole_quality(contour, component_area, filled_area) == pytest.approx(expected)


def test_hole_quality_rejects_inconsistent_component_area():
    contour = _rectangle_contour(50, 50, 149, 149)
    filled_area = cad.measure_filled_area(contour)
    with pytest.raises(ValueError):
        cad.measure_hole_quality(contour, filled_area + 1, filled_area)


def test_profile_helper_returns_polygon_side_profiles_only():
    gray = np.tile(np.arange(220, dtype=np.uint8), (220, 1))
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask, (110, 110), 55, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    profiles, lengths = cad._sample_grayscale_profiles(gray, contour)
    assert profiles.shape[1] == 2 * cad.EDGE_PROFILE_RADIUS_PX + 1
    assert len(profiles) == len(lengths)
    assert 0 < len(profiles) < len(contour)
    assert np.all(lengths > 0.0)


def test_clean_error_function_edge_gets_high_reliability_and_finite_distance():
    gray = _erf_edge_image(center=120.0, sigma=3.0)
    contour = _rectangle_contour(45, 45, 116, 174)
    distance, reliability = cad.measure_edge_alignment(gray, contour)
    assert math.isfinite(distance)
    assert 0.0 <= distance < 15.0
    assert reliability > 0.25


def test_non_sigmoid_profile_reduces_edge_reliability():
    clean = _erf_edge_image(center=120.0, sigma=3.0)
    x = np.arange(220, dtype=np.float64)
    disturbance = 55.0 * np.sin((x - 105.0) * math.pi / 6.0)
    disturbed = np.clip(clean.astype(np.float64) + disturbance[None, :], 0, 255).astype(np.uint8)
    contour = _rectangle_contour(45, 45, 116, 174)
    _, clean_reliability = cad.measure_edge_alignment(clean, contour)
    _, disturbed_reliability = cad.measure_edge_alignment(disturbed, contour)
    assert disturbed_reliability < clean_reliability


def test_stageb_source_has_only_one_edge_profile_helper_and_no_solidity_descriptor():
    text = Path(cad.__file__).read_text(encoding="utf-8")
    for removed in (
        "def _nearest_mask(",
        "def _contour_normals(",
        "def _linear_fit(",
        "def _sample_edge_profiles(",
        "def measure_solidity(",
        "def measure_internal_dark_fraction(",
        "EDGE_SLOPE_RECOVERY_FRACTION",
        "EDGE_SLOPE_RECOVERY_PERSISTENCE_PX",
        "EDGE_PROFILE_MIN_SAMPLE_STRIDE",
        "EDGE_PROFILE_MAX_SAMPLE_COUNT",
        "EDGE_NORMAL_TANGENT_HALF_SPAN",
    ):
        assert removed not in text
    assert "def _sample_grayscale_profiles(" in text
    assert "def measure_edge_alignment(" in text
    assert "def measure_hole_quality(" in text
    assert "edge_reliability**2" not in text
    assert "0.5 * q_solidity" not in text
    assert "math.erf(" in text
