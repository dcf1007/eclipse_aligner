import cv2
import numpy as np

import circle_arc_detector as cad


def test_measure_filled_area_includes_lattice_boundary_without_helper():
    mask = np.zeros((11, 11), bool)
    mask[3:8, 3:8] = True
    contour = cad.find_external_contour(mask)
    assert cad.measure_filled_area(contour) == 25



def test_find_external_contour_returns_application_xy_int32_without_geometry_change():
    mask = np.zeros((31, 37), bool)
    mask[5:24, 7:29] = True
    mask[9:13, 29:33] = True

    contour = cad.find_external_contour(mask)
    assert contour.dtype == np.int32
    assert contour.ndim == 2 and contour.shape[1] == 2

    source = np.where(mask, 255, 0).astype(np.uint8)
    opencv_contours, _ = cv2.findContours(
        source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    expected = max(opencv_contours, key=cv2.contourArea).reshape(-1, 2)
    assert np.array_equal(contour, expected)

def test_refinement_window_is_base_through_base_plus_ten(monkeypatch):
    gray = np.zeros((61, 61), np.uint8)
    cv2.circle(gray, (30, 30), 18, 180, -1)
    gray[30, 30] = 240
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(gray, result)
    assert result.separation_threshold_complete
    base_threshold = result.full_res_separation_threshold

    seen = []
    real = cad.morphological_cleanup

    def record(source, kernel, threshold=None):
        if threshold is not None:
            seen.append(threshold)
        return real(source, kernel, threshold)

    monkeypatch.setattr(cad, "morphological_cleanup", record)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    cad.refine_threshold(gray, result)
    # Stage B thresholds once then applies P357 to the existing mask, so no
    # threshold-mode morphology is used during refinement.
    assert seen == []
    assert base_threshold <= result.full_res_refined_threshold <= min(
        255, base_threshold + cad.MAX_T_REFINEMENT_STEPS
    )


def test_refinement_inlines_progressive_p357_cleanup(monkeypatch):
    gray = np.zeros((81, 81), np.uint8)
    cv2.circle(gray, (40, 40), 20, 180, -1)
    gray[40, 40] = 240
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(gray, result)
    assert result.separation_threshold_complete

    calls = []
    real = cad.morphological_cleanup

    def record(source, kernel, threshold=None):
        calls.append(kernel.shape)
        return real(source, kernel, threshold)

    monkeypatch.setattr(cad, "morphological_cleanup", record)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    cad.refine_threshold(gray, result)
    assert calls[:3] == [(3, 3), (5, 5), (7, 7)]
    assert len(calls) % 3 == 0
