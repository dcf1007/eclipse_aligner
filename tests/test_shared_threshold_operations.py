import cv2
import numpy as np

import circle_arc_detector as cad


def test_derive_full_res_seed_and_guard_matches_stage_a_identity_products():
    gray = np.zeros((181, 257), np.uint8)
    cv2.circle(gray, (128, 90), 35, 220, -1)
    gray[90, 128] = 250
    work_shape = cad.calculate_work_res_shape(gray.shape)
    work_gray = cad.resize_img(gray, work_shape)
    kernel = cad.generate_kernel((5, 5), round_kernel=False)
    start = cad.find_histogram_start_threshold(work_gray)
    _, _, work_component = cad.find_work_res_separation_threshold(
        work_gray, start, kernel
    )

    expected_seed, expected_guard = cad.derive_full_res_seed_and_guard(
        gray, work_component, kernel
    )
    state = {"auto_threshold_result": None}
    cad.find_separation_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert result.full_res_seed_point == expected_seed
    assert np.array_equal(
        cad.decompress_image(result.full_res_separation_guard_mask),
        expected_guard,
    )


def test_extract_separated_seed_component_runs_exact_progressive_p357(monkeypatch):
    threshold_mask = np.zeros((81, 93), np.uint8)
    cv2.circle(threshold_mask, (46, 40), 18, 255, -1)
    guard = np.zeros((81, 93), bool)
    guard[10:71, 12:81] = True
    boundary = cad.find_guard_boundary(guard)
    calls = []
    real = cad.morphological_cleanup

    def record(source, kernel, threshold=None):
        calls.append(kernel.shape)
        return real(source, kernel, threshold)

    monkeypatch.setattr(cad, "morphological_cleanup", record)
    candidate = cad.extract_separated_seed_component(
        threshold_mask, (46, 40), guard, boundary
    )
    assert candidate is not None
    component, contour = candidate
    assert calls == [(3, 3), (5, 5), (7, 7)]
    assert component[40, 46]
    assert contour.dtype == np.int32


def test_extract_separated_seed_component_rejects_guard_boundary_contact():
    threshold_mask = np.full((41, 51), 255, np.uint8)
    guard = np.ones((41, 51), bool)
    boundary = cad.find_guard_boundary(guard)
    assert cad.extract_separated_seed_component(
        threshold_mask, (25, 20), guard, boundary
    ) is None
