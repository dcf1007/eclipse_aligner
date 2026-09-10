import inspect

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_bool_mask_resize_uses_zero_copy_opencv_views(monkeypatch):
    source = np.zeros((5, 7), dtype=bool)
    source[1:4, 2:6] = True
    real_resize = cv2.resize
    seen = []

    def resize(src, size, dst=None, interpolation=None):
        seen.append((src, dst, interpolation))
        return real_resize(src, size, dst=dst, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", resize)

    resized = cad.resize_img(source, (13, 17), mask=True)

    assert resized.dtype == bool
    assert seen and np.shares_memory(seen[0][0], source)
    assert np.shares_memory(seen[0][1], resized)
    assert seen[0][0].dtype == np.uint8 and seen[0][1].dtype == np.uint8
    assert seen[0][2] == cv2.INTER_NEAREST_EXACT


def test_one_bit_decoder_reuses_unpackbits_storage_as_bool():
    mask = np.zeros((19, 23), dtype=bool)
    mask[3:15, 5:20] = True
    restored = cad.decompress_image(cad.compress_image(mask))

    assert restored.dtype == bool
    assert np.array_equal(restored, mask)
    source = inspect.getsource(cad.decompress_image)
    assert ".reshape(shape).view(np.bool_)" in source
    assert ".reshape(shape) != 0" not in source


def test_largest_component_uses_bool_storage_as_opencv_input():
    binary = np.zeros((31, 41), dtype=bool)
    binary[5:20, 8:25] = True
    component = cad.largest_enclosed_bright_component(binary)

    assert component is not None
    assert component.dtype == bool
    assert np.all(component[5:20, 8:25])


def test_component_extraction_rejects_non_boolean_processing_raster():
    binary = np.zeros((17, 19), dtype=np.uint8)
    binary[3:14, 4:15] = 1

    with pytest.raises(ValueError, match="authoritative.*bool"):
        cad.extract_component(binary, (8, 8))


def test_component_extraction_allocates_boolean_result_and_preserves_input():
    binary = np.zeros((37, 53), dtype=bool)
    binary[5:30, 7:24] = True
    binary[12:27, 31:47] = True
    before = binary.copy()

    component = cad.extract_component(binary, (12, 15))

    assert component is not None
    assert component.dtype == bool
    assert np.array_equal(binary, before)
    assert np.all(component.view(np.uint8) <= 1)
    assert component[15, 12]
    assert not component[15, 35]


def test_guard_boundary_indices_match_established_inner_boundary_without_guard_copy():
    guard = np.zeros((61, 83), dtype=bool)
    guard[7:54, 11:72] = True
    guard[25:31, 34:42] = False
    before = guard.copy()

    guard_255 = guard.astype(np.uint8) * 255
    eroded = cv2.erode(
        guard_255,
        cad.GUARD_BOUNDARY_KERNEL,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
    expected = np.flatnonzero(guard & ~eroded)

    actual = cad.find_guard_boundary_indices(guard)

    assert np.array_equal(actual, expected)
    assert np.array_equal(guard, before)


def test_guard_boundary_helper_uses_one_boolean_scratch_and_uint8_views_only():
    source = inspect.getsource(cad.find_guard_boundary_indices)
    assert "boundary = np.empty_like(guard_mask)" in source
    assert "boundary.view(np.uint8)" in source
    assert "guard_mask.view(np.uint8)" in source
    assert "cv2.erode(" in source
    assert "cv2.bitwise_xor(" in source
    assert "cv2.findNonZero(" in source
    assert "bool_mask_to_uint8" not in source


def test_stage_b_binary_interfaces_are_boolean_outside_morphology(monkeypatch):
    gray = np.zeros((81, 81), dtype=np.uint8)
    cv2.circle(gray, (40, 40), 20, 180, -1)
    gray[40, 40] = 240
    state = {"auto_threshold_result": cad.AutoThresholdResult()}
    cad.find_separation_threshold(gray, state)

    real = cad.extract_separated_seed_component
    seen = []

    def record(threshold_mask, seed, guard_mask, boundary_indices):
        seen.append((threshold_mask.dtype, guard_mask.dtype))
        return real(threshold_mask, seed, guard_mask, boundary_indices)

    monkeypatch.setattr(cad, "extract_separated_seed_component", record)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    cad.refine_threshold(gray, state)

    assert seen
    assert all(threshold_dtype == bool and guard_dtype == bool for threshold_dtype, guard_dtype in seen)


def test_extract_separated_seed_component_accepts_and_returns_boolean_masks():
    threshold_mask = np.zeros((81, 93), dtype=bool)
    cv2.circle(threshold_mask.view(np.uint8), (46, 40), 18, 1, -1)
    guard = np.zeros_like(threshold_mask)
    guard[10:71, 12:81] = True

    candidate = cad.extract_separated_seed_component(
        threshold_mask,
        (46, 40),
        guard,
        cad.find_guard_boundary_indices(guard),
    )

    assert candidate is not None
    component, contour = candidate
    assert component.dtype == bool
    assert np.all(component.view(np.uint8) <= 1)
    assert contour.dtype == np.int32


def test_coarse_full_resolution_result_is_boolean():
    gray = np.zeros((101, 141), dtype=np.uint8)
    cv2.circle(gray, (75, 50), 22, 30, -1)
    gray[49:52, :76] = 20
    guard = np.zeros_like(gray, dtype=bool)
    guard[10:91, 30:121] = True

    _, component = cad.find_full_res_separation_threshold(gray, 20, (75, 50), guard)

    assert component.dtype == bool
    assert np.all(component.view(np.uint8) <= 1)



def test_morphological_cleanup_uses_bool_owner_as_opencv_source_and_destination(monkeypatch):
    mask = np.zeros((41, 53), dtype=bool)
    cv2.circle(mask.view(np.uint8), (26, 20), 12, 1, -1)
    mask[3, 3] = True
    kernel = cad.generate_kernel((5, 5), round_kernel=True)
    before = mask.copy()
    expected_u8 = before.astype(np.uint8) * 255
    expected_u8 = cv2.morphologyEx(expected_u8, cv2.MORPH_OPEN, kernel)
    expected_u8 = cv2.morphologyEx(expected_u8, cv2.MORPH_CLOSE, kernel)
    expected = expected_u8 != 0

    real = cv2.morphologyEx
    seen = []

    def morphology(src, op, morphology_kernel, dst=None, iterations=1):
        seen.append((src, dst, op))
        return real(src, op, morphology_kernel, dst=dst, iterations=iterations)

    monkeypatch.setattr(cv2, "morphologyEx", morphology)
    returned = cad.morphological_cleanup(mask, kernel)

    assert returned is mask
    assert np.array_equal(mask, expected)
    assert np.all(mask.view(np.uint8) <= 1)
    assert [entry[2] for entry in seen] == [cv2.MORPH_OPEN, cv2.MORPH_CLOSE]
    assert all(np.shares_memory(src, mask) for src, _, _ in seen)
    assert all(np.shares_memory(dst, mask) for _, dst, _ in seen)


def test_extract_separated_seed_component_consumes_the_supplied_processing_mask():
    processing_mask = np.zeros((81, 93), dtype=bool)
    cv2.circle(processing_mask.view(np.uint8), (46, 40), 18, 1, -1)
    processing_mask[5, 5] = True
    guard = np.zeros_like(processing_mask)
    guard[10:71, 12:81] = True
    before = processing_mask.copy()

    candidate = cad.extract_separated_seed_component(
        processing_mask,
        (46, 40),
        guard,
        cad.find_guard_boundary_indices(guard),
    )

    assert candidate is not None
    component, _ = candidate
    assert component.dtype == bool
    assert not np.array_equal(processing_mask, before)
    assert np.all(processing_mask.view(np.uint8) <= 1)
    assert not processing_mask[5, 5]


def test_reused_cv_threshold_writes_gray_gt_t_into_existing_bool_storage():
    gray = np.arange(15 * 19, dtype=np.uint8).reshape(15, 19)
    processing_mask = gray > 17
    allocation = processing_mask

    for threshold in (63, 127, 201):
        _, returned_view = cv2.threshold(
            gray,
            threshold,
            1,
            cv2.THRESH_BINARY,
            dst=processing_mask.view(np.uint8),
        )
        assert processing_mask is allocation
        assert np.shares_memory(returned_view, processing_mask)
        assert np.array_equal(processing_mask, gray > threshold)
        assert np.all(processing_mask.view(np.uint8) <= 1)
