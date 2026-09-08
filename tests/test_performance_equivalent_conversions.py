import cv2
import numpy as np

import circle_arc_detector as cad


def test_bool_mask_to_uint8_is_exact_zero_255_conversion():
    mask = np.array([[False, True, False], [True, True, False]], dtype=bool)
    converted = cad.bool_mask_to_uint8(mask)
    assert converted.dtype == np.uint8
    assert np.array_equal(
        converted,
        np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8),
    )


def test_uint16_to_uint8_opencv_mapping_matches_original_formula_exhaustively():
    values = np.arange(65536, dtype=np.uint16).reshape(-1, 1)
    expected = ((values.astype(np.uint32) + 128) // 257).astype(np.uint8)
    converted = cv2.convertScaleAbs(values, alpha=1.0 / 257.0)
    assert np.array_equal(converted, expected)


def test_uint8_to_uint16_in_place_expansion_is_exact_for_every_input_value():
    values = np.arange(256, dtype=np.uint8)
    expanded = values.astype(np.uint16)
    expanded *= 257
    assert np.array_equal(expanded, values.astype(np.uint16) * 257)
