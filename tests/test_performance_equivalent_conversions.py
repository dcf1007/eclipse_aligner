import cv2
import numpy as np

import circle_arc_detector as cad


def test_uint8_binary_normalization_matches_bool_conversion_without_bool_temporary():
    mask_u8 = np.array(
        [[0, 1, 2, 255], [7, 0, 128, 0]],
        dtype=np.uint8,
    )
    expected = (mask_u8 != 0).astype(np.uint8)
    expected *= 255
    converted = cv2.compare(mask_u8, 0, cv2.CMP_NE)
    assert np.array_equal(converted, expected)


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
