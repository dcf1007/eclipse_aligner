import struct
import zlib

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_codec_roundtrips_bool_mask_and_embeds_shape():
    mask = np.zeros((13, 17), bool)
    mask[2:10, 3:15] = True
    payload = cad.compress_array(mask)
    restored = cad.decompress_array(payload)
    assert restored.dtype == bool
    assert restored.shape == mask.shape
    assert np.array_equal(restored, mask)
    raw = zlib.decompress(payload)
    assert struct.unpack('<IIB', raw[:9]) == (13, 17, 1)


def test_codec_roundtrips_uint8_gray_and_bgra():
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    bgra = np.dstack([gray, gray, gray, np.full_like(gray, 255)])
    for array in (gray, bgra):
        restored = cad.decompress_array(cad.compress_array(array))
        assert restored.dtype == np.uint8
        assert restored.shape == array.shape
        assert np.array_equal(restored, array)


def test_codec_roundtrips_uint16_in_explicit_little_endian_storage():
    master = np.arange(4 * 6 * 4, dtype=np.uint16).reshape(4, 6, 4) * 521
    payload = cad.compress_array(master)
    raw = zlib.decompress(payload)
    assert struct.unpack('<IIB', raw[:9]) == (4, 6, 16)
    restored = cad.decompress_array(payload)
    assert restored.dtype.itemsize == 2
    assert restored.shape == master.shape
    assert np.array_equal(restored, master)


def test_codec_rejects_unsupported_dtype_and_channels():
    with pytest.raises(ValueError):
        cad.compress_array(np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError):
        cad.compress_array(np.zeros((3, 4, 2), np.uint8))


def test_codec_rejects_invalid_bit_depth():
    raw = struct.pack('<IIB', 2, 2, 7) + b'1234'
    with pytest.raises(ValueError, match='bit depth'):
        cad.decompress_array(zlib.compress(raw))


def test_morphological_cleanup_threshold_mode_matches_explicit_open_close():
    gray = np.zeros((31, 31), np.uint8)
    gray[6:25, 6:25] = 180
    gray[3, 3] = 255
    kernel = cad.generate_kernel((5, 5), round_kernel=True)
    expected = cv2.compare(gray, 100, cv2.CMP_GT)
    expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN, kernel)
    expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, kernel)
    assert np.array_equal(cad.morphological_cleanup(gray, kernel, 100), expected)


def test_morphological_cleanup_mask_mode_matches_explicit_open_close():
    mask = np.zeros((31, 31), bool)
    mask[6:25, 6:25] = True
    mask[3, 3] = True
    kernel = cad.generate_kernel((3, 3), round_kernel=True)
    expected = np.where(mask, 255, 0).astype(np.uint8)
    expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN, kernel)
    expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, kernel)
    assert np.array_equal(cad.morphological_cleanup(mask, kernel), expected)


def test_solardata_masks_use_shared_self_describing_codec_and_reuse_exact_state(monkeypatch):
    gray = np.zeros((81, 81), np.uint8)
    cv2.circle(gray, (40, 40), 18, 180, -1)
    gray[40, 40] = 240
    state = {"settings": cad.ImageSettings(threshold=100), "auto_threshold_result": None, "solar_data": None}
    first = cad.resolve_threshold(gray, 100, state)
    solar = state["solar_data"]
    assert isinstance(solar, cad.SolarData)
    assert np.array_equal(cad.decompress_array(solar.component_mask), first)
    assert cad.decompress_array(solar.roi_6_5_mask).shape == gray.shape
    assert cad.decompress_array(solar.guard_19_5_mask).shape == gray.shape
    monkeypatch.setattr(cad, 'largest_enclosed_bright_component', lambda *_: (_ for _ in ()).throw(AssertionError('recomputed')))
    second = cad.resolve_threshold(gray, 100, state)
    assert state['solar_data'] is solar
    assert np.array_equal(second, first)
