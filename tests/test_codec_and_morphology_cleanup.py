import struct
import zlib

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def test_image_codec_roundtrips_bool_shapes_and_records_channel_axis_exactly():
    mask = np.zeros((13, 17), bool)
    mask[2:10, 3:15] = True
    for image, channels in ((mask, 0), (mask[..., None], 1)):
        payload = cad.compress_image(image)
        restored = cad.decompress_image(payload)
        assert restored.dtype == bool
        assert restored.shape == image.shape
        assert np.array_equal(restored, image)
        raw = zlib.decompress(payload)
        assert struct.unpack("<IIBB", raw[:10]) == (13, 17, channels, 1)


def test_image_codec_roundtrips_uint8_gray_gray_alpha_and_bgra():
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    gray_alpha = np.dstack([gray, np.full_like(gray, 173)])
    bgra = np.dstack([gray, gray, gray, np.full_like(gray, 255)])

    for image, channels in (
        (gray, 0),
        (gray[..., None], 1),
        (gray_alpha, 2),
        (bgra, 4),
    ):
        payload = cad.compress_image(image)
        raw = zlib.decompress(payload)
        assert struct.unpack("<IIBB", raw[:10]) == (5, 7, channels, 8)
        restored = cad.decompress_image(payload)
        assert restored.dtype == np.uint8
        assert restored.shape == image.shape
        assert np.array_equal(restored, image)


def test_image_codec_roundtrips_uint16_gray_alpha_in_explicit_little_endian_storage():
    gray = (np.arange(24, dtype=np.uint16).reshape(4, 6) * 521)
    gray_alpha = np.dstack([gray, np.full_like(gray, 65535)])
    payload = cad.compress_image(gray_alpha)
    raw = zlib.decompress(payload)
    assert struct.unpack("<IIBB", raw[:10]) == (4, 6, 2, 16)
    restored = cad.decompress_image(payload)
    assert restored.dtype == np.uint16
    assert restored.shape == gray_alpha.shape
    assert np.array_equal(restored, gray_alpha)


def test_image_codec_preserves_uint16_two_dimensional_and_explicit_one_channel_shapes():
    gray = (np.arange(20, dtype=np.uint16).reshape(4, 5) * 977)
    for image, channels in ((gray, 0), (gray[..., None], 1)):
        payload = cad.compress_image(image)
        raw = zlib.decompress(payload)
        assert struct.unpack("<IIBB", raw[:10]) == (4, 5, channels, 16)
        restored = cad.decompress_image(payload)
        assert restored.dtype == np.uint16
        assert restored.shape == image.shape
        assert np.array_equal(restored, image)


def test_image_codec_rejects_unsupported_dtype_and_channels():
    with pytest.raises(ValueError, match="dtype"):
        cad.compress_image(np.zeros((3, 4), np.float32))
    with pytest.raises(ValueError, match="channel count"):
        cad.compress_image(np.zeros((3, 4, 5), np.uint8))


def test_image_decoder_rejects_invalid_channel_count_bit_depth_and_payload_length():
    bad_channels = struct.pack("<IIBB", 2, 2, 5, 8) + b"1234"
    with pytest.raises(ValueError, match="channel count"):
        cad.decompress_image(zlib.compress(bad_channels))

    bad_depth = struct.pack("<IIBB", 2, 2, 0, 7) + b"1234"
    with pytest.raises(ValueError, match="bit depth"):
        cad.decompress_image(zlib.compress(bad_depth))

    bad_length = struct.pack("<IIBB", 2, 2, 2, 8) + b"1234567"
    with pytest.raises(ValueError, match="expected 8"):
        cad.decompress_image(zlib.compress(bad_length))


def test_image_decoder_rejects_multichannel_one_bit_payload():
    raw = struct.pack("<IIBB", 2, 2, 2, 1) + b"\x00"
    with pytest.raises(ValueError, match="exactly one channel"):
        cad.decompress_image(zlib.compress(raw))


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
    assert np.array_equal(cad.decompress_image(solar.component_mask), first)
    assert cad.decompress_image(solar.roi_6_5_mask).shape == gray.shape
    assert cad.decompress_image(solar.guard_19_5_mask).shape == gray.shape
    monkeypatch.setattr(cad, 'largest_enclosed_bright_component', lambda *_: (_ for _ in ()).throw(AssertionError('recomputed')))
    second = cad.resolve_threshold(gray, 100, state)
    assert state['solar_data'] is solar
    assert np.array_equal(second, first)
