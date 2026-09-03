from pathlib import Path
import cv2
import numpy as np
import pytest
import circle_arc_detector as cad

SOURCE = Path(cad.__file__).read_text(encoding='utf-8')

def test_resize_mask_flag_owns_nearest_neighbor_selection(monkeypatch):
    source = np.array([[0, 0, 255, 0, 0]], dtype=np.uint8)
    seen = []
    def record_resize(array, size, interpolation):
        seen.append(interpolation)
        return np.zeros((size[1], size[0]), dtype=array.dtype)
    monkeypatch.setattr(cad.cv2, 'resize', record_resize)
    cad.resize_img(source, (3, 1))
    cad.resize_img(source, (3, 1), mask=True)
    assert seen == [cv2.INTER_AREA, cv2.INTER_NEAREST_EXACT]

def test_stage_a_d7_removes_thin_background_bridge_and_returns_only_t():
    gray = np.zeros((41, 61), dtype=np.uint8)
    gray[15:26, 25:36] = 30
    gray[20, 35:56] = 10
    guard = np.zeros(gray.shape, dtype=bool)
    guard[5:36, 5:56] = True
    result = cad.find_lowest_full_res_threshold(gray, 10, (30, 20), guard)
    assert isinstance(result, int)
    assert result == 0

def test_auto_stage_b_receives_t_seed_and_guard_not_stage_a_component(monkeypatch):
    gray = np.zeros((31, 31), dtype=np.uint8)
    state = {'settings': cad.ImageSettings(), 'auto_threshold_result': None, 'solar_data': None}
    monkeypatch.setattr(cad, 'find_histogram_start_threshold', lambda _gray: 10)
    def work(work_gray, _start, _kernel):
        component = np.zeros(work_gray.shape, bool)
        component[8:23, 8:23] = True
        return 7, component
    monkeypatch.setattr(cad, 'find_work_res_solar_component', work)
    monkeypatch.setattr(cad, 'brightest_supported_component_point', lambda *_args: (15, 15))
    guard = np.ones(gray.shape, bool)
    monkeypatch.setattr(cad, 'dilate_component_mask', lambda *_args: guard)
    monkeypatch.setattr(cad, 'find_lowest_full_res_threshold', lambda *_args: 6)
    seen = {}
    def stage_b(_gray, base_T, seed, received_guard):
        seen.update(T=base_T, seed=seed, guard=received_guard)
        return cad.ThresholdTopologySelection(base_T, base_T, 0, (), (), ())
    monkeypatch.setattr(cad, 'optimize_separated_threshold', stage_b)
    assert cad.find_auto_threshold(gray, state) == 6
    assert seen['T'] == 6 and seen['seed'] == (15, 15)
    assert seen['guard'] is guard

def test_uint8_bgr_normalizes_losslessly_to_uint16_bgra_and_gray8():
    bgr8 = np.array([[[0, 64, 255], [10, 20, 30]]], dtype=np.uint8)
    master = cad.normalize_master_bgra16(bgr8)
    expected_bgra8 = cv2.cvtColor(bgr8, cv2.COLOR_BGR2BGRA)
    assert master.dtype == np.uint16 and master.shape == (1, 2, 4)
    assert np.array_equal(master, expected_bgra8.astype(np.uint16) * 257)
    gray8 = cad.master_bgra16_to_gray8(master)
    assert np.array_equal(gray8, cv2.cvtColor(bgr8, cv2.COLOR_BGR2GRAY))

def test_uint16_bgra_preserves_alpha_and_round_trips_compression():
    master = np.array(
        [[[1, 2, 3, 4], [65535, 40000, 12345, 22222]]],
        dtype=np.uint16,
    )
    normalized = cad.normalize_master_bgra16(master)
    assert np.array_equal(normalized, master)
    payload = cad.compress_master_bgra16(normalized)
    restored = cad.decompress_master_bgra16(payload, normalized.shape)
    assert np.array_equal(restored, normalized)
    assert restored.dtype == np.uint16

def test_display_mapping_is_fixed_full_range_and_preserves_alpha_scale():
    master = np.array([[[0, 257, 65535, 32768]]], dtype=np.uint16)
    display = cad.master_bgra16_to_display_bgra8(master)
    assert display.tolist() == [[[0, 1, 255, 128]]]

def test_production_load_path_uses_unchanged_master_and_no_binary_inference():
    load_block = SOURCE.split('def load_image_at(self, index: int):', 1)[1].split('def previous_image', 1)[0]
    resize_block = SOURCE.split('def resize_img(', 1)[1].split('def normalize_master_bgra16', 1)[0]
    assert 'cv2.IMREAD_UNCHANGED' in load_block
    assert 'cv2.IMREAD_COLOR' not in load_block
    assert 'is_binary' not in resize_block
    assert 'if mask:' in resize_block
    assert 'size: tuple[int, int]' in SOURCE
