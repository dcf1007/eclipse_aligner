from __future__ import annotations

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _state():
    return {
        "settings": cad.ImageSettings(),
        "auto_threshold_result": None,
        "solar_data": None,
    }


def test_resize_img_binary_uses_exact_nearest_coordinate_mapping():
    source = np.array([[0, 0, 255, 0, 0]], dtype=np.uint8)
    resized = cad.resize_img(source, (3, 1))
    assert resized.dtype == source.dtype
    assert resized.shape == (1, 3)
    assert resized.tolist() == [[0, 255, 0]]


def test_resize_img_preserves_bgra_uint16_dtype_channels_and_aspect_ratio():
    image = np.arange(4 * 6 * 4, dtype=np.uint16).reshape(4, 6, 4) * 17
    resized = cad.resize_img(image, (3, 2))
    assert resized.dtype == np.uint16
    assert resized.shape == (2, 3, 4)


def test_resize_img_calculates_one_missing_dimension():
    image = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    assert cad.resize_img(image, (3, None)).shape == (2, 3)
    assert cad.resize_img(image, (None, 2)).shape == (2, 3)

def test_histogram_start_threshold_returns_only_preceding_left_valley():
    gray = np.concatenate(
        (np.full(100, 40, dtype=np.uint8), np.full(200, 100, dtype=np.uint8))
    )
    assert cad.find_histogram_start_threshold(gray) == 98


def test_generate_kernel_unifies_square_and_exact_euclidean_disk_geometry():
    square = cad.generate_kernel(5, round_kernel=False)
    assert square.shape == (5, 5)
    assert square.dtype == np.uint8
    assert np.all(square == 1)

    expected_pixels = {3: 5, 5: 13, 7: 29}
    for size, pixel_count in expected_pixels.items():
        disk = cad.generate_kernel(size, round_kernel=True)
        assert disk.shape == (size, size)
        assert int(disk.sum()) == pixel_count
        assert np.array_equal(disk, np.rot90(disk))

    with pytest.raises(ValueError):
        cad.generate_kernel(4)


def test_supported_point_returns_none_without_requested_support():
    gray = np.zeros((21, 21), np.uint8)
    component = np.zeros_like(gray)
    component[10, 3:18] = 255
    gray[10, 10] = 250
    assert (
        cad.brightest_supported_component_point(
            gray, component, cad.generate_kernel(5)
        )
        is None
    )
    assert cad.brightest_supported_component_point(
        gray, component, cad.generate_kernel(1)
    ) == (10, 10)


def test_work_search_stops_rediscovering_candidates_after_seed_is_established(monkeypatch):
    gray = np.zeros((31, 31), np.uint8)
    gray[11:20, 11:20] = 15
    gray[14:17, 14:17] = 30

    real_largest = cad.largest_enclosed_bright_component
    calls = []

    def counted(binary):
        calls.append(1)
        return real_largest(binary)

    monkeypatch.setattr(cad, "largest_enclosed_bright_component", counted)
    work_T, component = cad.find_work_res_solar_component(
        gray,
        20,
        cad.generate_kernel(5),
    )

    # T20..T15 expose only the unsupported 3x3 core; T14 exposes the 9x9 body.
    # Once that 5x5-supported seed exists, lower thresholds use seed flood tracking.
    assert len(calls) == 7
    assert work_T == 0
    assert component.dtype == bool
    assert int(component.sum()) == 81


def test_work_search_fails_only_after_no_supported_seed_exists_through_zero():
    gray = np.zeros((21, 21), np.uint8)
    gray[9:12, 9:12] = 30
    with pytest.raises(cad.ThresholdResolutionError, match="through T=0"):
        cad.find_work_res_solar_component(gray, 20, cad.generate_kernel(5))


def _guard(shape=(21, 21)):
    guard = np.zeros(shape, bool)
    guard[3:18, 3:18] = True
    return guard


def test_full_res_threshold_search_moves_down_only_when_start_is_enclosed():
    gray = np.zeros((21, 21), np.uint8)
    gray[8:13, 8:13] = 30
    gray[10, 12:18] = 10  # joins the guard boundary only when T drops below 10
    T, component = cad.find_lowest_full_res_threshold(
        gray,
        10,
        (10, 10),
        _guard(),
    )
    assert T == 10
    assert component[10, 10]
    assert not component[10, 17]


def test_full_res_threshold_search_moves_up_only_when_start_touches_guard():
    gray = np.zeros((21, 21), np.uint8)
    gray[8:13, 8:13] = 30
    gray[10, 12:18] = 11  # light at T10, dark at T11
    T, component = cad.find_lowest_full_res_threshold(
        gray,
        10,
        (10, 10),
        _guard(),
    )
    assert T == 11
    assert component[10, 10]
    assert not component[10, 17]


def test_dilate_component_mask_returns_only_one_full_resolution_mask():
    component = np.zeros((31, 41), bool)
    component[14:17, 19:22] = True
    same = cad.dilate_component_mask(component, 0.0)
    grown = cad.dilate_component_mask(component, 4.0)
    assert same.dtype == bool and same.shape == component.shape
    assert grown.dtype == bool and grown.shape == component.shape
    assert np.array_equal(same, component)
    assert np.all(grown[component])
    assert int(grown.sum()) > int(component.sum())


def test_auto_result_contains_only_persistent_streamlined_search_state():
    fields = tuple(cad.AutoThresholdResult.__dataclass_fields__)
    assert fields == (
        "threshold",
        "histogram_start_threshold",
        "work_res_threshold",
        "full_res_seed_point",
        "resolved",
        "reason",
    )


def test_auto_threshold_keeps_synthetic_bridge_result_with_streamlined_flow():
    gray = np.zeros((220, 320), dtype=np.uint8)
    cv2.circle(gray, (180, 110), 46, 30, -1)
    gray[106:115, 0:180] = 8
    cv2.circle(gray, (190, 105), 12, 70, -1)

    state = _state()
    threshold = cad.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert result.resolved
    assert threshold == result.threshold == 8
    assert result.work_res_threshold == 8
    assert result.full_res_seed_point is not None


def test_auto_maps_work_component_directly_to_exact_full_res_size(monkeypatch):
    full_res_gray = np.zeros((800, 1259), np.uint8)
    monkeypatch.setattr(cad, "find_histogram_start_threshold", lambda _gray: 20)

    def fake_work(work_res_gray, _start_T, _kernel):
        component = np.zeros(work_res_gray.shape, bool)
        component[300:450, 500:700] = True
        return 12, component

    monkeypatch.setattr(cad, "find_work_res_solar_component", fake_work)
    seen = {}

    def fake_seed(_gray, search_mask, _kernel):
        seen["shape"] = search_mask.shape
        return (629, 400)

    monkeypatch.setattr(cad, "brightest_supported_component_point", fake_seed)
    monkeypatch.setattr(cad, "dilate_component_mask", lambda mask, _margin: np.ones(mask.shape, bool))
    monkeypatch.setattr(
        cad,
        "find_lowest_full_res_threshold",
        lambda _gray, start_T, _seed, _guard: (start_T, np.ones(full_res_gray.shape, bool)),
    )
    monkeypatch.setattr(
        cad,
        "optimize_separated_threshold",
        lambda _gray, base_T, _seed, _component: cad.ThresholdTopologySelection(
            threshold=base_T,
            base_threshold=base_T,
            delta=0,
            trajectory=(),
            net_quality=(),
            knee_curve=(),
        ),
    )
    assert cad.find_auto_threshold(full_res_gray, _state()) == 12
    assert seen["shape"] == full_res_gray.shape

def test_resize_img_is_the_only_direct_cv2_resize_call_in_production_source():
    from pathlib import Path

    source = Path(cad.__file__).read_text(encoding="utf-8")
    assert source.count("cv2.resize(") == 1


def test_gui_auto_threshold_success_status_uses_streamlined_result_fields():
    from pathlib import Path

    source = Path(cad.__file__).read_text(encoding="utf-8")
    block = source.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius(self):", 1
    )[0]
    assert "result.work_res_threshold" in block
    assert "result.histogram_start_threshold" in block
    assert "result.coarse_threshold" not in block
    assert "result.histogram_left_edge" not in block
