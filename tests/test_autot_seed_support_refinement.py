import numpy as np
import pytest
import circle_arc_detector as cad


def test_seed_helper_uses_explicit_kernel_and_returns_none_without_fallback():
    gray = np.zeros((21, 21), np.uint8)
    comp = np.zeros_like(gray)
    comp[10, 3:18] = 255
    gray[10, 10] = 250
    assert cad.brightest_supported_component_point(gray, comp, cad.generate_kernel((5, 5))) is None
    assert cad.brightest_supported_component_point(gray, comp, cad.generate_kernel((1, 1))) == (10, 10)


def test_work_seed_kernel_is_generated_as_fixed_5x5_square():
    kernel = cad.generate_kernel((cad.TRACKING_SEED_KERNEL_SIZE, cad.TRACKING_SEED_KERNEL_SIZE), round_kernel=False)
    assert kernel.shape == (5, 5)
    assert np.all(kernel == 1)


def test_full_seed_support_maps_6016_to_25_square():
    full_shape = (4000, 6016)
    work_shape = (798, 1200)
    mapped = cad.TRACKING_SEED_KERNEL_SIZE * max(full_shape) / float(max(work_shape))
    low = max(1, int(np.floor(mapped)))
    if low % 2 == 0:
        low -= 1
    high = low + 2
    size = low if abs(mapped - low) <= abs(high - mapped) else high
    kernel = cad.generate_kernel((size, size), round_kernel=False)
    assert kernel.shape == (25, 25)
    assert np.all(kernel == 1)


def test_full_seed_support_stays_5_if_no_downscale():
    full_shape = (800, 1200)
    work_shape = full_shape
    mapped = cad.TRACKING_SEED_KERNEL_SIZE * max(full_shape) / float(max(work_shape))
    assert round(mapped) == 5
    assert cad.generate_kernel((5, 5)).shape == (5, 5)


def test_work_search_continues_below_unsupported_candidate():
    gray = np.zeros((31, 31), np.uint8)
    gray[11:20, 11:20] = 15
    gray[14:17, 14:17] = 30
    work_T, component = cad.find_work_res_solar_component(
        gray, 20, cad.generate_kernel((5, 5))
    )
    assert work_T == 0
    assert int(component.sum()) == 81


def test_work_search_errors_if_nothing_supported_through_zero():
    gray = np.zeros((21, 21), np.uint8)
    gray[9:12, 9:12] = 30
    with pytest.raises(cad.ThresholdResolutionError, match="through T=0"):
        cad.find_work_res_solar_component(gray, 20, cad.generate_kernel((5, 5)))


def test_unresolved_auto_is_stored_but_not_returned_as_histogram_fallback():
    gray = np.full((40, 50), 100, np.uint8)
    state = {"settings": cad.ImageSettings(), "auto_threshold_result": None, "solar_data": None}
    with pytest.raises(cad.ThresholdResolutionError):
        cad.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    assert not result.resolved
    assert result.threshold is None
    assert result.work_res_threshold is None
    assert result.full_res_seed_point is None
