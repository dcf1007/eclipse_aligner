from copy import deepcopy

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


def _disk_gray(shape=(61, 61), center=(30, 30), radius=18):
    gray = np.zeros(shape, np.uint8)
    cv2.circle(gray, center, radius, 180, -1)
    gray[center[1], center[0]] = 240
    return gray


def _run_autot(gray):
    state = {"auto_threshold_result": None, "solar_data": None}
    cad.find_separation_threshold(gray, state)
    selected = cad.refine_threshold(gray, state)
    assert selected is not None
    return state, selected


def test_exact_autot_winner_reuses_seed_guard_component_and_contour_without_reprocessing(monkeypatch):
    gray = _disk_gray()
    state, selected = _run_autot(gray)
    result = state["auto_threshold_result"]
    expected_component = cad.decompress_image(result.full_res_refined_component_mask)

    monkeypatch.setattr(
        cad,
        "extract_separated_seed_component",
        lambda *_: (_ for _ in ()).throw(AssertionError("winner recomputed")),
    )
    monkeypatch.setattr(
        cad,
        "derive_full_res_seed_and_guard",
        lambda *_: (_ for _ in ()).throw(AssertionError("identity recomputed")),
    )
    resolved = cad.resolve_threshold(gray, selected, state)
    solar = state["solar_data"]
    assert np.array_equal(resolved, expected_component)
    assert solar.seed_point == result.full_res_seed_point
    assert solar.guard_mask is result.full_res_separation_guard_mask
    assert solar.component_mask is result.full_res_refined_component_mask
    assert solar.component_contour is result.full_res_refined_component_contour


def test_manual_selected_t_uses_autot_identity_without_mutating_autot():
    gray = _disk_gray()
    state, selected = _run_autot(gray)
    result = state["auto_threshold_result"]
    before = vars(result).copy()
    manual_t = min(255, selected + 1) if selected < 255 else selected - 1
    resolved = cad.resolve_threshold(gray, manual_t, state)
    assert np.any(resolved)
    assert vars(result) == before
    assert state["solar_data"].threshold == manual_t


def test_failed_autot_with_full_seed_and_guard_is_consumed_without_retry(monkeypatch):
    gray = _disk_gray()
    seed = (30, 30)
    guard = np.zeros_like(gray, bool)
    guard[5:56, 5:56] = True
    result = cad.AutoThresholdResult(
        work_res_separation_component_mask=cad.compress_image(gray > 100),
        full_res_seed_point=seed,
        full_res_separation_guard_mask=cad.compress_image(guard),
        failure_reason="full-resolution separation: failed",
    )
    state = {"auto_threshold_result": result, "solar_data": None}
    monkeypatch.setattr(
        cad,
        "find_separation_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Auto-T retried")),
    )
    monkeypatch.setattr(
        cad,
        "refine_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Auto-T refined")),
    )
    resolved = cad.resolve_threshold(gray, 100, state)
    assert np.any(resolved)
    assert state["auto_threshold_result"] is result


def test_failed_autot_without_mature_work_component_uses_exact_manual_t_proposal():
    gray = _disk_gray()
    result = cad.AutoThresholdResult(
        failure_reason="work-resolution separation: no Auto-T proposal"
    )
    state = {"auto_threshold_result": result, "solar_data": None}
    resolved = cad.resolve_threshold(gray, 100, state)
    assert np.any(resolved)
    assert state["auto_threshold_result"] is result
    assert result.work_res_separation_component_mask is None


def test_manual_t_does_not_search_neighboring_thresholds_when_exact_proposal_is_absent(monkeypatch):
    gray = np.zeros((81, 101), np.uint8)
    gray[:, :20] = 220
    result = cad.AutoThresholdResult(
        failure_reason="work-resolution separation: no Auto-T proposal"
    )
    state = {"auto_threshold_result": result, "solar_data": None}
    calls = []
    real = cad.largest_enclosed_bright_component

    def record(binary):
        calls.append(binary.copy())
        return real(binary)

    monkeypatch.setattr(cad, "largest_enclosed_bright_component", record)
    with pytest.raises(cad.ThresholdResolutionError, match="selected T=100"):
        cad.resolve_threshold(gray, 100, state)
    assert len(calls) == 1


def test_mismatched_solardata_is_invalidated_and_rebuilt_by_resolver():
    gray = _disk_gray()
    state, selected = _run_autot(gray)
    cad.resolve_threshold(gray, selected, state)
    old = state["solar_data"]
    manual_t = min(255, selected + 1) if selected < 255 else selected - 1
    cad.resolve_threshold(gray, manual_t, state)
    assert state["solar_data"] is not old
    assert state["solar_data"].threshold == manual_t


def test_corrupt_same_t_solardata_raises_instead_of_rebuilding():
    gray = _disk_gray()
    state, selected = _run_autot(gray)
    stale = cad.SolarData(
        threshold=selected,
        seed_point=(30, 30),
        component_mask=b"not-zlib",
        guard_mask=b"not-zlib",
        component_contour=b"not-zlib",
    )
    state["solar_data"] = stale
    with pytest.raises(ValueError, match="same-T SolarData payload is corrupt"):
        cad.resolve_threshold(gray, selected, state)
    assert state["solar_data"] is stale


def test_resolver_requests_autot_stages_without_constructing_or_mutating_result_fields():
    import ast
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(cad.resolve_threshold))
    tree = ast.parse(source)
    assert "find_separation_threshold(" in source
    assert "refine_threshold(" in source
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AutoThresholdResult"
        for node in ast.walk(tree)
    )
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "auto_threshold_result"
            ):
                raise AssertionError("resolver must not assign Auto-T result fields")

    gray = _disk_gray()
    state = {"auto_threshold_result": None, "solar_data": None}
    resolved = cad.resolve_threshold(gray, 100, state)
    assert np.any(resolved)
    assert isinstance(state["auto_threshold_result"], cad.AutoThresholdResult)


def test_0114_t22_regression_reuses_autot_winner_when_raw_sun_is_boundary_connected(monkeypatch):
    gray = np.zeros((101, 141), np.uint8)
    component_u8 = np.zeros_like(gray)
    cv2.circle(component_u8, (75, 50), 22, 255, -1)
    component = component_u8 != 0
    gray[component] = 30
    gray[50, 75] = 40
    gray[49:52, :54] = 23
    gray[49:52, 53:76] = 23
    assert np.any((gray > 22)[:, 0])

    guard = np.zeros_like(gray, bool)
    guard[15:86, 35:116] = True
    contour = cad.find_external_contour(component)
    result = cad.AutoThresholdResult(
        histogram_start_threshold=30,
        work_res_seed_point=(75, 50),
        work_res_separation_threshold=22,
        work_res_separation_component_mask=cad.compress_image(component),
        full_res_seed_point=(75, 50),
        full_res_separation_threshold=22,
        full_res_separation_component_mask=cad.compress_image(component),
        full_res_separation_guard_mask=cad.compress_image(guard),
        full_res_refined_threshold=22,
        full_res_refined_component_mask=cad.compress_image(component),
        full_res_refined_component_contour=cad.compress_contour(contour),
    )
    state = {"auto_threshold_result": result, "solar_data": None}
    monkeypatch.setattr(
        cad,
        "largest_enclosed_bright_component",
        lambda *_: (_ for _ in ()).throw(AssertionError("raw identity reselected")),
    )
    resolved = cad.resolve_threshold(gray, 22, state)
    assert np.array_equal(resolved, component)
    assert not np.any(resolved[:, 0])
    assert state["solar_data"].threshold == 22
