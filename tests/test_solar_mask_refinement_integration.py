from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import cv2
import numpy as np

import circle_arc_detector as candidate

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = ROOT / "snippets" / "refine_solar_component_mask.py"


def _snippet_module():
    spec = importlib.util.spec_from_file_location("refinement_snippet", SNIPPET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _function_ast(path: Path, name: str):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node, include_attributes=False)


def _state(threshold):
    return {
        "settings": candidate.ImageSettings(threshold=threshold),
        "auto_threshold_result": None,
        "solar_data": None,
    }


def test_production_uses_exact_tested_refinement_function_and_shared_kernel():
    assert _function_ast(ROOT / "circle_arc_detector.py", "refine_solar_component_mask") == _function_ast(
        SNIPPET, "refine_solar_component_mask"
    )
    snippet = _snippet_module()
    assert candidate.SOLAR_COMPONENT_KERNEL_SIZE == snippet.SOLAR_COMPONENT_KERNEL_SIZE == 7
    assert candidate.REFINEMENT_ITERATIONS == snippet.REFINEMENT_ITERATIONS == 1
    assert np.array_equal(candidate.SOLAR_COMPONENT_KERNEL, snippet.SOLAR_COMPONENT_KERNEL)


def test_refinement_is_inside_atomic_current_t_resolver():
    auto_source = inspect.getsource(candidate.find_auto_threshold)
    resolve_source = inspect.getsource(candidate.resolve_threshold)
    assert "refine_solar_component_mask" not in auto_source
    assert "refine_solar_component_mask" in resolve_source
    assert resolve_source.index("brightest_supported_component_point") < resolve_source.index("refine_solar_component_mask")
    assert resolve_source.index("refine_solar_component_mask") < resolve_source.index("SolarData(")
    assert resolve_source.index("SolarData(") < resolve_source.index('image_state["solar_data"]')


def test_resolver_returns_and_stores_same_refined_component():
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[20:61, 20:61] = True
    raw[19, 19] = True
    raw[40, 40] = False
    gray[raw] = 220
    gray[40, 41] = 250
    state = _state(100)

    expected = candidate.refine_solar_component_mask(raw)
    refined = candidate.resolve_threshold(gray, 100, state)
    stored = candidate.decompress_full_mask(state["solar_data"].component_mask, gray.shape)
    assert np.array_equal(refined, expected)
    assert np.array_equal(stored, refined)


def test_roi_guard_and_contour_derive_from_same_returned_refined_component():
    gray = np.zeros((101, 121), np.uint8)
    raw_u8 = np.zeros_like(gray, np.uint8)
    cv2.circle(raw_u8, (60, 50), 24, 255, -1)
    raw = raw_u8 != 0
    raw[25, 35] = True
    gray[raw] = 220
    gray[50, 60] = 250
    state = _state(100)

    refined = candidate.resolve_threshold(gray, 100, state)
    solar = state["solar_data"]
    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5

    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        stored = candidate.decompress_full_mask(payload, gray.shape)
        region = candidate._build_observation_region(gray, refined, fraction * image_scale, solar.seed_point)
        expected = np.zeros(gray.shape, bool)
        x0, y0, x1, y1 = region.bbox
        expected[y0:y1, x0:x1] = region.allowed_u8 != 0
        assert np.array_equal(stored, expected)

    contour = solar.component_contour
    assert np.all(refined[contour[:, 1], contour[:, 0]])
