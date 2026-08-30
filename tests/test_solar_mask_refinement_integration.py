from __future__ import annotations

import ast
import inspect
import importlib.util
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
    node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.dump(node, include_attributes=False)


def test_production_uses_exact_tested_refinement_function():
    assert _function_ast(ROOT / "circle_arc_detector.py", "refine_solar_component_mask") == _function_ast(
        SNIPPET, "refine_solar_component_mask"
    )
    snippet = _snippet_module()
    assert candidate.REFINEMENT_KERNEL_SIZE == snippet.REFINEMENT_KERNEL_SIZE == 7
    assert candidate.REFINEMENT_ITERATIONS == snippet.REFINEMENT_ITERATIONS == 1


def test_refinement_is_post_threshold_only():
    auto_source = inspect.getsource(candidate.auto_threshold)
    build_source = inspect.getsource(candidate.build_solar_data)
    assert "refine_solar_component_mask" not in auto_source
    assert "refine_solar_component_mask" in build_source


def test_build_solar_data_stores_refined_mask_not_raw_mask():
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[20:61, 20:61] = True
    raw[19, 19] = True
    raw[40, 40] = False
    gray[raw] = 220
    seed = (41, 40)

    expected = candidate.refine_solar_component_mask(raw)
    assert not np.array_equal(raw, expected)
    solar = candidate.build_solar_data(gray, 100, seed, raw)
    stored = candidate.decompress_full_mask(solar.component_mask, gray.shape)
    assert np.array_equal(stored, expected)


def test_roi_guard_and_contour_all_derive_from_same_refined_component():
    gray = np.zeros((101, 121), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    cv2.circle(raw.view(np.uint8), (60, 50), 24, 1, -1)
    raw[25, 35] = True
    gray[raw] = 220
    seed = (60, 50)
    refined = candidate.refine_solar_component_mask(raw)
    solar = candidate.build_solar_data(gray, 100, seed, raw)

    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5
    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        stored = candidate.decompress_full_mask(payload, gray.shape)
        region = candidate._build_observation_region(
            gray, refined, fraction * image_scale, seed
        )
        expected = np.zeros(gray.shape, bool)
        x0, y0, x1, y1 = region.bbox
        expected[y0:y1, x0:x1] = region.allowed_u8 != 0
        assert np.array_equal(stored, expected)

    contour = solar.component_contour
    assert np.all(refined[contour[:, 1], contour[:, 0]])
