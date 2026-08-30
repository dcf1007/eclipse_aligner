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
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.dump(node, include_attributes=False)


def _auto_state(threshold, seed):
    return {
        "settings": candidate.ImageSettings(),
        "auto_threshold_result": candidate.AutoThresholdResult(
            threshold=threshold,
            histogram_peak=200,
            histogram_left_edge=threshold,
            seed_threshold=threshold,
            coarse_threshold=threshold,
            roi_seed_threshold=threshold,
            full_seed_point=seed,
            used_guard=False,
            resolved=True,
        ),
        "solar_data": None,
    }


def test_production_uses_exact_tested_refinement_function():
    assert _function_ast(
        ROOT / "circle_arc_detector.py",
        "refine_solar_component_mask",
    ) == _function_ast(SNIPPET, "refine_solar_component_mask")
    snippet = _snippet_module()
    assert candidate.REFINEMENT_KERNEL_SIZE == snippet.REFINEMENT_KERNEL_SIZE == 7
    assert candidate.REFINEMENT_ITERATIONS == snippet.REFINEMENT_ITERATIONS == 1


def test_refinement_is_after_threshold_and_inside_current_t_solar_build():
    auto_source = inspect.getsource(candidate.auto_threshold)
    build_source = inspect.getsource(candidate.build_solar_data_at_threshold)

    assert "refine_solar_component_mask" not in auto_source
    assert "refine_solar_component_mask" in build_source
    assert build_source.index("seed_x") < build_source.index("cv2.compare")
    assert build_source.index("cv2.compare") < build_source.index("refine_solar_component_mask")
    assert build_source.index("refine_solar_component_mask") < build_source.index("SolarData(")
    assert build_source.index("SolarData(") < build_source.index('image_state["solar_data"]')


def test_build_returns_and_stores_same_refined_component():
    gray = np.zeros((81, 81), np.uint8)
    raw = np.zeros_like(gray, dtype=bool)
    raw[20:61, 20:61] = True
    raw[19, 19] = True
    raw[40, 40] = False
    gray[raw] = 220
    seed = (41, 40)
    state = _auto_state(100, seed)

    expected = candidate.refine_solar_component_mask(raw)
    assert not np.array_equal(raw, expected)

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    stored = candidate.decompress_full_mask(
        state["solar_data"].component_mask,
        gray.shape,
    )

    assert np.array_equal(refined, expected)
    assert np.array_equal(stored, refined)


def test_roi_guard_and_contour_derive_from_same_returned_refined_component():
    gray = np.zeros((101, 121), np.uint8)
    raw_u8 = np.zeros_like(gray, np.uint8)
    cv2.circle(raw_u8, (60, 50), 24, 255, -1)
    raw = raw_u8 != 0
    raw[25, 35] = True
    gray[raw] = 220
    seed = (60, 50)
    state = _auto_state(100, seed)

    refined = candidate.build_solar_data_at_threshold(gray, 100, state)
    solar = state["solar_data"]
    image_scale = (gray.shape[0] * gray.shape[1]) ** 0.5

    for payload, fraction in (
        (solar.roi_6_5_mask, candidate.ROI_DILATION_FRACTION),
        (solar.guard_19_5_mask, candidate.GUARD_DILATION_FRACTION),
    ):
        stored = candidate.decompress_full_mask(payload, gray.shape)
        region = candidate._build_observation_region(
            gray,
            refined,
            fraction * image_scale,
            seed,
        )
        expected = np.zeros(gray.shape, bool)
        x0, y0, x1, y1 = region.bbox
        expected[y0:y1, x0:x1] = region.allowed_u8 != 0
        assert np.array_equal(stored, expected)

    contour = solar.component_contour
    assert np.all(refined[contour[:, 1], contour[:, 0]])
