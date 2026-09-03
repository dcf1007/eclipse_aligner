import importlib.util
from pathlib import Path

import cv2
import numpy as np

import circle_arc_detector as cad

SNIPPET = Path(__file__).parents[1] / "snippets" / "refine_solar_component_mask.py"
spec = importlib.util.spec_from_file_location("mask_refinement", SNIPPET)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
refine = module.refine_solar_component_mask


def test_refinement_snippet_uses_authoritative_production_implementation():
    assert refine is cad.refine_solar_component_mask
    expected = cad.generate_kernel((7, 7), round_kernel=True)
    assert expected.sum() == 29
    assert np.array_equal(module.SOLAR_COMPONENT_KERNEL, expected)


def test_rejects_empty_mask():
    try:
        refine(np.zeros((15, 15), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("empty component must be rejected")


def test_removes_small_outward_burr():
    mask = np.zeros((41, 41), dtype=bool)
    mask[10:31, 10:31] = True
    mask[19:22, 31:35] = True
    refined = refine(mask)
    assert not refined[20, 34]
    assert refined[20, 20]


def test_fills_small_interior_concavity():
    mask = np.zeros((41, 41), dtype=bool)
    mask[8:33, 8:33] = True
    mask[19:22, 19:22] = False
    refined = refine(mask)
    assert np.all(refined[19:22, 19:22])


def test_large_interior_survives():
    mask = np.zeros((61, 61), dtype=bool)
    cv2.circle(mask.view(np.uint8), (30, 30), 20, 1, -1)
    refined = refine(mask)
    assert refined[30, 30]
    assert np.count_nonzero(refined) > 0.97 * np.count_nonzero(mask)
