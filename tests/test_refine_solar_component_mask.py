import importlib.util
from pathlib import Path

import cv2
import numpy as np

SNIPPET = Path(__file__).parents[1] / "snippets" / "refine_solar_component_mask.py"
spec = importlib.util.spec_from_file_location("mask_refinement", SNIPPET)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
refine = module.refine_solar_component_mask


def test_kernel_contract_is_approved_7x7_single_pass():
    assert module.REFINEMENT_KERNEL_SIZE == 7
    assert module.REFINEMENT_ITERATIONS == 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    assert kernel.shape == (7, 7)


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
