import importlib.util
from pathlib import Path

import numpy as np

SNIPPET = Path(__file__).parents[1] / "snippets" / "refine_solar_component_mask.py"
spec = importlib.util.spec_from_file_location("mask_refinement", SNIPPET)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
refine = module.refine_solar_component_mask


def test_rejects_empty_mask():
    try:
        refine(np.zeros((9, 9), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("empty component must be rejected")


def test_removes_one_pixel_diagonal_burr():
    mask = np.zeros((21, 21), dtype=bool)
    mask[5:16, 5:16] = True
    # An 8-connected diagonal one-pixel burr is unsupported by the 3x3
    # elliptical (cross-shaped) structuring element and should disappear.
    mask[4, 4] = True
    refined = refine(mask)
    assert not refined[4, 4]
    assert refined[10, 10]


def test_fills_single_pixel_hole_after_cleanup():
    mask = np.zeros((21, 21), dtype=bool)
    mask[5:16, 5:16] = True
    mask[10, 10] = False
    refined = refine(mask)
    assert refined[10, 10]


def test_large_interior_survives():
    mask = np.zeros((31, 31), dtype=bool)
    mask[6:25, 6:25] = True
    refined = refine(mask)
    assert refined[15, 15]
    assert np.count_nonzero(refined) > 0.95 * np.count_nonzero(mask)
