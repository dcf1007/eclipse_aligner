import ast
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
RETAINED_PATCH = ROOT / "snippets" / "apply_streamlined_autot.py"


def _legacy_cropped_dilation(component: np.ndarray, margin: float) -> np.ndarray:
    ys, xs = np.nonzero(component)
    height, width = component.shape
    padding = math.ceil(margin)
    x0 = max(0, int(xs.min()) - padding - 2)
    x1 = min(width, int(xs.max()) + padding + 3)
    y0 = max(0, int(ys.min()) - padding - 2)
    y1 = min(height, int(ys.max()) + padding + 3)
    crop = component[y0:y1, x0:x1]
    outside = np.where(crop, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    result = np.zeros(component.shape, dtype=bool)
    result[y0:y1, x0:x1] = distance <= margin
    return result


def test_nearest_positive_odd_preserves_lower_tie_rule():
    assert cad.nearest_positive_odd(23.9) == 23
    assert cad.nearest_positive_odd(24.0) == 23
    assert cad.nearest_positive_odd(24.1) == 25
    assert cad.nearest_positive_odd(25.0) == 25
    with pytest.raises(ValueError):
        cad.nearest_positive_odd(0)


def test_uint16_binary_resize_preserves_exact_values_when_declared_mask():
    mask = np.zeros((5, 7), dtype=np.uint16)
    mask[1:4, 2:6] = 65535
    resized = cad.resize_img(mask, (17, 13), mask=True)
    assert resized.dtype == np.uint16
    assert set(np.unique(resized)) == {0, 65535}


def test_full_frame_distance_dilation_matches_previous_cropped_semantics():
    component = np.zeros((91, 123), dtype=bool)
    component[31:58, 47:76] = True
    component[42:49, 76:83] = True
    for margin in (0.0, 1.0, 6.5, 17.25):
        expected = _legacy_cropped_dilation(component, margin)
        actual = cad.dilate_component_mask(component, margin)
        assert np.array_equal(actual, expected)


def test_component_dilation_rejects_negative_margin_instead_of_repairing_it():
    component = np.zeros((11, 11), dtype=bool)
    component[5, 5] = True
    with pytest.raises(ValueError):
        cad.dilate_component_mask(component, -1)


def test_topology_optimizer_has_no_silent_fallback(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("descriptor invariant failed")

    monkeypatch.setattr(cad, "topology_trajectory_from_separation_threshold", fail)
    with pytest.raises(ValueError, match="descriptor invariant failed"):
        cad.optimize_separated_threshold(
            np.ones((9, 9), dtype=np.uint8),
            3,
            (4, 4),
            np.ones((9, 9), dtype=bool),
        )


def test_production_contains_no_lambda_or_obsolete_kernel_size_constants():
    source = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Lambda) for node in ast.walk(tree))
    for name in (
        "WORK_MAX_DIM",
        "TRACKING_SEED_KERNEL_SIZE",
        "SOLAR_COMPONENT_KERNEL_SIZE",
        "CLEANUP_KERNEL_SIZES",
        "REFINEMENT_ITERATIONS",
    ):
        assert name not in source
    assert "component_crop" not in source
    assert "max(1, round(" not in source


def test_final_refinement_kernel_uses_shared_euclidean_generator():
    expected = cad.generate_kernel((7, 7), round_kernel=True)
    assert expected.sum() == 29
    assert np.array_equal(cad.SOLAR_COMPONENT_KERNEL, expected)


def test_retained_streamlined_patch_delegates_current_cleanup():
    text = RETAINED_PATCH.read_text(encoding="utf-8")
    assert "apply_readability_cleanup.py" in text
    assert "apply_production_cleanup" in text
