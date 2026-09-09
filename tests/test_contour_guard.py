import inspect
import math

import cv2
import numpy as np

import circle_arc_detector as cad


def test_zero_margin_preserves_component_exactly():
    component = np.zeros((31, 43), dtype=bool)
    component[6:25, 9:34] = True
    component[12:19, 16:27] = False

    guard = cad.dilate_component_mask(component, 0.0)

    assert guard.dtype == bool
    assert guard is not component
    assert np.array_equal(guard, component)


def test_guard_fills_external_polygon_and_ignores_internal_holes():
    component = np.zeros((81, 101), dtype=bool)
    cv2.circle(component.view(np.uint8), (50, 40), 22, 1, -1)
    cv2.circle(component.view(np.uint8), (50, 40), 9, 0, -1)

    guard = cad.dilate_component_mask(component, 8.0)

    # The guard is a filled outer envelope, not a topology-preserving dilation:
    # the original internal hole is intentionally valid guard territory.
    assert guard[40, 50]
    assert np.all(guard[component])
    assert guard[40, 20]
    assert guard[40, 80]


def test_guard_uses_shared_half_pixel_diagonal_raster_simplification():
    contour = np.array(
        [[2, 2], [3, 2], [4, 2], [5, 3], [5, 4], [5, 5], [4, 5], [3, 5], [2, 5], [2, 4], [2, 3]],
        dtype=np.int32,
    )
    expected = cv2.approxPolyDP(
        contour,
        math.hypot(0.5, 0.5),
        True,
    ).reshape(-1, 2)

    simplified = cad.simplify_raster_contour(contour)

    assert simplified.dtype == np.int32
    assert np.array_equal(simplified, expected)


def test_guard_construction_uses_no_distance_transform(monkeypatch):
    component = np.zeros((51, 67), dtype=bool)
    component[17:34, 23:44] = True

    def forbidden(*_args, **_kwargs):
        raise AssertionError("guard construction must not allocate a distance field")

    monkeypatch.setattr(cv2, "distanceTransform", forbidden)
    guard = cad.dilate_component_mask(component, 6.5)

    assert guard.dtype == bool
    assert guard.shape == component.shape
    assert np.all(guard[component])


def test_guard_draws_directly_into_final_bool_storage(monkeypatch):
    component = np.zeros((51, 67), dtype=bool)
    component[17:34, 23:44] = True
    seen = {}
    real_fill = cv2.fillPoly
    real_lines = cv2.polylines

    def fill(image, *args, **kwargs):
        seen["fill"] = image
        return real_fill(image, *args, **kwargs)

    def lines(image, *args, **kwargs):
        seen["lines"] = image
        return real_lines(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "fillPoly", fill)
    monkeypatch.setattr(cv2, "polylines", lines)
    guard = cad.dilate_component_mask(component, 6.5)

    assert seen["fill"].dtype == np.uint8
    assert seen["lines"] is seen["fill"]
    assert np.shares_memory(seen["fill"], guard)
    assert set(np.unique(seen["fill"])).issubset({0, 1})


def test_stage_b_profiles_share_guard_raster_simplifier():
    source = inspect.getsource(cad.sample_grayscale_profiles)
    assert "polygon = simplify_raster_contour(contour)" in source
    assert "cv2.approxPolyDP" not in source
