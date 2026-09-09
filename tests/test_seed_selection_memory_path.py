import inspect

import cv2
import numpy as np

import circle_arc_detector as cad


SQUARE_5 = cad.generate_kernel((5, 5), round_kernel=False)


def _supported_disk(dtype=bool):
    gray = np.zeros((51, 51), dtype=np.uint8)
    component_u8 = np.zeros((51, 51), dtype=np.uint8)
    cv2.circle(component_u8, (25, 25), 15, 255, -1)
    component = component_u8 != 0 if dtype == bool else component_u8
    return gray, component


def test_bool_component_seed_selection_does_not_expand_to_0255_copy(monkeypatch):
    gray, component = _supported_disk(bool)
    gray[25, 25] = 240

    # The optimized bool path must use the existing one-byte mask storage as a
    # zero-copy uint8 0/1 view. Calling the generic 0/255 converter here would
    # recreate a full-frame byte copy that this path intentionally avoids.
    monkeypatch.setattr(
        cad,
        "bool_mask_to_uint8",
        lambda *_: (_ for _ in ()).throw(AssertionError("bool mask copied")),
    )

    assert cad.brightest_supported_component_point(gray, component, SQUARE_5) == (
        25,
        25,
    )


def test_seed_selection_does_not_materialize_float_scores(monkeypatch):
    gray, component = _supported_disk(bool)
    gray[25, 20] = 240
    gray[25, 25] = 240

    # The final depth tie-break is a masked minMaxLoc on the distance field. A
    # separate np.where(..., distance, -1) float32 scores raster must never return.
    monkeypatch.setattr(
        cad.np,
        "where",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full-frame scores allocated")
        ),
    )

    assert cad.brightest_supported_component_point(gray, component, SQUARE_5) == (
        25,
        25,
    )


def test_seed_selection_preserves_row_major_tie_break_for_equal_depth():
    gray, component = _supported_disk(bool)

    # These two pixels are symmetric about the disk center, so they have the same
    # grayscale and the same distance-to-background. The historical np.argmax path
    # chose the first location in row-major order; OpenCV's masked minMaxLoc must
    # preserve that observable tie behavior.
    gray[25, 20] = 240
    gray[25, 30] = 240

    assert cad.brightest_supported_component_point(gray, component, SQUARE_5) == (
        20,
        25,
    )


def test_seed_selection_bool_and_uint8_components_are_equivalent():
    gray, component_bool = _supported_disk(bool)
    component_u8 = component_bool.astype(np.uint8) * 255
    gray[20, 25] = 241
    gray[25, 25] = 241

    assert cad.brightest_supported_component_point(
        gray,
        component_bool,
        SQUARE_5,
    ) == cad.brightest_supported_component_point(
        gray,
        component_u8,
        SQUARE_5,
    )


def test_seed_selection_source_documents_storage_reuse_and_masked_ranking():
    source = inspect.getsource(cad.brightest_supported_component_point)

    # Keep the memory contract visible in the implementation rather than allowing
    # future cleanup to silently reintroduce the removed full-frame temporaries.
    assert "component.view(np.uint8)" in source
    assert "supported.view(np.bool_)" in source
    assert source.count("cv2.minMaxLoc") == 2
    assert "scores =" not in source
    assert "max_gray = int(gray[supported].max())" not in source
