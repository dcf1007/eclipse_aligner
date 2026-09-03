from pathlib import Path

import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


SOURCE = Path(cad.__file__).read_text(encoding="utf-8")


def test_resize_same_size_is_an_exact_copy_without_opencv(monkeypatch):
    image = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)

    def fail_resize(*_args, **_kwargs):
        raise AssertionError("same-size resize should not call OpenCV")

    monkeypatch.setattr(cad.cv2, "resize", fail_resize)
    resized = cad.resize_img(image, (7, 6))
    assert np.array_equal(resized, image)
    assert resized is not image


def test_resize_uses_area_when_only_height_shrinks(monkeypatch):
    image = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)
    seen = {}

    def record_resize(source, size, interpolation):
        seen["size"] = size
        seen["interpolation"] = interpolation
        return np.zeros((size[1], size[0]), dtype=source.dtype)

    monkeypatch.setattr(cad.cv2, "resize", record_resize)
    cad.resize_img(image, (7, 3))
    assert seen == {
        "size": (7, 3),
        "interpolation": cv2.INTER_AREA,
    }


def test_supported_seed_requires_full_kernel_inside_raster_boundary():
    gray = np.zeros((5, 5), dtype=np.uint8)
    gray[0, 0] = 255
    gray[2, 2] = 200
    component = np.ones((5, 5), dtype=bool)
    seed = cad.brightest_supported_component_point(
        gray,
        component,
        cad.generate_kernel((5, 5), round_kernel=False),
    )
    assert seed == (2, 2)


def test_stage_b_threshold_resolution_error_is_not_classified_as_stage_a_failure(monkeypatch):
    gray = np.zeros((31, 31), dtype=np.uint8)
    state = {
        "settings": cad.ImageSettings(),
        "auto_threshold_result": None,
        "solar_data": None,
    }

    def histogram_start(_gray):
        return 10

    def work_component(work_gray, _start_T, _kernel):
        component = np.zeros(work_gray.shape, dtype=bool)
        component[8:23, 8:23] = True
        return 7, component

    def source_seed(_gray, _component, _kernel):
        return (15, 15)

    def guard(component, _margin):
        return np.ones(component.shape, dtype=bool)

    def source_boundary(_gray, _start_T, _seed, _guard):
        return 6

    def fail_stage_b(*_args, **_kwargs):
        raise cad.ThresholdResolutionError("stage-b sentinel")

    monkeypatch.setattr(cad, "find_histogram_start_threshold", histogram_start)
    monkeypatch.setattr(cad, "find_work_res_solar_component", work_component)
    monkeypatch.setattr(cad, "brightest_supported_component_point", source_seed)
    monkeypatch.setattr(cad, "dilate_component_mask", guard)
    monkeypatch.setattr(cad, "find_lowest_full_res_threshold", source_boundary)
    monkeypatch.setattr(cad, "optimize_separated_threshold", fail_stage_b)

    with pytest.raises(cad.ThresholdResolutionError, match="stage-b sentinel"):
        cad.find_auto_threshold(gray, state)
    assert state["auto_threshold_result"] is None


def test_source_is_ordered_by_processing_stage():
    generic = SOURCE.index("# Generic image and kernel utilities")
    settings = SOURCE.index("# Per-image processing settings")
    stage_a = SOURCE.index("# Auto-T Stage A: source separation")
    find_auto = SOURCE.index("def find_auto_threshold(")
    stage_b = SOURCE.index("# Auto-T Stage B: threshold optimization")
    final_t = SOURCE.index("# Final-T full-resolution solar resolution and persistence")
    gui = SOURCE.index("class DetectorApp:")
    assert generic < settings < stage_a < find_auto < stage_b < final_t < gui

    stage_a_text = SOURCE[stage_a:stage_b]
    stage_b_text = SOURCE[stage_b:final_t]
    final_text = SOURCE[final_t:gui]
    assert "AUTO_T_GUARD_DILATION_FRACTION" in stage_a_text
    assert "TOPOLOGY_OPTIMIZATION_STEPS" not in stage_a_text
    assert "TOPOLOGY_OPTIMIZATION_STEPS" in stage_b_text
    assert "def _component_descriptor(" in stage_b_text
    assert "ROI_DILATION_FRACTION" not in stage_a_text
    assert "ROI_DILATION_FRACTION" in final_text
    assert "SOLAR_COMPONENT_KERNEL" in final_text


def test_current_documentation_and_retained_validation_are_not_stale():
    assert "7x7 elliptical OPEN/CLOSE" not in SOURCE
    assert "fixed 10% L2-distance" in SOURCE

    validator = Path("snippets/validate_streamlined_autot.py").read_text(encoding="utf-8")
    for obsolete in (
        "WORK_MAX_DIM",
        "TRACKING_SEED_KERNEL_SIZE",
        "max(1, round(",
        "/ float(max(",
    ):
        assert obsolete not in validator
    assert "WORK_RES_MAX_DIM" in validator
    assert "nearest_positive_odd" in validator


def test_retained_rebuild_chain_applies_final_stage_a_cleanup():
    streamlined = Path("snippets/apply_streamlined_autot.py").read_text(encoding="utf-8")
    assert "apply_stage_a_final_cleanup.py" in streamlined
    assert 'finalizer["finalize_stage_a_source"]' in streamlined
