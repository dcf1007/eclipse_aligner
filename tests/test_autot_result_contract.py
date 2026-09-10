import cv2
import numpy as np
import pytest

import circle_arc_detector as cad


SEPARATION_FIELDS = (
    "histogram_start_threshold",
    "work_res_seed_point",
    "work_res_separation_threshold",
    "work_res_separation_component_mask",
    "full_res_seed_point",
    "full_res_separation_threshold",
    "full_res_separation_component_mask",
    "full_res_separation_guard_mask",
)
REFINEMENT_FIELDS = (
    "full_res_refined_threshold",
    "full_res_refined_component_mask",
    "full_res_refined_component_contour",
)


def _disk_gray(size=81):
    gray = np.zeros((size, size), np.uint8)
    center = size // 2
    cv2.circle(gray, (center, center), 20, 220, -1)
    gray[center, center] = 240
    return gray


def _structurally_complete_result() -> cad.AutoThresholdResult:
    return cad.AutoThresholdResult(
        histogram_start_threshold=0,
        work_res_seed_point=(0, 0),
        work_res_separation_threshold=0,
        work_res_separation_component_mask=b"work",
        full_res_seed_point=(0, 0),
        full_res_separation_threshold=0,
        full_res_separation_component_mask=b"separation",
        full_res_separation_guard_mask=b"guard",
        full_res_refined_threshold=0,
        full_res_refined_component_mask=b"refined",
        full_res_refined_component_contour=b"contour",
    )


def test_autot_completeness_properties_are_structural_and_zero_is_populated():
    result = cad.AutoThresholdResult()
    assert not result.separation_threshold_complete
    assert not result.threshold_refinement_complete

    complete = _structurally_complete_result()
    assert complete.separation_threshold_complete
    assert complete.threshold_refinement_complete

    for field in SEPARATION_FIELDS:
        candidate = _structurally_complete_result()
        setattr(candidate, field, None)
        assert not candidate.separation_threshold_complete, field

    for field in REFINEMENT_FIELDS:
        candidate = _structurally_complete_result()
        setattr(candidate, field, None)
        assert not candidate.threshold_refinement_complete, field


def test_stage_a_populates_progressive_result_and_completeness():
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(_disk_gray(), {"auto_threshold_result": result})

    assert result.failure_reason is None
    assert result.separation_threshold_complete
    assert not result.threshold_refinement_complete


def test_stage_b_mutates_same_result_and_returns_final_t():
    gray = _disk_gray()
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(gray, {"auto_threshold_result": result})
    identity = id(result)

    final_t = cad.refine_threshold(gray, {"auto_threshold_result": result})

    assert id(result) == identity
    assert final_t == result.full_res_refined_threshold
    assert result.separation_threshold_complete
    assert result.threshold_refinement_complete
    stored_contour = cad.decompress_contour(result.full_res_refined_component_contour)
    assert stored_contour.dtype == np.int32
    assert stored_contour.ndim == 2
    assert stored_contour.shape[1] == 2

    winning_component = cad.decompress_image(result.full_res_refined_component_mask)
    expected_contour = cad.find_external_contour(winning_component)
    assert np.array_equal(stored_contour, expected_contour)


def test_complete_result_is_reused_by_find_auto_threshold_without_rerun(monkeypatch):
    gray = _disk_gray()
    state = {"auto_threshold_result": None}
    selected = cad.find_auto_threshold(gray, state)
    result = state["auto_threshold_result"]
    before = vars(result).copy()

    monkeypatch.setattr(
        cad,
        "find_separation_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage A reran")),
    )
    monkeypatch.setattr(
        cad,
        "refine_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage B reran")),
    )
    assert cad.find_auto_threshold(gray, state) == selected
    assert vars(result) == before


def test_incomplete_nonfailed_stage_a_restarts_whole_autot_on_same_object():
    gray = _disk_gray()
    result = _structurally_complete_result()
    result.full_res_separation_component_mask = None
    identity = id(result)

    cad.find_separation_threshold(gray, {"auto_threshold_result": result})

    assert id(result) == identity
    assert result.failure_reason is None
    assert result.separation_threshold_complete
    assert not result.threshold_refinement_complete
    assert result.full_res_refined_threshold is None
    assert result.full_res_refined_component_mask is None
    assert result.full_res_refined_component_contour is None


def test_stage_a_domain_failure_is_registered_and_not_propagated(monkeypatch):
    result = cad.AutoThresholdResult()
    monkeypatch.setattr(
        cad,
        "find_work_res_separation_threshold",
        lambda *_: (_ for _ in ()).throw(
            cad.ThresholdResolutionError("no component")
        ),
    )

    returned = cad.find_separation_threshold(_disk_gray(), {"auto_threshold_result": result})

    assert returned is None
    assert result.failure_reason == "work-resolution separation: no component"
    assert not result.separation_threshold_complete


def test_existing_failed_result_returns_unchanged_without_retry(monkeypatch):
    result = cad.AutoThresholdResult(failure_reason="existing failure")
    state = {"auto_threshold_result": result}
    before = vars(result).copy()
    monkeypatch.setattr(
        cad,
        "find_separation_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Stage A retried")),
    )
    monkeypatch.setattr(
        cad,
        "refine_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Stage B retried")),
    )
    assert cad.find_auto_threshold(_disk_gray(), state) is None
    assert vars(result) == before


def test_find_auto_threshold_calls_stage_a_for_nonfailed_incomplete_separation(monkeypatch):
    gray = _disk_gray()
    result = cad.AutoThresholdResult()
    state = {"auto_threshold_result": result}
    real_stage_a = cad.find_separation_threshold
    calls = []

    def counted_stage_a(source, target_state):
        calls.append(id(target_state["auto_threshold_result"]))
        return real_stage_a(source, target_state)

    monkeypatch.setattr(cad, "find_separation_threshold", counted_stage_a)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))
    selected = cad.find_auto_threshold(gray, state)

    assert calls == [id(result)]
    assert selected is not None
    assert result.separation_threshold_complete
    assert result.threshold_refinement_complete


def test_find_auto_threshold_rejects_impossible_normal_stage_a_return(monkeypatch):
    result = cad.AutoThresholdResult()
    state = {"auto_threshold_result": result}
    monkeypatch.setattr(cad, "find_separation_threshold", lambda *_: None)

    with pytest.raises(
        ValueError,
        match="separation threshold remained incomplete without a recorded failure",
    ):
        cad.find_auto_threshold(_disk_gray(), state)


def test_find_auto_threshold_failed_result_returns_none_without_stage_retry(monkeypatch):
    result = cad.AutoThresholdResult(failure_reason="failed")
    state = {"auto_threshold_result": result}
    monkeypatch.setattr(
        cad,
        "find_separation_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Auto-T retried")),
    )
    monkeypatch.setattr(
        cad,
        "refine_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("failed Auto-T refined")),
    )

    assert cad.find_auto_threshold(_disk_gray(), state) is None
    assert result.failure_reason == "failed"


def test_find_auto_threshold_complete_result_reuses_stored_threshold_without_processing(monkeypatch):
    result = _structurally_complete_result()
    result.full_res_refined_threshold = 17
    state = {"auto_threshold_result": result}
    monkeypatch.setattr(
        cad,
        "find_separation_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage A reran")),
    )
    monkeypatch.setattr(
        cad,
        "refine_threshold",
        lambda *_: (_ for _ in ()).throw(AssertionError("Stage B reran")),
    )

    assert cad.find_auto_threshold(_disk_gray(), state) == 17


def test_stage_b_partial_result_is_cleared_before_refinement_retry(monkeypatch):
    gray = _disk_gray()
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(gray, {"auto_threshold_result": result})
    guard_payload = result.full_res_separation_guard_mask
    result.full_res_refined_threshold = 123
    result.full_res_refined_component_mask = b"stale"
    result.full_res_refined_component_contour = None
    real_decompress = cad.decompress_image
    inspected = []

    def inspect_refinement_reset(payload):
        if payload == guard_payload and not inspected:
            inspected.append(True)
            assert result.full_res_refined_threshold is None
            assert result.full_res_refined_component_mask is None
            assert result.full_res_refined_component_contour is None
        return real_decompress(payload)

    monkeypatch.setattr(cad, "decompress_image", inspect_refinement_reset)
    monkeypatch.setattr(cad, "measure_edge_alignment", lambda *_: (0.5, 1.0))

    selected = cad.refine_threshold(gray, {"auto_threshold_result": result})

    assert inspected == [True]
    assert selected is not None
    assert result.threshold_refinement_complete


def test_stage_b_domain_failure_is_registered_and_not_propagated(monkeypatch):
    gray = _disk_gray()
    result = cad.AutoThresholdResult()
    cad.find_separation_threshold(gray, {"auto_threshold_result": result})
    monkeypatch.setattr(
        cad,
        "find_guard_boundary_indices",
        lambda *_: (_ for _ in ()).throw(cad.ThresholdResolutionError("bad guard")),
    )

    selected = cad.refine_threshold(gray, {"auto_threshold_result": result})

    assert selected is None
    assert result.failure_reason == "fine refinement: bad guard"
    assert not result.threshold_refinement_complete


def test_histogram_without_local_peak_uses_global_peak_fallback(monkeypatch):
    signal = np.ones(256, dtype=np.float64)
    signal[0] = 0.0
    monkeypatch.setattr(cad.np, "convolve", lambda *_args, **_kwargs: signal)

    assert int(np.argmax(signal)) == 1
    assert cad.find_histogram_start_threshold(np.zeros((1, 1), np.uint8)) == 1


def test_histogram_peak_without_preceding_valley_falls_back_to_peak_not_zero():
    values = np.repeat(np.arange(256, dtype=np.uint8), np.arange(1, 257))
    gray = values.reshape(1, -1)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    signal = np.convolve(histogram, cad.PEAK_KERNEL, mode="same")
    expected_peak = 254

    assert int(np.argmax(signal)) == expected_peak
    assert cad.find_histogram_start_threshold(gray) == expected_peak


def test_autot_helper_contract_errors_remain_value_errors():
    with pytest.raises(ValueError, match="empty solar component"):
        cad.dilate_component_mask(np.zeros((5, 5), bool), 1.0)

    gray = np.zeros((5, 5), np.uint8)
    with pytest.raises(ValueError, match="guard is empty"):
        cad.find_full_res_separation_threshold(gray, 0, (2, 2), np.zeros_like(gray, bool))
    with pytest.raises(ValueError, match="outside the image"):
        cad.find_full_res_separation_threshold(gray, 0, (9, 9), np.ones_like(gray, bool))

    guard = np.ones_like(gray, bool)
    guard[2, 2] = False
    with pytest.raises(ValueError, match="outside the Auto-T guard"):
        cad.find_full_res_separation_threshold(gray, 0, (2, 2), guard)

    with pytest.raises(ValueError, match="empty or not two-dimensional"):
        cad.find_external_contour(np.zeros((5, 5), bool))
    degenerate = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match="perimeter is empty"):
        cad.measure_hole_quality(degenerate, component_area=1, filled_area=1)
