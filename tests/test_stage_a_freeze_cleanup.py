from pathlib import Path

import numpy as np
import pytest

import circle_arc_detector as cad


SOURCE = Path(cad.__file__).read_text(encoding="utf-8")


def _state():
    return {
        "settings": cad.ImageSettings(),
        "auto_threshold_result": None,
        "solar_data": None,
    }


def test_source_seed_is_checked_after_d7_not_before():
    gray = np.zeros((31, 31), dtype=np.uint8)
    gray[5:26, 5:26] = 30
    gray[15, 15] = 10  # dark at T10, but the agreed D7 CLOSE fills this one-pixel defect
    guard = np.zeros(gray.shape, dtype=bool)
    guard[2:29, 2:29] = True

    assert cad.find_lowest_full_res_threshold(gray, 10, (15, 15), guard) == 0


def test_source_seed_must_still_survive_d7_cleanup():
    gray = np.zeros((31, 31), dtype=np.uint8)
    gray[13:18, 13:18] = 30  # raw seed exists, but the 5x5 body is erased by D7 OPEN
    guard = np.zeros(gray.shape, dtype=bool)
    guard[2:29, 2:29] = True

    with pytest.raises(cad.ThresholdResolutionError, match="does not survive D7 cleanup"):
        cad.find_lowest_full_res_threshold(gray, 10, (15, 15), guard)


def test_find_auto_threshold_requires_authoritative_uint8_gray():
    gray16 = np.zeros((31, 31), dtype=np.uint16)
    with pytest.raises(ValueError, match="requires authoritative uint8 grayscale"):
        cad.find_auto_threshold(gray16, _state())


def test_source_support_mapping_uses_actual_work_kernel(monkeypatch):
    gray = np.zeros((120, 240), dtype=np.uint8)
    monkeypatch.setattr(cad, "WORK_RES_MAX_DIM", 120)
    monkeypatch.setattr(cad, "find_histogram_start_threshold", lambda _gray: 20)

    real_generate_kernel = cad.generate_kernel
    first_work_kernel = True

    def generated(size, round_kernel=False):
        nonlocal first_work_kernel
        if first_work_kernel and size == (5, 5) and not round_kernel:
            first_work_kernel = False
            return np.ones((7, 7), dtype=np.uint8)
        return real_generate_kernel(size, round_kernel=round_kernel)

    monkeypatch.setattr(cad, "generate_kernel", generated)

    def work_component(work_gray, _start_T, received_kernel):
        assert received_kernel.shape == (7, 7)
        component = np.zeros(work_gray.shape, dtype=bool)
        component[10:50, 30:90] = True
        return 12, component

    monkeypatch.setattr(cad, "find_work_res_solar_component", work_component)

    seen = {}

    def source_seed(_gray, _mask, source_kernel):
        seen["source_kernel_shape"] = source_kernel.shape
        return (120, 60)

    monkeypatch.setattr(cad, "brightest_supported_component_point", source_seed)
    monkeypatch.setattr(cad, "dilate_component_mask", lambda mask, _margin: np.ones(mask.shape, bool))
    monkeypatch.setattr(cad, "find_lowest_full_res_threshold", lambda _gray, start_T, _seed, _guard: start_T)
    monkeypatch.setattr(
        cad,
        "optimize_separated_threshold",
        lambda _gray, base_T, _seed, _guard: cad.ThresholdTopologySelection(
            threshold=base_T,
            base_threshold=base_T,
            delta=0,
            trajectory=(),
            net_quality=(),
            knee_curve=(),
        ),
    )

    assert cad.find_auto_threshold(gray, _state()) == 12
    # 7 work pixels at 2x source scale => 14; nearest-positive-odd tie chooses 13.
    assert seen["source_kernel_shape"] == (13, 13)


def test_work_failure_reports_actual_support_kernel_geometry():
    gray = np.zeros((21, 21), dtype=np.uint8)
    gray[9:12, 9:12] = 30
    with pytest.raises(cad.ThresholdResolutionError, match="7x7-supported"):
        cad.find_work_res_solar_component(
            gray,
            20,
            cad.generate_kernel((7, 7), round_kernel=False),
        )


def test_freeze_cleanup_removes_stale_helper_and_documents_stage_boundary():
    assert "def to_gray(" not in SOURCE
    assert "def opaque_bgra(" in SOURCE
    assert "D7-cleaned seeded component stays inside that fixed guard" in SOURCE
    assert "Stage A's proven full-resolution T, fixed seed, and fixed guard" in SOURCE
    assert "Stage A's proven full-resolution T, component, and seed" not in SOURCE


def test_retained_rebuild_and_validator_use_final_freeze_contract():
    chain = Path("snippets/apply_streamlined_autot.py").read_text(encoding="utf-8")
    validator = Path("snippets/validate_streamlined_autot.py").read_text(encoding="utf-8")

    assert "apply_stage_a_freeze_cleanup.py" in chain
    assert 'freezer["apply_freeze_cleanup"]' in chain
    assert "max(work_kernel.shape)" in validator
    assert "mapped_kernel_size = 5 *" not in validator
