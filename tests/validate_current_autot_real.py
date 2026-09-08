"""Manual real-frame regression for the current authoritative Auto-T/SolarData contract.

Requires a directory containing the 21 established eclipse regression frames. The
expected winners are the previously validated production winners for the current
Auto-T algorithm; this script does not derive or tune them.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import circle_arc_detector as cad

EXPECTED_THRESHOLDS = {
    "0018": 4,
    "0049": 5,
    "0070": 6,
    "0100": 3,
    "0112": 6,
    "0113": 24,
    "0114": 22,
    "0115": 25,
    "0116": 24,
    "0117": 28,
    "0118": 50,
    "0121": 6,
    "0132": 11,
    "0143": 10,
    "0154": 17,
    "0165": 9,
    "0173": 9,
    "0180": 17,
    "0190": 21,
    "0200": 24,
    "0203": 26,
}


def authoritative_gray(path: Path) -> np.ndarray:
    """Reproduce the GUI's lossless-master-to-authoritative-gray conversion."""
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"Could not read {path}")
    if source.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"unsupported source dtype: {source.dtype}")
    if source.ndim == 2:
        master = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    elif source.ndim == 3 and source.shape[2] == 3:
        master = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    elif source.ndim == 3 and source.shape[2] == 4:
        master = source
    else:
        raise ValueError(f"unsupported source shape: {source.shape}")
    if master.dtype == np.uint8:
        master = master.astype(np.uint16)
        master *= 257
    gray16 = cv2.cvtColor(np.ascontiguousarray(master, dtype=np.uint16), cv2.COLOR_BGRA2GRAY)
    return cv2.convertScaleAbs(gray16, alpha=1.0 / 257.0)


def frame_for_prefix(corpus: Path, prefix: str) -> Path:
    matches = sorted(path for path in corpus.iterdir() if path.is_file() and path.name.startswith(prefix + "__"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one frame starting {prefix}__ in {corpus}, found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    passed = True
    for prefix, expected_threshold in EXPECTED_THRESHOLDS.items():
        path = frame_for_prefix(args.corpus, prefix)
        gray = authoritative_gray(path)
        state = {
            "settings": cad.ImageSettings(),
            "auto_threshold_result": None,
            "solar_data": None,
        }

        selected = cad.find_auto_threshold(gray, state)
        result = state["auto_threshold_result"]
        auto_ok = (
            isinstance(result, cad.AutoThresholdResult)
            and result.failure_reason is None
            and result.separation_threshold_complete
            and result.threshold_refinement_complete
            and selected == expected_threshold
            and result.full_res_refined_threshold == expected_threshold
        )

        component = cad.resolve_threshold(gray, expected_threshold, state)
        solar = state["solar_data"]
        resolver_ok = (
            isinstance(solar, cad.SolarData)
            and solar.threshold == expected_threshold
            and solar.seed_point == result.full_res_seed_point
            and solar.component_mask == result.full_res_refined_component_mask
            and solar.guard_mask == result.full_res_separation_guard_mask
            and solar.component_contour == result.full_res_refined_component_contour
            and np.array_equal(component, cad.decompress_image(solar.component_mask))
        )

        ok = auto_ok and resolver_ok
        passed &= ok
        print(
            path.name,
            "PASS" if ok else "FAIL",
            f"expected={expected_threshold}",
            f"actual={selected}",
            f"failure={result.failure_reason!r}",
            flush=True,
        )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
