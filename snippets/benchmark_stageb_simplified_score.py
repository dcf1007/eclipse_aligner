"""Direct real-image regression for the simplified Stage-B descriptor score.

The script loads the authoritative source-image grayscale exactly as the application
load path does, runs Stage A and production ``refine_threshold()`` without cached
measurements, and records the selected T plus the winning component-mask hash.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np


CASES = (
    ("0018", "before", "0018__DSC0460 (2).jpg", 7, 4),
    ("0049", "before", "0049__DSC0748 (2).jpg", 5, 5),
    ("0070", "before", "0070__DSC0938 (2).jpg", 8, 6),
    ("0100", "before", "0100__DSC0212.jpg", 2, 3),
    ("0112", "before", "0112__DSC0322.jpg", 5, 6),
    ("0121", "after", "0121__DSC0358.jpg", 6, 6),
    ("0132", "after", "0132__DSC0456.jpg", 7, 11),
    ("0143", "after", "0143__DSC0556.jpg", 6, 10),
    ("0154", "after", "0154__DSC0660.jpg", 14, 17),
    ("0165", "after", "0165__DSC0760.jpg", 10, 9),
    ("0173", "horizon", "0173__DSC0834.jpg", 5, 9),
    ("0180", "horizon", "0180__DSC0847.jpg", 17, 17),
    ("0190", "horizon", "0190__DSC0871.jpg", 19, 21),
    ("0200", "horizon", "0200__DSC0891.jpg", 24, 24),
    ("0203", "horizon", "0203__DSC0894.jpg", 26, 26),
    ("0113", "totality", "0113__DSC0343.jpg", 24, 24),
    ("0114", "totality", "0114__DSC0344.jpg", 22, 22),
    ("0115", "totality", "0115__DSC0345.jpg", 26, 25),
    ("0116", "totality", "0116__DSC0346.jpg", 24, 24),
    ("0117", "totality", "0117__DSC0347.jpg", 29, 28),
    ("0118", "totality", "0118__DSC0348.jpg", 50, 50),
)


def load_module(source: Path):
    spec = importlib.util.spec_from_file_location("cad_stageb_regression", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cad_stageb_regression"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_authoritative_gray8(path: Path) -> np.ndarray:
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"cannot load {path}")
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
    master16 = (
        master.astype(np.uint16) * 257
        if master.dtype == np.uint8
        else np.ascontiguousarray(master, dtype=np.uint16)
    )
    gray16 = cv2.cvtColor(master16, cv2.COLOR_BGRA2GRAY)
    return ((gray16.astype(np.uint32) + 128) // 257).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--images",
        default="",
        help="comma-separated image IDs; default runs all 21 cases",
    )
    args = parser.parse_args()

    cad = load_module(args.source)
    selected = {value.strip().zfill(4) for value in args.images.split(",") if value.strip()}
    cases = [case for case in CASES if not selected or case[0] in selected]
    rows = []
    for image_id, category, filename, previous_t, expected_t in cases:
        gray = load_authoritative_gray8(args.inputs / category / filename)
        started = time.perf_counter()
        result = cad.AutoThresholdResult()
        cad.find_separation_threshold(gray, result)
        selected_t = cad.refine_threshold(gray, result)
        mask = cad.decompress_array(result.full_res_refined_component_mask)
        mask_sha256 = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        row = {
            "image": image_id,
            "category": category,
            "base_T": result.full_res_separation_threshold,
            "previous_T": previous_t,
            "expected_simplified_T": expected_t,
            "selected_T": selected_t,
            "matches_expected": selected_t == expected_t,
            "mask_area": int(np.count_nonzero(mask)),
            "mask_sha256": mask_sha256,
            "seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(
            f"{image_id}: base={row['base_T']} previous={previous_t} "
            f"selected={selected_t} expected={expected_t} "
            f"match={row['matches_expected']} {row['seconds']:.1f}s",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if not all(row["matches_expected"] for row in rows):
        raise SystemExit("one or more Stage-B regression cases changed")


if __name__ == "__main__":
    main()
