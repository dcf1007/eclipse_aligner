"""Verify mathematically identical edge-implementation optimizations on real candidates."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_stageb_simplified_score import CASES, load_authoritative_gray8, load_module


def cleaned_component_at(cad, gray: np.ndarray, threshold: int, result):
    guard = cad.decompress_array(result.full_res_separation_guard_mask)
    cleaned = cv2.compare(gray, threshold, cv2.CMP_GT)
    for kernel in cad.SOLAR_CLEANUP_KERNELS:
        cleaned = cad.morphological_cleanup(cleaned, kernel)
    cleaned[~guard] = 0
    return cad.extract_component(cleaned, result.full_res_seed_point)


def same_float(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--optimized-source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", default="")
    args = parser.parse_args()

    baseline = load_module(args.baseline_source)
    optimized = load_module(args.optimized_source)
    rows = []
    wanted = {v.strip().zfill(4) for v in args.images.split(",") if v.strip()}

    for image_id, category, filename, _, _ in CASES:
        if wanted and image_id not in wanted:
            continue
        gray = load_authoritative_gray8(args.inputs / category / filename)
        result = optimized.AutoThresholdResult()
        optimized.find_separation_threshold(gray, result)
        base_t = result.full_res_separation_threshold
        if base_t is None:
            raise RuntimeError(f"{image_id}: no Stage-A threshold")

        for threshold in range(base_t, base_t + optimized.MAX_T_REFINEMENT_STEPS + 1):
            component = cleaned_component_at(optimized, gray, threshold, result)
            if component is None:
                continue
            contour = optimized.find_external_contour(component)
            old_distance, old_reliability = baseline.measure_edge_alignment(gray, contour)
            new_distance, new_reliability = optimized.measure_edge_alignment(gray, contour)
            row = {
                "image": image_id,
                "T": threshold,
                "baseline_distance": old_distance,
                "optimized_distance": new_distance,
                "distance_equal": same_float(old_distance, new_distance),
                "baseline_reliability": old_reliability,
                "optimized_reliability": new_reliability,
                "reliability_equal": same_float(old_reliability, new_reliability),
            }
            rows.append(row)
            if not row["distance_equal"] or not row["reliability_equal"]:
                raise SystemExit(f"optimization changed edge measurement: {row}")
        print(f"{image_id}: edge candidates equivalent", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"verified {len(rows)} candidate edge measurements")


if __name__ == "__main__":
    main()
