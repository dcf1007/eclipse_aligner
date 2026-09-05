"""Profile the simplified Stage-B edge descriptor on selected real frames."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from benchmark_stageb_simplified_score import CASES, load_authoritative_gray8, load_module


def cleaned_component_at(cad, gray: np.ndarray, threshold: int, result):
    guard = cad.decompress_array(result.full_res_separation_guard_mask)
    cleaned = cv2.compare(gray, threshold, cv2.CMP_GT)
    for kernel in cad.SOLAR_CLEANUP_KERNELS:
        cleaned = cad.morphological_cleanup(cleaned, kernel)
    cleaned[~guard] = 0
    component = cad.extract_component(cleaned, result.full_res_seed_point)
    if component is None:
        raise RuntimeError("no component")
    return component


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--images", default="0165,0200,0117")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    cad = load_module(args.source)
    wanted = {v.strip().zfill(4) for v in args.images.split(",") if v.strip()}

    for image_id, category, filename, _, expected_t in CASES:
        if image_id not in wanted:
            continue
        gray = load_authoritative_gray8(args.inputs / category / filename)
        result = cad.AutoThresholdResult()
        cad.find_separation_threshold(gray, result)
        component = cleaned_component_at(cad, gray, expected_t, result)
        contour = cad.find_external_contour(component)

        # Warm up OpenCV allocations/caches.
        cad.measure_edge_alignment(gray, contour)
        sample_times = []
        edge_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            profiles, lengths = cad._sample_grayscale_profiles(gray, contour)
            sample_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            cad.measure_edge_alignment(gray, contour)
            edge_times.append(time.perf_counter() - start)
        print(
            f"{image_id}: contour={len(contour)} profiles={len(profiles)} "
            f"samples={int(np.ceil(lengths).sum())} "
            f"sample={np.median(sample_times)*1000:.2f}ms "
            f"edge_total={np.median(edge_times)*1000:.2f}ms "
            f"edge_after_sample~={(np.median(edge_times)-np.median(sample_times))*1000:.2f}ms"
        )


if __name__ == "__main__":
    main()
