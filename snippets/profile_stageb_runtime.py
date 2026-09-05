"""Compare Stage-B runtime between two detector sources on real images.

Both modules run their own Stage A before timing, so the measured interval is only
``refine_threshold()`` and no cached candidate measurements are reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_stageb_simplified_score import CASES, load_authoritative_gray8, load_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--images", required=True)
    args = parser.parse_args()

    baseline = load_module(args.baseline_source)
    candidate = load_module(args.candidate_source)
    wanted = {v.strip().zfill(4) for v in args.images.split(",") if v.strip()}

    for image_id, category, filename, _, _ in CASES:
        if image_id not in wanted:
            continue
        gray = load_authoritative_gray8(args.inputs / category / filename)
        row = [image_id]
        for module in (baseline, candidate):
            result = module.AutoThresholdResult()
            module.find_separation_threshold(gray, result)
            started = time.perf_counter()
            selected = module.refine_threshold(gray, result)
            row.extend((selected, time.perf_counter() - started))
        print(
            f"{row[0]} baseline=T{row[1]} {row[2]:.3f}s "
            f"candidate=T{row[3]} {row[4]:.3f}s "
            f"ratio={row[4]/row[2]:.3f}"
        )


if __name__ == "__main__":
    main()
