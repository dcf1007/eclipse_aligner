"""Manual real-frame validation of post-threshold SolarData construction.

Requires the external image corpus directory supplied with ``--corpus``.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd

import circle_arc_detector as cad

EXPECTED = Path(__file__).resolve().parent / "data" / "expected_seed_corpus_52.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=52)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    expected = pd.read_csv(EXPECTED).iloc[args.start:args.end]
    rows = []
    for _, row in expected.iterrows():
        gray = cv2.imread(str(args.corpus / row["file"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {row['file']}")
        threshold = int(row["new_t"])
        seed = ast.literal_eval(row["new_seed"])

        t0 = time.perf_counter()
        component = cad.solar_component_from_seed_at_threshold(gray, threshold, seed)
        solar = cad.build_solar_data(gray, threshold, seed, component)
        seconds = time.perf_counter() - t0

        restored_component = cad.decompress_full_mask(solar.component_mask, gray.shape)
        roi = cad.decompress_full_mask(solar.roi_6_5_mask, gray.shape)
        guard = cad.decompress_full_mask(solar.guard_19_5_mask, gray.shape)
        contour = solar.component_contour
        ok = (
            solar.threshold == threshold
            and solar.seed_point == seed
            and np.array_equal(restored_component, component)
            and np.all(roi[component])
            and np.all(guard[roi])
            and contour.dtype == np.uint16
            and contour.ndim == 2
            and contour.shape[1] == 2
            and len(contour) > 0
            and np.all(component[contour[:, 1], contour[:, 0]])
        )
        rows.append({
            "file": row["file"],
            "group": row["group"],
            "threshold": threshold,
            "component_bytes": len(solar.component_mask),
            "roi_bytes": len(solar.roi_6_5_mask),
            "guard_bytes": len(solar.guard_19_5_mask),
            "contour_points": len(contour),
            "contour_bytes": int(contour.nbytes),
            "seconds": seconds,
            "ok": ok,
        })
        print(row["file"], "PASS" if ok else "FAIL", flush=True)

    out = pd.DataFrame(rows)
    if args.output is not None:
        out.to_csv(args.output, index=False)
    if not bool(out["ok"].all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
