"""Manual 52-frame regression for automatic T/seed stability.

Requires the external image corpus directory supplied with ``--corpus``. The
expected results are retained in ``tests/data/expected_seed_corpus_52.csv``.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import time

import cv2
import pandas as pd

import circle_arc_detector as cad

ROOT = Path(__file__).resolve().parents[1]
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
        path = args.corpus / row["file"]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {path}")
        t0 = time.perf_counter()
        result = cad.auto_threshold(gray)
        seconds = time.perf_counter() - t0
        expected_seed = ast.literal_eval(row["new_seed"])
        expected_resolved = bool(row["new_resolved"])
        ok = (
            result.threshold == int(row["new_t"])
            and result.resolved == expected_resolved
            and result.full_seed_point == expected_seed
        )
        rows.append({
            "file": row["file"],
            "group": row["group"],
            "expected_t": int(row["new_t"]),
            "actual_t": result.threshold,
            "expected_resolved": expected_resolved,
            "actual_resolved": result.resolved,
            "expected_seed": expected_seed,
            "actual_seed": result.full_seed_point,
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
