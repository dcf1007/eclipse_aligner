"""Manual real-frame check for fixed-T solar establishment without an auto seed."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import circle_arc_detector as cad

EXPECTED = Path(__file__).resolve().parent / "data" / "expected_seed_corpus_52.csv"
DEFAULT_FRAMES = [
    "0174__DSC0835.jpg",
    "0188__DSC0862.jpg",
    "0200__DSC0891.jpg",
    "0001__DSC0035 (2).jpg",
    "0113__DSC0343.jpg",
    "0118__DSC0348.jpg",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    expected = pd.read_csv(EXPECTED).set_index("file")
    ok_all = True
    for name in DEFAULT_FRAMES:
        row = expected.loc[name]
        threshold = int(row.new_t)
        auto_seed = ast.literal_eval(row.new_seed)
        gray = cv2.imread(str(args.corpus / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {name}")

        expected_component = cad.solar_component_from_seed_at_threshold(
            gray, threshold, auto_seed
        )
        manual_seed, manual_component = cad.establish_solar_component_at_threshold(
            gray, threshold, preferred_seed=None
        )
        solar = cad.build_solar_data(gray, threshold, manual_seed, manual_component)
        ok = (
            np.array_equal(manual_component, expected_component)
            and solar.threshold == threshold
            and solar.seed_point == manual_seed
        )
        ok_all &= ok
        print(name, "PASS" if ok else "FAIL", "auto", auto_seed, "manual", manual_seed)

    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
