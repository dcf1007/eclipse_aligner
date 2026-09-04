"""Write deterministic threshold-refinement measurements for research review.

This helper consumes the production ``ThresholdMeasurement`` trajectory returned by
``refine_threshold``. It does not construct alternate morphology candidates or
reimplement descriptor scoring.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import circle_arc_detector as cad


def write_threshold_measurements(
    rows: Iterable[cad.ThresholdMeasurement],
    output: str | Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "threshold",
        "component_area",
        "filled_area",
        "roughness",
        "solidity",
        "internal_dark_fraction",
        "edge_alignment",
        "edge_credible_fraction",
        "edge_transition_monotonicity",
        "q_roughness",
        "q_holes",
        "q_area",
        "q_solidity",
        "q_edge",
        "score",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})
