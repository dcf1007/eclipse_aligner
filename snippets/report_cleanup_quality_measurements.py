"""Reusable research helper for inspecting unweighted adaptive-cleanup measurements.

This module deliberately delegates candidate construction and metric computation to
``circle_arc_detector`` so weighting experiments exercise the exact production
machinery rather than a parallel reimplementation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import circle_arc_detector as cad


def write_cleanup_measurements(rows: Iterable[cad.CleanupCandidateEvaluation], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "name", "area", "contour_n", "roughness", "solidity",
            "internal_dark_fraction", "contour_cleanup", "roughness_cleanup",
            "solidity_gain", "internal_dark_cleanup", "area_loss", "solidity_loss",
        ])
        writer.writeheader()
        for row in rows:
            d, m = row.topology, row.metrics
            writer.writerow({
                "name": row.name,
                "area": d.area,
                "contour_n": d.contour_n,
                "roughness": d.roughness,
                "solidity": d.solidity,
                "internal_dark_fraction": d.internal_dark_fraction,
                "contour_cleanup": m.contour_cleanup,
                "roughness_cleanup": m.roughness_cleanup,
                "solidity_gain": m.solidity_gain,
                "internal_dark_cleanup": m.internal_dark_cleanup,
                "area_loss": m.area_loss,
                "solidity_loss": m.solidity_loss,
            })
