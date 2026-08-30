"""Promote the visually approved 7x7 elliptical SolarData mask refinement.

This patch is deliberately narrow: it changes the existing post-threshold
refinement scale from 3x3 to 7x7. Opening remains first, closing remains second,
and both remain one iteration. Threshold selection and SolarData structure are
untouched.
"""
from __future__ import annotations

from pathlib import Path

SOURCE = Path("circle_arc_detector.py")


def apply(path: Path = SOURCE) -> None:
    text = path.read_text()

    old_constant = "REFINEMENT_KERNEL_SIZE = 3"
    new_constant = "REFINEMENT_KERNEL_SIZE = 7"
    if text.count(old_constant) != 1:
        raise RuntimeError(
            f"Expected exactly one current 3x3 refinement constant; found {text.count(old_constant)}"
        )
    text = text.replace(old_constant, new_constant, 1)

    old_burr = "The opening suppresses one-pixel/thin outward burrs.  The following closing"
    new_burr = "The opening suppresses small/thin outward burrs.  The following closing"
    if text.count(old_burr) != 1:
        raise RuntimeError(
            f"Expected exactly one current refinement burr description; found {text.count(old_burr)}"
        )
    text = text.replace(old_burr, new_burr, 1)

    path.write_text(text)


if __name__ == "__main__":
    apply()
