"""Apply the visually approved SolarData mask refinement to production source.

This patch intentionally reuses the tested 3x3 elliptical opening -> closing helper
from ``refine_solar_component_mask.py``. It changes only post-threshold SolarData
construction; automatic threshold selection and raw seeded-component identity stay
unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys


HELPER_BLOCK = '''REFINEMENT_KERNEL_SIZE = 3
REFINEMENT_ITERATIONS = 1


def refine_solar_component_mask(component: np.ndarray) -> np.ndarray:
    """Return a conservatively smoothed boolean solar-component mask.

    The opening suppresses one-pixel/thin outward burrs.  The following closing
    fills comparably small inward notches/holes.  No thresholding, ellipse, radius,
    horizon, EXIF, or cross-image information participates.
    """
    component = np.asarray(component, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if not np.any(component):
        raise ValueError("component mask is empty")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (REFINEMENT_KERNEL_SIZE, REFINEMENT_KERNEL_SIZE),
    )
    u8 = np.where(component, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(
        u8,
        cv2.MORPH_OPEN,
        kernel,
        iterations=REFINEMENT_ITERATIONS,
    )
    refined = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=REFINEMENT_ITERATIONS,
    )
    return refined != 0
'''


def apply(source_path: Path) -> None:
    text = source_path.read_text()

    marker = '''# ---------------------------------------------------------------------------\n# Post-threshold full-resolution solar data\n# ---------------------------------------------------------------------------\n'''
    if marker not in text:
        raise RuntimeError("post-threshold SolarData section marker not found")
    if "def refine_solar_component_mask(" in text:
        raise RuntimeError("mask refinement is already present")

    text = text.replace(marker, marker + HELPER_BLOCK + "\n\n", 1)

    old_doc = '    """Solar geometry established at exactly one full-resolution threshold T."""'
    new_doc = '    """Refined solar geometry established at exactly one full-resolution threshold T."""'
    if old_doc not in text:
        raise RuntimeError("SolarData docstring anchor not found")
    text = text.replace(old_doc, new_doc, 1)

    old = '''    if int(full_gray[seed_y, seed_x]) <= threshold:\n        raise ThresholdResolutionError("SolarData seed is not light at its threshold")\n\n    image_scale = math.sqrt(float(width) * float(height))\n'''
    new = '''    if int(full_gray[seed_y, seed_x]) <= threshold:\n        raise ThresholdResolutionError("SolarData seed is not light at its threshold")\n\n    # Threshold selection and seeded identity are complete before this point.\n    # Refine only the finalized component used for persistent SolarData geometry.\n    component = refine_solar_component_mask(component)\n    if not component[seed_y, seed_x]:\n        raise ThresholdResolutionError(\n            "Refined SolarData component no longer contains the solar seed"\n        )\n\n    image_scale = math.sqrt(float(width) * float(height))\n'''
    if old not in text:
        raise RuntimeError("build_solar_data insertion anchor not found")
    text = text.replace(old, new, 1)

    source_path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_solar_mask_refinement.py PATH_TO_circle_arc_detector.py")
    apply(Path(sys.argv[1]))
