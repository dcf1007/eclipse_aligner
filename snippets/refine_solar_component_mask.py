"""Conservative binary refinement for a finalized seeded solar-component mask.

This operates only after automatic threshold selection is complete.  It performs
one 3x3 elliptical opening followed by one 3x3 elliptical closing, exactly as the
mask-refinement experiment under review.
"""
from __future__ import annotations

import cv2
import numpy as np

REFINEMENT_KERNEL_SIZE = 3
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
