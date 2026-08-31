"""Conservative refinement using the shared 7x7 elliptical solar kernel."""
from __future__ import annotations

import cv2
import numpy as np

SOLAR_COMPONENT_KERNEL_SIZE = 7
SOLAR_COMPONENT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (SOLAR_COMPONENT_KERNEL_SIZE, SOLAR_COMPONENT_KERNEL_SIZE),
)
REFINEMENT_ITERATIONS = 1


def refine_solar_component_mask(component: np.ndarray) -> np.ndarray:
    """Apply the agreed 7x7 elliptical OPEN then CLOSE to one solar component."""
    component = np.asarray(component, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if not np.any(component):
        raise ValueError("component mask is empty")

    u8 = np.where(component, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(
        u8,
        cv2.MORPH_OPEN,
        SOLAR_COMPONENT_KERNEL,
        iterations=REFINEMENT_ITERATIONS,
    )
    refined = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        SOLAR_COMPONENT_KERNEL,
        iterations=REFINEMENT_ITERATIONS,
    )
    return refined != 0
