"""5x5-square-supported solar seed helper with thin-component fallback.

This restores the seed-selection behavior used by the validated Auto-T implementation
at commit 8ea0e44. Candidates normally must survive a 5x5 square erosion. If that
erosion removes the whole component, the component itself remains eligible so a very
thin early Auto-T tracking component can still establish identity.
"""
import cv2
import numpy as np


class ThresholdResolutionError(RuntimeError):
    """Raised when a solar component is empty or otherwise unusable."""


def brightest_supported_component_point(
    gray: np.ndarray,
    component_u8: np.ndarray,
) -> tuple[int, int]:
    """Choose brightest 5x5-supported pixel; fall back to component; depth ties."""
    source = (component_u8 != 0).astype(np.uint8)
    if gray.shape != source.shape:
        raise ValueError("gray and component must have identical shapes")
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")

    supported = cv2.erode(
        source,
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ) != 0
    if not np.any(supported):
        supported = source != 0

    max_gray = int(gray[supported].max())
    brightest = supported & (gray == max_gray)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    scores = np.where(brightest, distance, -1.0)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(x), int(y)
