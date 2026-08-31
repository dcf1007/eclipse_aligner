"""Validated 7x7-supported solar seed selection helper.

Seed candidates must survive the same 7x7 elliptical support erosion used by the
solar-component refinement. There is deliberately no unsupported fallback.
"""
import cv2
import numpy as np

SOLAR_COMPONENT_KERNEL_SIZE = 7
SOLAR_COMPONENT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (SOLAR_COMPONENT_KERNEL_SIZE, SOLAR_COMPONENT_KERNEL_SIZE),
)


class ThresholdResolutionError(RuntimeError):
    """Raised when a solar component cannot supply a robust interior seed."""


def brightest_supported_component_point(
    gray: np.ndarray,
    component_u8: np.ndarray,
) -> tuple[int, int]:
    """Choose brightest 7x7-supported pixel; depth breaks brightness ties."""
    source = (component_u8 != 0).astype(np.uint8)
    if gray.shape != source.shape:
        raise ValueError("gray and component must have identical shapes")
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")

    supported = cv2.erode(source, SOLAR_COMPONENT_KERNEL, iterations=1) != 0
    if not np.any(supported):
        raise ThresholdResolutionError(
            "Solar component has no 7x7-supported interior seed"
        )

    max_gray = int(gray[supported].max())
    brightest = supported & (gray == max_gray)
    # The 5 is the L2 distance-transform approximation mask size, not support size.
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    scores = np.where(brightest, distance, -1.0)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(x), int(y)
