"""Explicit-kernel supported solar-seed selection helper.

The caller owns support geometry. Unsupported components fail explicitly; there is
no fallback to a boundary/thin pixel that did not survive the requested erosion.
"""
import cv2
import numpy as np


class ThresholdResolutionError(RuntimeError):
    """Raised when a component cannot supply the requested supported seed."""


def brightest_supported_component_point(
    gray: np.ndarray,
    component_u8: np.ndarray,
    support_kernel: np.ndarray,
) -> tuple[int, int]:
    """Choose brightest support-eligible pixel; depth breaks brightness ties."""
    source = (component_u8 != 0).astype(np.uint8)
    support_kernel = np.asarray(support_kernel, dtype=np.uint8)
    if gray.shape != source.shape:
        raise ValueError("gray and component must have identical shapes")
    if support_kernel.ndim != 2 or support_kernel.size == 0 or not np.any(support_kernel):
        raise ValueError("support kernel must be a non-empty two-dimensional mask")
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")
    supported = cv2.erode(source, support_kernel, iterations=1) != 0
    if not np.any(supported):
        kh, kw = support_kernel.shape
        raise ThresholdResolutionError(
            f"Solar component has no {kw}x{kh}-supported interior seed"
        )
    max_gray = int(gray[supported].max())
    brightest = supported & (gray == max_gray)
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    scores = np.where(brightest, distance, -1.0)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(x), int(y)
