"""Validated bright, well-supported solar seed selection helper.

Production integration replaces deepest-only seed selection at the coarse and
mapped full-resolution seed-establishment points. The threshold algorithm itself
is otherwise unchanged.
"""
import cv2
import numpy as np


class ThresholdResolutionError(RuntimeError):
    """Raised when a solar component cannot supply a seed."""


def brightest_supported_component_point(
    gray: np.ndarray,
    component_u8: np.ndarray,
) -> tuple[int, int]:
    """Choose the brightest robust interior seed; depth breaks brightness ties.

    Seed candidates normally must survive a 5x5 erosion of the component. This
    prevents an isolated hot pixel, one-pixel filament, or boundary artifact from
    becoming the solar seed. If a very thin component has no 5x5-supported pixel,
    the component itself is used as a deterministic fallback.
    """
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
