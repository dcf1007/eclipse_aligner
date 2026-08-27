"""Optimized grayscale-only eclipse automatic threshold finder candidate.

The algorithm intentionally uses only grayscale histogram modes and 8-connected
component topology. Threshold semantics are fixed:

    dark  = gray <= T
    light = gray > T

No HSV/color interpretation, Otsu thresholding, morphology, ellipse fitting,
bright-pixel dominance, competitor gain, or horizon logic is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

WORK_MAX_DIM = 1200
HISTOGRAM_SIGMA = 3.0
ROI_DILATION_FRACTION = 0.065
GUARD_DILATION_FRACTION = 0.195


class ThresholdResolutionError(RuntimeError):
    """Expected inability to establish/track a separated solar component."""


@dataclass(frozen=True)
class HistogramSeed:
    peak: int
    left_edge: int
    threshold: int
    point: tuple[int, int]
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class CoarseThresholdResult:
    histogram_peak: int
    histogram_left_edge: int
    seed_threshold: int
    seed_point: tuple[int, int]
    seed_mask: np.ndarray
    threshold: int
    component_mask: np.ndarray
    component_area: int
    component_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class AutoThresholdResult:
    threshold: int
    histogram_peak: int
    histogram_left_edge: int
    seed_threshold: int
    coarse_threshold: int | None
    roi_seed_threshold: int | None
    full_seed_point: tuple[int, int] | None
    used_guard: bool
    resolved: bool
    reason: str = ""


@dataclass(frozen=True)
class ObservationRegion:
    """Rounded component dilation stored in the smallest useful rectangular crop."""
    bbox: tuple[int, int, int, int]       # x0, y0, x1, y1 in full image
    allowed_u8: np.ndarray               # 255 inside rounded dilation, else 0
    boundary: np.ndarray                 # True on the INNER edge of allowed region
    gray: np.ndarray                     # view into full-resolution gray image
    seed_local: tuple[int, int]
    touches_left_image_edge: bool
    touches_right_image_edge: bool
    touches_top_image_edge: bool
    touches_bottom_image_edge: bool


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert once to authoritative 8-bit grayscale."""
    if image.ndim == 2:
        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_gray_max_dim(gray: np.ndarray, max_dim: int = WORK_MAX_DIM) -> np.ndarray:
    """Downscale grayscale with area averaging; never upscale."""
    h, w = gray.shape
    if max(h, w) <= max_dim:
        return gray.copy()
    scale = float(max_dim) / float(max(h, w))
    return cv2.resize(
        gray,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smoothed_histogram(gray: np.ndarray, sigma: float = HISTOGRAM_SIGMA):
    """Return exact 256-bin histogram plus a peak-finding-only smoothed copy."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    smooth = np.convolve(hist, _gaussian_kernel_1d(sigma), mode="same")
    return hist, smooth


def _local_peaks(values: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    # Saturation itself is allowed to be the rightmost mode.
    if len(values) >= 2 and values[-1] > values[-2]:
        peaks.append(len(values) - 1)
    return peaks or [int(np.argmax(values))]


def _preceding_valley(values: np.ndarray, peak: int) -> int:
    """Nearest local histogram minimum on the left; zero is deterministic fallback."""
    for i in range(peak - 1, 0, -1):
        if values[i] <= values[i - 1] and values[i] < values[i + 1]:
            return i
    return 0


def rightmost_histogram_peak(gray: np.ndarray) -> tuple[int, int]:
    """Rightmost smoothed mode and the valley defining its left edge."""
    _hist, smooth = smoothed_histogram(gray)
    peak = max(_local_peaks(smooth))
    return int(peak), int(_preceding_valley(smooth, peak))


def deepest_component_point(component_u8: np.ndarray) -> tuple[int, int]:
    """Choose an interior seed point; unlike a centroid this stays inside crescents."""
    source = (component_u8 != 0).astype(np.uint8)
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


def _component_metadata(component: np.ndarray):
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        raise ThresholdResolutionError("Empty component")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return int(len(xs)), (x0, y0, x1 - x0, y1 - y0)


def largest_nonborder_component(binary: np.ndarray) -> np.ndarray | None:
    """Largest 8-connected bright component enclosed by the current raster."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary != 0).astype(np.uint8), 8
    )
    h, w = binary.shape
    best = None
    for label in range(1, count):
        x, y, cw, ch, area = map(int, stats[label])
        if x == 0 or y == 0 or x + cw >= w or y + ch >= h:
            continue
        candidate = (area, label)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return labels == best[1]


def establish_histogram_seed(work_gray: np.ndarray) -> HistogramSeed:
    """Start at the left edge of the rightmost peak and descend until Sun appears."""
    peak, left_edge = rightmost_histogram_peak(work_gray)
    for threshold in range(left_edge, -1, -1):
        component = largest_nonborder_component(work_gray > threshold)
        if component is None:
            continue
        mask = np.where(component, 255, 0).astype(np.uint8)
        area, bbox = _component_metadata(component)
        return HistogramSeed(
            peak=peak,
            left_edge=left_edge,
            threshold=int(threshold),
            point=deepest_component_point(mask),
            mask=mask,
            area=area,
            bbox=bbox,
        )
    raise ThresholdResolutionError("No enclosed bright component exists")


def _flood_component_bool(binary: np.ndarray, seed: tuple[int, int]) -> np.ndarray | None:
    x, y = map(int, seed)
    h, w = binary.shape
    if not (0 <= x < w and 0 <= y < h) or not bool(binary[y, x]):
        return None
    work = cv2.compare(binary.astype(np.uint8), 0, cv2.CMP_GT)
    cv2.floodFill(work, None, (x, y), 128, flags=8)
    return work == 128


def _touches_border(component: np.ndarray | None) -> bool:
    if component is None or not np.any(component):
        return True
    return bool(
        np.any(component[0]) or np.any(component[-1])
        or np.any(component[:, 0]) or np.any(component[:, -1])
    )


def coarse_threshold_search(work_gray: np.ndarray) -> CoarseThresholdResult:
    """Track the same high-T seed downward; lowest enclosed T defines coarse ROI."""
    seed = establish_histogram_seed(work_gray)
    lowest_t = seed.threshold
    lowest_component = seed.mask != 0

    # Connectivity is monotone while T is lowered: pixels are only added, so once
    # this tracked component reaches the image border it cannot become enclosed at
    # any still-lower threshold. It is therefore safe to stop at first contact.
    for threshold in range(seed.threshold, -1, -1):
        component = _flood_component_bool(work_gray > threshold, seed.point)
        if component is None:
            continue
        if _touches_border(component):
            break
        lowest_t = threshold
        lowest_component = component

    area, bbox = _component_metadata(lowest_component)
    return CoarseThresholdResult(
        histogram_peak=seed.peak,
        histogram_left_edge=seed.left_edge,
        seed_threshold=seed.threshold,
        seed_point=seed.point,
        seed_mask=seed.mask,
        threshold=int(lowest_t),
        component_mask=np.where(lowest_component, 255, 0).astype(np.uint8),
        component_area=area,
        component_bbox=bbox,
    )


def _map_interval(lo: int, hi: int, source_n: int, target_n: int) -> tuple[int, int]:
    out_lo = int(math.floor(lo * target_n / source_n))
    out_hi = int(math.ceil(hi * target_n / source_n))
    return max(0, out_lo), min(target_n, max(out_lo + 1, out_hi))


def establish_full_seed(
    full_gray: np.ndarray,
    work_shape: tuple[int, int],
    coarse_seed_mask: np.ndarray,
    seed_threshold: int,
) -> tuple[int, int]:
    """Map only the coarse seed bbox and pick an actual bright original pixel."""
    ys, xs = np.nonzero(coarse_seed_mask)
    if len(xs) == 0:
        raise ThresholdResolutionError("Coarse seed mask is empty")
    sh, sw = work_shape
    fh, fw = full_gray.shape
    sx0, sx1 = int(xs.min()), int(xs.max()) + 1
    sy0, sy1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = _map_interval(sx0, sx1, sw, fw)
    y0, y1 = _map_interval(sy0, sy1, sh, fh)

    coarse_crop = (coarse_seed_mask[sy0:sy1, sx0:sx1] != 0).astype(np.uint8)
    mapped = cv2.resize(coarse_crop, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) != 0
    local_gray = full_gray[y0:y1, x0:x1]
    candidate = mapped & (local_gray > int(seed_threshold))

    if not np.any(candidate):
        # This does not choose T; it only recovers an original-resolution coordinate
        # underneath the known coarse solar seed if area averaging shifted all local
        # values across Tstart.
        cy, cx = np.nonzero(mapped)
        if len(cx) == 0:
            raise ThresholdResolutionError("Mapped solar seed is empty")
        values = local_gray[cy, cx]
        i = int(np.argmax(values))
        return x0 + int(cx[i]), y0 + int(cy[i])

    local = np.where(candidate, 255, 0).astype(np.uint8)
    px, py = deepest_component_point(local)
    return x0 + px, y0 + py


def _full_flood_component(gray: np.ndarray, threshold: int, seed: tuple[int, int]):
    binary = cv2.compare(gray, int(threshold), cv2.CMP_GT)
    x, y = seed
    if binary[y, x] == 0:
        return None
    cv2.floodFill(binary, None, seed, 128, flags=8)
    return binary == 128


def full_roi_seed_component(
    full_gray: np.ndarray,
    coarse_threshold: int,
    seed_threshold: int,
    seed_point: tuple[int, int],
):
    """Obtain an ACTUAL full-resolution enclosed component for ROI dilation."""
    start = max(0, int(coarse_threshold))
    for threshold in range(start, 256):
        component = _full_flood_component(full_gray, threshold, seed_point)
        if component is not None and not _touches_border(component):
            return int(threshold), component
        # Usually resolved at coarse T or one level higher. The loop remains
        # exhaustive/defensive so a reduced-resolution mismatch cannot fail T.
    raise ThresholdResolutionError("No enclosed original-resolution solar component")


def _build_observation_region(
    full_gray: np.ndarray,
    component: np.ndarray,
    margin: float,
    seed_point: tuple[int, int],
) -> ObservationRegion:
    """Distance-dilate the actual component by margin and cache its inner boundary."""
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        raise ThresholdResolutionError("Cannot dilate empty solar component")
    h, w = full_gray.shape
    padding = max(0, int(math.ceil(float(margin))))
    x0 = max(0, int(xs.min()) - padding - 2)
    x1 = min(w, int(xs.max()) + padding + 3)
    y0 = max(0, int(ys.min()) - padding - 2)
    y1 = min(h, int(ys.max()) + padding + 3)

    component_crop = component[y0:y1, x0:x1]
    outside = np.where(component_crop, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    allowed = distance <= float(margin)
    allowed_u8 = np.where(allowed, 255, 0).astype(np.uint8)

    # Inner edge of the rounded region. If the tracked component reaches it, the
    # observation is inconclusive and the caller retries the same T in 19.5% guard.
    eroded = cv2.erode(allowed_u8, np.ones((3, 3), np.uint8), iterations=1)
    boundary = allowed & (eroded == 0)

    return ObservationRegion(
        bbox=(x0, y0, x1, y1),
        allowed_u8=allowed_u8,
        boundary=boundary,
        gray=full_gray[y0:y1, x0:x1],
        seed_local=(seed_point[0] - x0, seed_point[1] - y0),
        touches_left_image_edge=x0 == 0,
        touches_right_image_edge=x1 == w,
        touches_top_image_edge=y0 == 0,
        touches_bottom_image_edge=y1 == h,
    )


def _region_status(region: ObservationRegion, threshold: int):
    """Return (exists, touches_artificial_boundary, touches_true_image_border)."""
    work = cv2.compare(region.gray, int(threshold), cv2.CMP_GT)
    cv2.bitwise_and(work, region.allowed_u8, dst=work)
    sx, sy = region.seed_local
    if not (0 <= sx < work.shape[1] and 0 <= sy < work.shape[0]) or work[sy, sx] == 0:
        return False, False, False

    cv2.floodFill(work, None, (sx, sy), 128, flags=8)
    component = work == 128
    touches_artificial = bool(np.any(component & region.boundary))

    touches_true = False
    if region.touches_top_image_edge and np.any(component[0]):
        touches_true = True
    if region.touches_bottom_image_edge and np.any(component[-1]):
        touches_true = True
    if region.touches_left_image_edge and np.any(component[:, 0]):
        touches_true = True
    if region.touches_right_image_edge and np.any(component[:, -1]):
        touches_true = True
    return True, touches_artificial, touches_true


def find_lowest_full_threshold(
    full_gray: np.ndarray,
    seed_point: tuple[int, int],
    roi_seed_component: np.ndarray,
    histogram_upper_hint: int,
):
    """Scan T upward from zero and return the first genuinely separated threshold.

    The 6.5% * sqrt(pixel_count) dilation is always tried first. Only if that flood
    reaches its artificial rounded edge is the SAME T retried using 19.5%. A flood
    that also reaches the 19.5% guard is treated as background-connected at that T.
    """
    h, w = full_gray.shape
    image_scale = math.sqrt(float(w) * float(h))
    roi = _build_observation_region(
        full_gray, roi_seed_component, ROI_DILATION_FRACTION * image_scale, seed_point
    )
    guard = _build_observation_region(
        full_gray, roi_seed_component, GUARD_DILATION_FRACTION * image_scale, seed_point
    )

    used_guard = False
    upper = max(0, min(255, int(histogram_upper_hint)))
    for threshold in range(0, 256):
        # The histogram upper hint is diagnostic rather than a hard limit. We keep
        # scanning defensively above it if original-resolution topology requires it.
        exists, touches_roi, touches_true = _region_status(roi, threshold)
        if not exists or touches_true:
            continue
        if not touches_roi:
            return int(threshold), used_guard

        used_guard = True
        exists, touches_guard, touches_true = _region_status(guard, threshold)
        if exists and not touches_true and not touches_guard:
            return int(threshold), used_guard

    raise ThresholdResolutionError("Tracked solar component never became separated")


def auto_threshold_from_gray(full_gray: np.ndarray) -> AutoThresholdResult:
    """Run complete two-resolution threshold selection, with histogram-left fallback."""
    full_gray = to_gray(full_gray)
    work_gray = resize_gray_max_dim(full_gray)
    peak, left_edge = rightmost_histogram_peak(work_gray)

    try:
        coarse = coarse_threshold_search(work_gray)
        full_seed = establish_full_seed(
            full_gray, work_gray.shape, coarse.seed_mask, coarse.seed_threshold
        )
        roi_seed_t, roi_seed_component = full_roi_seed_component(
            full_gray, coarse.threshold, coarse.seed_threshold, full_seed
        )
        final_t, used_guard = find_lowest_full_threshold(
            full_gray,
            full_seed,
            roi_seed_component,
            histogram_upper_hint=max(coarse.seed_threshold, roi_seed_t),
        )
        return AutoThresholdResult(
            threshold=final_t,
            histogram_peak=coarse.histogram_peak,
            histogram_left_edge=coarse.histogram_left_edge,
            seed_threshold=coarse.seed_threshold,
            coarse_threshold=coarse.threshold,
            roi_seed_threshold=roi_seed_t,
            full_seed_point=full_seed,
            used_guard=used_guard,
            resolved=True,
        )
    except ThresholdResolutionError as exc:
        # User-selected deterministic fallback: histogram always has a mode, so if
        # component topology cannot be resolved we still return the left side of
        # the rightmost peak instead of reintroducing color/Otsu/fixed-T heuristics.
        return AutoThresholdResult(
            threshold=int(left_edge),
            histogram_peak=int(peak),
            histogram_left_edge=int(left_edge),
            seed_threshold=int(left_edge),
            coarse_threshold=None,
            roi_seed_threshold=None,
            full_seed_point=None,
            used_guard=False,
            resolved=False,
            reason=str(exc),
        )


def auto_threshold(image: np.ndarray) -> AutoThresholdResult:
    return auto_threshold_from_gray(to_gray(image))
