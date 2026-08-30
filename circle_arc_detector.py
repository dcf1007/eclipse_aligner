"""Eclipse alignment GUI with grayscale automatic threshold selection.

This module combines the application's user-interface foundation with the tested
image-only automatic threshold finder. The GUI owns image navigation, per-image
processing settings, control interaction, preview lifecycle, and cached automatic
threshold results. The threshold finder itself remains independent of GUI state: it accepts the
authoritative 8-bit grayscale brightness image and returns an
``AutoThresholdResult`` describing the selected threshold and the topology used
to obtain it. Color-to-grayscale conversion is an input-stage responsibility and
is performed before the threshold algorithm is called.

All processing controls are per-image. ``DetectorApp.image_state`` is keyed by the
absolute image path. Each image entry is a normal dictionary with three conceptual fields:
``settings`` stores sparse ``ImageSettings`` overrides,
``auto_threshold_result`` caches the complete automatic threshold result, and
``solar_data`` caches post-threshold full-resolution solar geometry when it has
been established for a specific T. Ordinary controls use the application defaults as their baseline;
the threshold uses the cached automatic threshold as its image-specific baseline.
Returning a control to its baseline removes that key from ``settings``. Reusing the
cached automatic result means the threshold algorithm is not rerun when Auto select
is clicked again for an unchanged loaded image.

Slider labels update continuously, but a setting is committed only after mouse
release or the final keyboard key release. Keyboard auto-repeat can emit temporary
release/press pairs on some Tk platforms, so releases are coalesced through a short
settling window. Checkboxes and radio buttons commit immediately. A completed
setting change calls ``_commit_setting_change(setting_name)``. Non-threshold controls
only persist their sparse override. A completed threshold change first displays the
pure ``gray > T`` raster, then immediately establishes full-resolution SolarData at
that exact T and replaces the threshold canvas with the finalized refined component.

``refresh_preview()`` remains the explicit boundary for later ellipse, arc, and
horizon processing, but threshold/SolarData establishment is already complete before
those later stages begin. Image load follows the same two-stage threshold display:
pure threshold first, finalized refined component second. Apply Full Resolution remains
a separate explicit action. Radius auto-selection, ellipse fitting, horizon handling,
centering, and export are not implemented yet.

The threshold algorithm uses authoritative 8-bit grayscale with fixed semantics
``dark = gray <= T`` and ``light = gray > T``. It derives a <=1200-pixel working
raster with INTER_AREA, uses the rightmost locally smoothed histogram mode only to
establish a solar seed, tracks 8-connected component topology, maps the seed back
to full resolution, constructs rounded 6.5% and 19.5% observation regions, and
establishes the lowest full-resolution threshold whose tracked component is genuinely
separated, then examines that same seeded component through T..T+5 and returns the
first topology knee where boundary cleanup has largely been gained before further
threshold increases mainly erode the solar component. If topology cannot be resolved,
the deterministic fallback is the left edge of the rightmost histogram peak. No HSV/color thresholding, Otsu thresholding,
ellipse-fit score, bright-pixel dominance, competitor gain, or horizon special case
is part of automatic threshold selection.
"""

import argparse
import base64
from dataclasses import dataclass
import math
import os
import tkinter as tk
import zlib
from tkinter import filedialog

import cv2
import numpy as np


IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)

SLIDER_KEY_RELEASE_SETTLE_MS = 45
PREVIEW_REDRAW_DELAY_MS = 60
THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS = 40


def transparent_bgra(width: int = 1, height: int = 1) -> np.ndarray:
    """Return a BGRA frame whose pixels are fully transparent (alpha = 0)."""
    width = max(1, int(width))
    height = max(1, int(height))
    return np.zeros((height, width, 4), dtype=np.uint8)


def opaque_bgra(bgr: np.ndarray) -> np.ndarray:
    """Convert a normal OpenCV BGR image to BGRA with fully opaque image pixels."""
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


# ---------------------------------------------------------------------------
# Per-image processing settings
# ---------------------------------------------------------------------------
@dataclass
class ImageSettings:
    """Sparse per-image overrides; ``None`` means use that setting's baseline."""

    threshold: int | None = None
    min_radius: int | None = None
    max_radius: int | None = None
    max_error: float | None = None
    min_coverage: int | None = None
    morphology: bool | None = None
    outer_limb_assistance: bool | None = None
    use_horizon: bool | None = None
    center_target: str | None = None


# ---------------------------------------------------------------------------
# Grayscale automatic threshold finder
# ---------------------------------------------------------------------------
WORK_MAX_DIM = 1200
PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)
ROI_DILATION_FRACTION = 0.065
GUARD_DILATION_FRACTION = 0.195


TOPOLOGY_OPTIMIZATION_STEPS = 5


@dataclass(frozen=True)
class ThresholdTopology:
    threshold: int
    area: int
    contour_n: int
    perimeter: float
    roughness: float
    solidity: float


@dataclass(frozen=True)
class ThresholdTopologySelection:
    threshold: int
    base_threshold: int
    delta: int
    trajectory: tuple[ThresholdTopology, ...]
    net_quality: tuple[float, ...]
    knee_curve: tuple[float, ...]


def _component_descriptor(component: np.ndarray, threshold: int) -> ThresholdTopology:
    component = np.asarray(component, dtype=bool)
    area = int(np.count_nonzero(component))
    if area <= 0:
        raise ValueError("empty component")
    u8 = np.where(component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("component has no external contour")
    contour = max(contours, key=cv2.contourArea)
    contour_n = int(len(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0.0 else 0.0
    roughness = float(perimeter / max(2.0 * math.sqrt(math.pi * area), 1e-9))
    return ThresholdTopology(
        threshold=int(threshold),
        area=area,
        contour_n=contour_n,
        perimeter=perimeter,
        roughness=roughness,
        solidity=solidity,
    )


def topology_trajectory_from_separated_component(
    full_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> tuple[ThresholdTopology, ...]:
    """Measure exact seeded topology for T..T+max_delta in the T-component crop.

    Because higher thresholds can only remove light pixels, a seeded component at
    T+k cannot gain pixels that were outside the seeded component at T. Restricting
    the scan to the base-component crop is therefore topology-equivalent to repeated
    full-frame floods and substantially cheaper on full-resolution photographs.
    """
    base_component = np.asarray(base_component, dtype=bool)
    ys, xs = np.nonzero(base_component)
    if len(xs) == 0:
        raise ValueError("base component is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    base_crop = base_component[y0:y1, x0:x1]
    gray_crop = full_gray[y0:y1, x0:x1]
    sx = int(seed_point[0]) - x0
    sy = int(seed_point[1]) - y0
    if not (0 <= sx < gray_crop.shape[1] and 0 <= sy < gray_crop.shape[0]):
        raise ValueError("seed lies outside base component crop")

    trajectory: list[ThresholdTopology] = []
    for delta in range(max(0, int(max_delta)) + 1):
        threshold = min(255, int(base_threshold) + delta)
        light = base_crop & (gray_crop > threshold)
        if not bool(light[sy, sx]):
            break
        flood = np.where(light, 255, 0).astype(np.uint8)
        cv2.floodFill(flood, None, (sx, sy), 128, flags=8)
        component = flood == 128
        if not np.any(component):
            break
        trajectory.append(_component_descriptor(component, threshold))
        if threshold >= 255:
            break
    if not trajectory:
        raise ValueError("no valid topology samples")
    return tuple(trajectory)


def select_topology_knee(
    trajectory: tuple[ThresholdTopology, ...] | list[ThresholdTopology],
) -> ThresholdTopologySelection:
    """Select the first knee of cleanup benefit versus component erosion.

    All quantities are dimensionless relative changes from the separated component.

    Benefit terms (equal weight):
      * external contour-count reduction,
      * perimeter-normalized roughness reduction,
      * positive solidity gain normalized by the base solidity headroom to 1.

    Cost terms (equal weight):
      * solar-component area loss,
      * solidity loss normalized by the base solidity.

    The resulting net-quality trajectory is replaced by its best-so-far envelope;
    a higher T that is already worse cannot become the selected knee. The envelope
    is normalized to its observed maximum and compared with uniform threshold
    progress (0..1). Maximizing ``quality_progress - threshold_progress`` is the
    standard discrete elbow construction and deliberately chooses the *first* point
    where most available topology improvement has already been realized.
    """
    rows = tuple(trajectory)
    if not rows:
        raise ValueError("empty topology trajectory")
    if len(rows) == 1:
        return ThresholdTopologySelection(
            threshold=rows[0].threshold,
            base_threshold=rows[0].threshold,
            delta=0,
            trajectory=rows,
            net_quality=(0.0,),
            knee_curve=(0.0,),
        )

    base = rows[0]
    base_area = max(float(base.area), 1.0)
    base_contour = max(float(base.contour_n), 1.0)
    base_roughness = max(float(base.roughness), 1e-12)
    base_solidity = max(float(base.solidity), 1e-12)
    solidity_headroom = max(1.0 - float(base.solidity), 1e-12)

    net: list[float] = []
    for row in rows:
        contour_cleanup = max(0.0, 1.0 - float(row.contour_n) / base_contour)
        roughness_cleanup = max(0.0, 1.0 - float(row.roughness) / base_roughness)
        solidity_gain = max(
            0.0,
            (float(row.solidity) - float(base.solidity)) / solidity_headroom,
        )
        benefit = (contour_cleanup + roughness_cleanup + solidity_gain) / 3.0

        area_loss = max(0.0, 1.0 - float(row.area) / base_area)
        solidity_loss = max(
            0.0,
            (float(base.solidity) - float(row.solidity)) / base_solidity,
        )
        cost = (area_loss + solidity_loss) / 2.0
        net.append(float(benefit - cost))

    best_so_far = np.maximum.accumulate(np.asarray(net, dtype=np.float64))
    best_value = float(np.max(best_so_far))
    if not math.isfinite(best_value) or best_value <= 0.0:
        selected_index = 0
        knee = np.zeros(len(rows), dtype=np.float64)
    else:
        quality_progress = best_so_far / best_value
        threshold_progress = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
        knee = quality_progress - threshold_progress
        selected_index = int(np.argmax(knee))  # np.argmax gives earliest exact tie.

    selected = rows[selected_index]
    return ThresholdTopologySelection(
        threshold=int(selected.threshold),
        base_threshold=int(base.threshold),
        delta=int(selected.threshold - base.threshold),
        trajectory=rows,
        net_quality=tuple(float(v) for v in net),
        knee_curve=tuple(float(v) for v in knee),
    )


def optimize_separated_threshold(
    full_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> ThresholdTopologySelection:
    """Optimize a valid separated T without ever lowering it or invalidating it.

    ``base_threshold`` has already been proven to separate the persistent solar
    component from the background. Topology optimization is therefore a refinement,
    not another resolution stage. If the optional descriptor scan cannot be formed,
    return the proven base threshold as a one-sample selection rather than turning
    a resolved automatic threshold into an unresolved one.
    """
    try:
        trajectory = topology_trajectory_from_separated_component(
            full_gray,
            int(base_threshold),
            tuple(map(int, seed_point)),
            base_component,
            max_delta=max_delta,
        )
        return select_topology_knee(trajectory)
    except (ValueError, cv2.error):
        try:
            base = _component_descriptor(base_component, int(base_threshold))
            trajectory = (base,)
        except (ValueError, cv2.error):
            base = ThresholdTopology(
                threshold=int(base_threshold),
                area=int(np.count_nonzero(base_component)),
                contour_n=0,
                perimeter=0.0,
                roughness=0.0,
                solidity=0.0,
            )
            trajectory = (base,)
        return ThresholdTopologySelection(
            threshold=int(base_threshold),
            base_threshold=int(base_threshold),
            delta=0,
            trajectory=trajectory,
            net_quality=(0.0,),
            knee_curve=(0.0,),
        )


class ThresholdResolutionError(RuntimeError):
    """Expected inability to establish/track a separated solar component."""


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


def find_rightmost_histogram_peak(gray: np.ndarray) -> tuple[int, int]:
    """Return the rightmost 3-bin-smoothed mode and its preceding left valley."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    signal = np.convolve(histogram, PEAK_KERNEL, mode="same")

    peaks: list[int] = []
    for index in range(1, len(signal) - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            peaks.append(index)
    # Saturation itself is allowed to be the rightmost histogram mode.
    if len(signal) >= 2 and signal[-1] > signal[-2]:
        peaks.append(len(signal) - 1)
    peak = max(peaks or [int(np.argmax(signal))])

    left_edge = 0
    for index in range(peak - 1, 0, -1):
        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:
            left_edge = index
            break

    return int(peak), int(left_edge)


def deepest_component_point(component_u8: np.ndarray) -> tuple[int, int]:
    """Choose an interior seed point; unlike a centroid this stays inside crescents."""
    source = (component_u8 != 0).astype(np.uint8)
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


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


def largest_enclosed_bright_component(binary: np.ndarray) -> np.ndarray | None:
    """Return the largest 8-connected bright component enclosed by the raster."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary != 0).astype(np.uint8), 8
    )
    height, width = binary.shape
    best = None
    for label in range(1, count):
        x, y, component_width, component_height, area = map(int, stats[label])
        if (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        ):
            continue
        candidate = (area, label)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return labels == best[1]


def _touches_image_border(component: np.ndarray | None) -> bool:
    """Return whether a tracked component is absent or reaches the image boundary."""
    if component is None or not np.any(component):
        return True
    return bool(
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    )


def coarse_threshold_search(work_gray: np.ndarray) -> CoarseThresholdResult:
    """Establish the solar seed, then track it to the lowest enclosed coarse T."""
    # Phase 1: histogram identity/start only. Descend from the left edge of the
    # rightmost mode until the first enclosed bright component identifies the Sun.
    peak, left_edge = find_rightmost_histogram_peak(work_gray)
    seed_threshold = None
    seed_point = None
    seed_mask = None

    for threshold in range(left_edge, -1, -1):
        component = largest_enclosed_bright_component(work_gray > threshold)
        if component is None:
            continue
        seed_threshold = int(threshold)
        seed_mask = np.where(component, 255, 0).astype(np.uint8)
        seed_point = brightest_supported_component_point(work_gray, seed_mask)
        break

    if seed_threshold is None or seed_point is None or seed_mask is None:
        raise ThresholdResolutionError("No enclosed bright component exists")

    # Phase 2: keep the same seed and lower T. Connectivity is monotone while T is
    # lowered: pixels are only added, so once this tracked 8-connected component
    # reaches the image border it cannot become enclosed again at a lower T.
    lowest_threshold = seed_threshold
    lowest_component = seed_mask != 0
    seed_x, seed_y = seed_point
    height, width = work_gray.shape

    for threshold in range(seed_threshold, -1, -1):
        binary = work_gray > threshold
        if not (0 <= seed_x < width and 0 <= seed_y < height) or not bool(binary[seed_y, seed_x]):
            continue

        flooded = cv2.compare(binary.astype(np.uint8), 0, cv2.CMP_GT)
        cv2.floodFill(flooded, None, seed_point, 128, flags=8)
        component = flooded == 128
        if _touches_image_border(component):
            break

        lowest_threshold = threshold
        lowest_component = component

    ys, xs = np.nonzero(lowest_component)
    if len(xs) == 0:
        raise ThresholdResolutionError("Empty component")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    component_area = int(len(xs))
    component_bbox = (x0, y0, x1 - x0, y1 - y0)

    return CoarseThresholdResult(
        histogram_peak=peak,
        histogram_left_edge=left_edge,
        seed_threshold=seed_threshold,
        seed_point=seed_point,
        seed_mask=seed_mask,
        threshold=int(lowest_threshold),
        component_mask=np.where(lowest_component, 255, 0).astype(np.uint8),
        component_area=component_area,
        component_bbox=component_bbox,
    )


def _map_interval(lo: int, hi: int, source_n: int, target_n: int) -> tuple[int, int]:
    out_lo = int(math.floor(lo * target_n / source_n))
    out_hi = int(math.ceil(hi * target_n / source_n))
    return max(0, out_lo), min(target_n, max(out_lo + 1, out_hi))


def establish_full_resolution_seed(
    full_gray: np.ndarray,
    work_shape: tuple[int, int],
    coarse_seed_mask: np.ndarray,
    seed_threshold: int,
) -> tuple[int, int]:
    """Map the coarse solar seed and choose a bright, well-supported source pixel."""
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

    source = candidate if np.any(candidate) else mapped
    if not np.any(source):
        raise ThresholdResolutionError("Mapped solar seed is empty")

    local = np.where(source, 255, 0).astype(np.uint8)
    px, py = brightest_supported_component_point(local_gray, local)
    return x0 + px, y0 + py


def find_full_resolution_enclosed_seed_component(
    full_gray: np.ndarray,
    coarse_threshold: int,
    seed_point: tuple[int, int],
):
    """Find an actual enclosed full-resolution component containing the solar seed."""
    start_threshold = max(0, int(coarse_threshold))
    seed_x, seed_y = seed_point

    for threshold in range(start_threshold, 256):
        binary = cv2.compare(full_gray, int(threshold), cv2.CMP_GT)
        if binary[seed_y, seed_x] == 0:
            continue

        cv2.floodFill(binary, None, seed_point, 128, flags=8)
        component = binary == 128
        if not _touches_image_border(component):
            return int(threshold), component

        # Usually resolved at the coarse T or one level higher. The loop remains
        # exhaustive so reduced-resolution mismatch cannot determine the final T.

    raise ThresholdResolutionError(
        "No enclosed original-resolution solar component"
    )


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


def _evaluate_observation_region(region: ObservationRegion, threshold: int):
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
    for threshold in range(0, 256):
        exists, touches_roi, touches_true = _evaluate_observation_region(roi, threshold)
        if not exists or touches_true:
            continue
        if not touches_roi:
            return int(threshold), used_guard

        used_guard = True
        exists, touches_guard, touches_true = _evaluate_observation_region(guard, threshold)
        if exists and not touches_true and not touches_guard:
            return int(threshold), used_guard

    raise ThresholdResolutionError("Tracked solar component never became separated")


def auto_threshold(gray: np.ndarray) -> AutoThresholdResult:
    """Select T from an authoritative 8-bit grayscale brightness image."""
    work_gray = resize_gray_max_dim(gray)
    peak, left_edge = find_rightmost_histogram_peak(work_gray)

    try:
        coarse = coarse_threshold_search(work_gray)
        full_seed = establish_full_resolution_seed(
            gray, work_gray.shape, coarse.seed_mask, coarse.seed_threshold
        )
        roi_seed_threshold, roi_seed_component = (
            find_full_resolution_enclosed_seed_component(
                gray, coarse.threshold, full_seed
            )
        )
        final_threshold, used_guard = find_lowest_full_threshold(
            gray,
            full_seed,
            roi_seed_component,
        )
        # The Auto-T seed is already authoritative here. Flood its exact
        # full-resolution component directly for topology-knee optimization rather
        # than routing through any later post-T geometry-establishment stage.
        separated_binary = cv2.compare(gray, int(final_threshold), cv2.CMP_GT)
        seed_x, seed_y = map(int, full_seed)
        if separated_binary[seed_y, seed_x] == 0:
            raise ThresholdResolutionError(
                f"Auto-T solar seed is not light at separated T={final_threshold}"
            )
        cv2.floodFill(separated_binary, None, (seed_x, seed_y), 128, flags=8)
        separated_component = separated_binary == 128
        if _touches_image_border(separated_component):
            raise ThresholdResolutionError(
                f"Auto-T solar component touches the image border at T={final_threshold}"
            )
        topology_selection = optimize_separated_threshold(
            gray,
            final_threshold,
            full_seed,
            separated_component,
        )
        return AutoThresholdResult(
            threshold=topology_selection.threshold,
            histogram_peak=coarse.histogram_peak,
            histogram_left_edge=coarse.histogram_left_edge,
            seed_threshold=coarse.seed_threshold,
            coarse_threshold=coarse.threshold,
            roi_seed_threshold=roi_seed_threshold,
            full_seed_point=full_seed,
            used_guard=used_guard,
            resolved=True,
        )
    except ThresholdResolutionError as exc:
        # Deterministic fallback: if component topology cannot be resolved, use
        # the left side of the rightmost histogram peak rather than introducing
        # color, Otsu, fixed-T, or ellipse-dependent heuristics.
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


# ---------------------------------------------------------------------------
# Post-threshold full-resolution solar data
# ---------------------------------------------------------------------------
REFINEMENT_KERNEL_SIZE = 7
REFINEMENT_ITERATIONS = 1


def refine_solar_component_mask(component: np.ndarray) -> np.ndarray:
    """Return a conservatively smoothed boolean solar-component mask.

    The opening suppresses small/thin outward burrs.  The following closing
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


@dataclass(frozen=True)
class SolarData:
    """Refined solar geometry established at exactly one full-resolution threshold T."""

    threshold: int
    seed_point: tuple[int, int]

    # Full-resolution masks encoded as np.packbits -> zlib level 1.
    component_mask: bytes
    roi_6_5_mask: bytes
    guard_19_5_mask: bytes

    # Ordered external full-resolution contour, shape (N, 2), dtype uint16.
    component_contour: np.ndarray


def compress_full_mask(mask: np.ndarray) -> bytes:
    """Pack a full-resolution boolean mask to one bit/pixel, then zlib level 1."""
    mask_bool = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask_bool.reshape(-1))
    return zlib.compress(packed.tobytes(), level=1)


def decompress_full_mask(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Restore a compressed full-resolution mask using the current image shape."""
    if len(shape) != 2:
        raise ValueError("mask shape must be (height, width)")
    height, width = map(int, shape)
    if height <= 0 or width <= 0:
        raise ValueError("mask shape dimensions must be positive")

    pixel_count = height * width
    raw = zlib.decompress(payload)
    expected_bytes = (pixel_count + 7) // 8
    if len(raw) != expected_bytes:
        raise ValueError(
            f"compressed mask has {len(raw)} packed bytes; expected {expected_bytes}"
        )

    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed, count=pixel_count)
    return bits.reshape((height, width)).astype(bool)


def _full_mask_from_observation_region(
    full_gray: np.ndarray,
    component: np.ndarray,
    margin: float,
    seed_point: tuple[int, int],
) -> np.ndarray:
    """Promote the existing cropped Euclidean observation dilation to full size."""
    region = _build_observation_region(full_gray, component, margin, seed_point)
    full_mask = np.zeros(full_gray.shape, dtype=bool)
    x0, y0, x1, y1 = region.bbox
    full_mask[y0:y1, x0:x1] = region.allowed_u8 != 0
    return full_mask


def _ordered_external_component_contour(component: np.ndarray) -> np.ndarray:
    """Return the ordered external CHAIN_APPROX_NONE contour as uint16 XY pairs."""
    component_u8 = np.where(component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ThresholdResolutionError("Solar component has no external contour")

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour.size == 0:
        raise ThresholdResolutionError("Solar component external contour is empty")
    if int(contour.max()) > np.iinfo(np.uint16).max:
        raise ThresholdResolutionError("Solar contour coordinates exceed uint16 range")
    return contour.astype(np.uint16, copy=False)


def build_solar_data_at_threshold(
    full_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Establish, refine, and persist SolarData at exactly one already-selected T.

    At the exact resolved automatic threshold, the Auto-T full-resolution seed is
    authoritative and must survive refinement. At every other T, a freshly
    calculated seed is used only to identify/flood the current-T raw component; the
    finalized refined component then receives its own robust interior seed. This
    prevents a valid manual/current-T component from failing merely because 7x7
    cleanup removed the particular pre-cleanup flood seed.
    """
    threshold = int(threshold)
    if full_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")

    existing = image_state.get("solar_data")
    if isinstance(existing, SolarData) and existing.threshold == threshold:
        return decompress_full_mask(existing.component_mask, full_gray.shape)

    auto_result = image_state.get("auto_threshold_result")
    if not isinstance(auto_result, AutoThresholdResult):
        raise ValueError("image state has no AutoThresholdResult")

    height, width = full_gray.shape
    exact_auto_threshold = bool(
        auto_result.resolved and threshold == int(auto_result.threshold)
    )

    # T is already known. Establish an identity/flood seed first. The exact
    # resolved Auto T owns its stored authoritative seed. Every other T establishes
    # solar identity and calculates a fresh seed at that exact threshold.
    if exact_auto_threshold:
        if auto_result.full_seed_point is None:
            raise ThresholdResolutionError(
                "Resolved Auto T has no full-resolution solar seed"
            )
        seed_x, seed_y = map(int, auto_result.full_seed_point)
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError(
                "Stored Auto-T solar seed lies outside the image"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                "Stored Auto-T solar seed is not light at its Auto T"
            )
    else:
        work_gray = resize_gray_max_dim(full_gray)
        work_component = largest_enclosed_bright_component(work_gray > threshold)
        if work_component is None:
            raise ThresholdResolutionError(
                f"No enclosed solar component exists at selected T={threshold}"
            )
        work_mask = np.where(work_component, 255, 0).astype(np.uint8)
        seed_x, seed_y = establish_full_resolution_seed(
            full_gray,
            work_gray.shape,
            work_mask,
            threshold,
        )
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError(
                "Calculated solar seed lies outside the image"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                f"Calculated solar seed is not light at threshold T={threshold}"
            )

    binary = cv2.compare(full_gray, threshold, cv2.CMP_GT)
    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    raw_component = binary == 128
    if _touches_image_border(raw_component):
        raise ThresholdResolutionError(
            f"Seeded solar component touches the image border at T={threshold}"
        )

    refined_component = refine_solar_component_mask(raw_component)
    if not np.any(refined_component):
        raise ThresholdResolutionError(
            f"Solar component was removed by refinement at T={threshold}"
        )

    if exact_auto_threshold:
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Refined solar component no longer contains the Auto-T solar seed"
            )
    else:
        # The current-T flood seed established component identity before cleanup.
        # Once cleanup has finalized that same component, seed the authoritative
        # SolarData from the finalized geometry rather than requiring one raw-mask
        # pixel to survive morphology.
        refined_u8 = np.where(refined_component, 255, 0).astype(np.uint8)
        seed_x, seed_y = brightest_supported_component_point(full_gray, refined_u8)
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Calculated refined-component seed lies outside the solar component"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                f"Calculated refined-component seed is not light at T={threshold}"
            )

    image_scale = math.sqrt(float(width) * float(height))
    roi_6_5 = _full_mask_from_observation_region(
        full_gray,
        refined_component,
        ROI_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
    )
    guard_19_5 = _full_mask_from_observation_region(
        full_gray,
        refined_component,
        GUARD_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
    )
    contour = _ordered_external_component_contour(refined_component)

    solar_data = SolarData(
        threshold=threshold,
        seed_point=(seed_x, seed_y),
        component_mask=compress_full_mask(refined_component),
        roi_6_5_mask=compress_full_mask(roi_6_5),
        guard_19_5_mask=compress_full_mask(guard_19_5),
        component_contour=contour,
    )

    # No partially constructed state is published if any step above fails.
    image_state["solar_data"] = solar_data
    return refined_component


class DetectorApp:
    """Own GUI state, per-image settings, cached auto thresholds, and previews.

    Public methods represent application actions. Underscore-prefixed methods are
    Tk callback or rendering internals. Automatic threshold selection remains a
    stateless image algorithm outside this class; this class only caches its result
    per image and manages the effective processing settings shown by the controls.
    """

    def __init__(self, root: tk.Tk, image_paths: list[str], args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path: str | None = None
        self.color_image = None
        self.gray_image = None
        self.threshold_preview = transparent_bgra()

        # Per-image state keeps sparse settings, the cached automatic threshold
        # result, and post-threshold SolarData when solar geometry has been built.
        # No ImageState wrapper class is needed: the outer dictionary directly
        # expresses the image-to-state hierarchy.
        self.image_state: dict[str, dict[str, object]] = {}

        self.preview_redraw_job = None
        self.threshold_refinement_job = None

        # Keyboard auto-repeat can emit intermediate release/press pairs on some
        # Tk platforms. Keep one deferred refresh job so only the final key-up
        # commits a slider-driven preview refresh.
        self.slider_keyboard_commit_job = None
        self.slider_keyboard_widget = None
        self.slider_keyboard_start_value = None

        self.threshold = tk.IntVar(value=args.threshold)
        self.min_radius = tk.IntVar(value=round(args.min_radius))
        self.max_radius = tk.IntVar(value=round(args.max_radius))
        self.max_error = tk.DoubleVar(value=args.max_error * 100.0)
        self.min_coverage = tk.IntVar(value=round(args.min_coverage * 100.0))
        self.morphology = tk.BooleanVar(value=False)
        self.outer_limb_assistance = tk.BooleanVar(value=False)
        self.use_horizon = tk.BooleanVar(value=True)

        # Mutually exclusive by construction: both Radiobuttons share this one
        # StringVar. Light is the requested default.
        self.center_target = tk.StringVar(value="light")
        self.center_preview_label = tk.StringVar()

        # Every processing control is per-image. Ordinary controls use these values
        # as their baseline; threshold instead uses the cached automatic T for the
        # current image. Only deviations from those baselines are stored.
        self.default_settings = ImageSettings(
            min_radius=int(self.min_radius.get()),
            max_radius=int(self.max_radius.get()),
            max_error=float(self.max_error.get()),
            min_coverage=int(self.min_coverage.get()),
            morphology=bool(self.morphology.get()),
            outer_limb_assistance=bool(self.outer_limb_assistance.get()),
            use_horizon=bool(self.use_horizon.get()),
            center_target=self.center_target.get(),
        )
        self.setting_variables = {
            "threshold": self.threshold,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "max_error": self.max_error,
            "min_coverage": self.min_coverage,
            "morphology": self.morphology,
            "outer_limb_assistance": self.outer_limb_assistance,
            "use_horizon": self.use_horizon,
            "center_target": self.center_target,
        }

        self.status = tk.StringVar(value="Threshold finder integrated. Load images to inspect automatic T selection.")
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector — threshold finder")
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_navigation_bar()
        self._build_settings_panel()
        self._build_preview_panes()
        self._update_center_preview_label()

        # Tk's toplevel bindtag receives mouse events from every child widget.
        # Use it to clear slider keyboard focus as soon as the user clicks
        # anywhere outside the currently focused slider.
        root.bind("<ButtonPress-1>", self._release_slider_focus_if_clicked_elsewhere, add="+")
        root.bind("<Return>", lambda _event: self.apply_full_resolution())
        root.bind("<Escape>", lambda _event: self.close())

        if self.image_paths:
            self.load_image_at(0)
        else:
            self.update_navigation_state()

    # ------------------------------------------------------------------
    # Image list / navigation (GUI support only)
    # ------------------------------------------------------------------
    def load_images(self):
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Select eclipse images",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not selected:
            return
        self.image_paths = [os.path.abspath(path) for path in selected]
        self.current_index = -1
        self.current_path = None
        self.color_image = None
        self.gray_image = None
        self.threshold_preview = transparent_bgra()
        self.load_image_at(0)

    def load_image_at(self, index: int):
        if not 0 <= index < len(self.image_paths):
            return

        path = self.image_paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.current_index = index
        self.current_path = path

        if image is None:
            self.color_image = None
            self.gray_image = None
            self.threshold_preview = transparent_bgra()
            self._redraw_previews()
            self.status.set(f"Could not load image: {path}")
        else:
            # Convert the original image to authoritative grayscale ONCE. The auto
            # threshold module derives its 1200px INTER_AREA working raster from
            # this grayscale image; no HSV/color threshold path exists.
            self.color_image = opaque_bgra(image)
            self.gray_image = to_gray(image)

            restored = path in self.image_state
            if restored:
                state = self.image_state[path]
                result = state["auto_threshold_result"]
                state.setdefault("solar_data", None)
            else:
                result = auto_threshold(self.gray_image)
                state = {
                    "settings": ImageSettings(),
                    "auto_threshold_result": result,
                    "solar_data": None,
                }
                self.image_state[path] = state

            settings = state["settings"]
            for setting_name, variable in self.setting_variables.items():
                if setting_name == "threshold":
                    baseline = int(result.threshold)
                else:
                    baseline = getattr(self.default_settings, setting_name)
                override = getattr(settings, setting_name)
                variable.set(baseline if override is None else override)

            self._update_center_preview_label()
            selected_threshold = int(self.threshold.get())

            # Stage 1: show the pure threshold result immediately.
            self.threshold_preview = self.gray_image > selected_threshold
            self._refresh_display_images()

            # Stage 2: establish the seed/component/SolarData at this exact T and
            # leave the final refined component as the persistent threshold canvas.
            try:
                refined_component = build_solar_data_at_threshold(
                    self.gray_image,
                    selected_threshold,
                    state,
                )
            except (ThresholdResolutionError, ValueError) as exc:
                self.status.set(
                    f"Threshold T={selected_threshold} is visible, but SolarData could not "
                    f"be established ({exc}). Adjust T if needed."
                )
            else:
                self.threshold_preview = refined_component
                if hasattr(self, "threshold_canvas"):
                    self.display_on_canvas(
                        self.threshold_canvas,
                        self.threshold_preview,
                    )

                if restored:
                    self.status.set(
                        f"Image loaded. Restored per-image settings at T={selected_threshold}; "
                        "final refined SolarData mask is current."
                    )
                elif not result.resolved:
                    self.status.set(
                        "Automatic component tracking was unresolved; using "
                        f"rightmost-histogram left edge T={selected_threshold}. SolarData was "
                        "established directly at that T and the refined component is displayed."
                    )
                else:
                    self.status.set(
                        "Image loaded. Automatic grayscale threshold "
                        f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                        f"histogram start={result.histogram_left_edge}); final refined SolarData "
                        "mask displayed."
                    )

        self.update_navigation_state()

    def previous_image(self):
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_image(self):
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    def update_navigation_state(self):
        count = len(self.image_paths)
        has_current = 0 <= self.current_index < count
        readable = has_current and self.color_image is not None

        self.previous_button.config(
            state=tk.NORMAL if has_current and self.current_index > 0 else tk.DISABLED
        )
        self.next_button.config(
            state=tk.NORMAL if has_current and self.current_index < count - 1 else tk.DISABLED
        )
        self.preview_button.config(state=tk.NORMAL if readable else tk.DISABLED)
        self.full_button.config(state=tk.NORMAL if readable else tk.DISABLED)

        if has_current:
            self.image_info.set(
                f"{self.current_index + 1} / {count}   {os.path.basename(self.current_path or '')}"
            )
        else:
            self.image_info.set("No image loaded" if not count else f"0 / {count}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_navigation_bar(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(4, weight=1)

        tk.Button(frame, text="Load images...", width=14, command=self.load_images).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.save_centered_button = tk.Button(
            frame,
            text="Save centered images",
            width=20,
            command=self.save_centered_images,
        )
        self.save_centered_button.grid(row=0, column=1, padx=(0, 10))
        self.previous_button = tk.Button(
            frame, text="◀ Previous", width=12, command=self.previous_image
        )
        self.previous_button.grid(row=0, column=2, padx=(0, 5))
        self.next_button = tk.Button(
            frame, text="Next ▶", width=12, command=self.next_image
        )
        self.next_button.grid(row=0, column=3, padx=(0, 10))
        tk.Label(frame, textvariable=self.image_info, anchor="w").grid(
            row=0, column=4, sticky="ew"
        )

    def _build_settings_panel(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        radius_limit = max(1600, round(max(self.args.max_radius, self.args.min_radius) * 1.5))

        slider_specs = [
            ("threshold", "Brightness threshold (dark <= T, light > T)", self.threshold,
             0, 255, 1, lambda v: str(int(float(v)))),
            ("min_radius", "Minimum FINAL fitted semi-axis radius (px)", self.min_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("max_radius", "Maximum FINAL fitted semi-axis radius (px)", self.max_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("max_error", "Maximum average normalized ellipse error (%)", self.max_error,
             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),
            ("min_coverage", "Minimum TOTAL supported ellipse arc (%)", self.min_coverage,
             0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(slider_specs):
            self._add_slider(frame, row, *spec)

        # Threshold Auto select is implemented; radius Auto select remains a placeholder. Threshold has
        # its own button; radius selection is a single operation represented by a
        # button spanning the paired minimum/maximum radius rows.
        self.threshold_auto_button = tk.Button(
            frame,
            text="Auto select",
            width=12,
            command=self.auto_select_threshold,
        )
        self.threshold_auto_button.grid(
            row=0, column=3, sticky="ns", padx=(10, 0), pady=2
        )
        self.radius_auto_button = tk.Button(
            frame,
            text="Auto select",
            width=12,
            command=self.auto_select_radius,
        )
        self.radius_auto_button.grid(
            row=1, column=3, rowspan=2, sticky="nsew", padx=(10, 0), pady=2
        )

        options = tk.Frame(frame)
        options.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 3))
        tk.Checkbutton(
            options,
            text="Morphology cleanup for candidate search",
            variable=self.morphology,
            command=lambda: self._commit_setting_change("morphology"),
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=lambda: self._commit_setting_change("outer_limb_assistance"),
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=lambda: self._commit_setting_change("use_horizon"),
            state=tk.DISABLED,
        )
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")

        # New requested alignment target. Sharing center_target makes these two
        # controls mutually exclusive without extra synchronization logic.
        center_frame = tk.Frame(frame)
        center_frame.grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 5))
        tk.Label(center_frame, text="Center full-color image on:").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        tk.Radiobutton(
            center_frame,
            text="Light ellipse",
            variable=self.center_target,
            value="light",
            command=self._handle_center_target_change,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))
        tk.Radiobutton(
            center_frame,
            text="Dark ellipse",
            variable=self.center_target,
            value="dark",
            command=self._handle_center_target_change,
        ).grid(row=0, column=2, sticky="w")

        button_frame = tk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.preview_button = tk.Button(
            button_frame,
            text="Refresh Preview",
            width=16,
            command=self.refresh_preview,
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 8))
        self.full_button = tk.Button(
            button_frame,
            text="Apply Full Resolution",
            width=20,
            command=self.apply_full_resolution,
        )
        self.full_button.grid(row=0, column=1)

        tk.Label(
            frame,
            textvariable=self.status,
            anchor="w",
            justify="left",
            wraplength=1150,
        ).grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))

    def _add_slider(self, parent, row, setting_name, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=42, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        slider = tk.Scale(
            parent,
            from_=low,
            to=high,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            showvalue=False,
            length=420,
            takefocus=True,
            highlightthickness=1,
        )
        slider.grid(row=row, column=1, sticky="ew", pady=2)
        slider._setting_name = setting_name

        # Tk Scale supports precise arrow-key adjustment while it owns keyboard
        # focus. Value traces update only the label. Preview refresh is deliberately
        # deferred until the user finishes the mouse or keyboard interaction.
        slider.bind("<ButtonPress-1>", self._focus_slider, add="+")
        slider.bind("<ButtonPress-1>", self._begin_slider_mouse_change, add="+")
        slider.bind("<ButtonRelease-1>", self._focus_slider, add="+")
        slider.bind("<ButtonRelease-1>", self._finish_slider_mouse_change, add="+")
        slider.bind("<KeyPress>", self._begin_slider_keyboard_change, add="+")
        slider.bind("<KeyRelease>", self._schedule_slider_keyboard_commit, add="+")
        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            # Do not refresh here: this trace fires continuously while the Scale is
            # dragged or while an arrow key auto-repeats.
            value_label.config(text=formatter(variable.get()))

        variable.trace_add("write", update_value)
        update_value()

    @staticmethod
    def _focus_slider(event):
        """Keep a clicked slider focused so arrow keys continue to adjust it."""
        event.widget.focus_set()

    @staticmethod
    def _begin_slider_mouse_change(event):
        """Remember the value before a mouse slider interaction begins."""
        event.widget._preview_mouse_start_value = event.widget.get()

    def _finish_slider_mouse_change(self, event):
        """Refresh once after a mouse slider change has actually finished."""
        start_value = getattr(event.widget, "_preview_mouse_start_value", None)
        if start_value is not None and event.widget.get() != start_value:
            self._commit_setting_change(event.widget._setting_name)

    def _cancel_pending_slider_keyboard_commit(self):
        """Cancel a release callback superseded by continuing keyboard input."""
        if self.slider_keyboard_commit_job is not None:
            self.root.after_cancel(self.slider_keyboard_commit_job)
            self.slider_keyboard_commit_job = None

    def _begin_slider_keyboard_change(self, event):
        """Begin or continue one keyboard slider interaction."""
        self._cancel_pending_slider_keyboard_commit()
        if self.slider_keyboard_widget is not event.widget:
            self.slider_keyboard_widget = event.widget
            self.slider_keyboard_start_value = event.widget.get()

    def _schedule_slider_keyboard_commit(self, event):
        """Schedule completion after a KeyRelease survives the repeat window."""
        if self.slider_keyboard_widget is not event.widget:
            self.slider_keyboard_widget = event.widget
            self.slider_keyboard_start_value = event.widget.get()
        self._cancel_pending_slider_keyboard_commit()
        self.slider_keyboard_commit_job = self.root.after(
            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change
        )

    def _finish_slider_keyboard_change(self):
        """Commit one preview refresh after keyboard slider input becomes idle."""
        self.slider_keyboard_commit_job = None
        widget = self.slider_keyboard_widget
        start_value = self.slider_keyboard_start_value
        self.slider_keyboard_widget = None
        self.slider_keyboard_start_value = None
        if widget is not None and start_value is not None and widget.get() != start_value:
            self._commit_setting_change(widget._setting_name)

    def _release_slider_focus_if_clicked_elsewhere(self, event):
        """Release slider focus immediately when the mouse clicks elsewhere.

        Clicking the focused slider itself keeps focus. Clicking a different slider
        transfers focus through that slider's own ButtonPress binding, so this
        handler also leaves it alone. Any non-slider click removes the keyboard
        focus ring from the previously focused slider.
        """
        focused = self.root.focus_get()
        if isinstance(focused, tk.Scale) and event.widget is not focused:
            self.root.focus_set()

    def _build_preview_panes(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(
            frame,
            text="Threshold preview: arcs / ellipses / detected horizon",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            textvariable=self.center_preview_label,
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        # The canvas is only a display surface. Transparency is retained in the
        # BGRA preview raster itself; it is not simulated by matching Tk colors.
        self.threshold_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.color_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.threshold_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.color_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.threshold_canvas.bind("<Configure>", self._schedule_preview_redraw)
        self.color_canvas.bind("<Configure>", self._schedule_preview_redraw)

    # ------------------------------------------------------------------
    # Application actions and threshold preview
    # ------------------------------------------------------------------
    def save_centered_images(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the threshold-finder stage."
        )

    def auto_select_threshold(self):
        """Restore the cached image-only automatic T and process that threshold."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        result = state["auto_threshold_result"]
        if result is None:
            result = auto_threshold(self.gray_image)
            state["auto_threshold_result"] = result

        selected_threshold = int(result.threshold)
        self.threshold.set(selected_threshold)
        self._commit_setting_change("threshold")

        solar_data = state.get("solar_data")
        if isinstance(solar_data, SolarData) and solar_data.threshold == selected_threshold:
            if result.resolved:
                self.status.set(
                    "Automatic grayscale threshold selected: "
                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                    f"histogram start={result.histogram_left_edge}); refined component displayed."
                )
            else:
                self.status.set(
                    "Automatic component tracking unresolved; "
                    f"using rightmost-histogram left edge T={selected_threshold}; "
                    "SolarData established directly at that T."
                )

    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the threshold-finder stage."
        )

    def _refresh_display_images(self):
        """Redraw the currently stored preview rasters without processing them."""
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.display_on_canvas(self.color_canvas, image)

    def _commit_setting_change(self, setting_name):
        """Persist one completed setting change, then refresh the current preview."""
        if self.current_path is None or self.current_path not in self.image_state:
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        value = self.setting_variables[setting_name].get()

        if setting_name == "threshold":
            baseline = int(state["auto_threshold_result"].threshold)
        else:
            baseline = getattr(self.default_settings, setting_name)

        setattr(settings, setting_name, None if value == baseline else value)

        # A completed control interaction is the processing boundary. Slider value
        # labels may update continuously while dragging/auto-repeating, but the
        # expensive preview refresh happens once when that interaction is committed.
        if self.gray_image is not None:
            self.refresh_preview(changed_setting=setting_name)

    def _selected_center_target_name(self):
        """Return the user-facing name of the selected centering target."""
        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"

    def _handle_center_target_change(self):
        self._update_center_preview_label()
        self._commit_setting_change("center_target")
        self.status.set(
            f"Centering target set to {self._selected_center_target_name()}. "
            "Actual centering will be implemented with ellipse detection."
        )

    def _update_center_preview_label(self):
        target = self._selected_center_target_name()
        self.center_preview_label.set(f"Full-color image — center on {target}")

    def refresh_preview(self, changed_setting: str | None = None, full_resolution: bool = False):
        """Show pure current-T threshold now, then refine it on the next GUI turn.

        Separating the two visual stages through ``after`` guarantees that Tk can
        paint the raw ``gray > T`` result before SolarData construction replaces it
        with the finalized refined component. Any pending stage-2 job is cancelled
        when a newer control commit arrives, so stale T values cannot win a race.
        """
        if self.gray_image is None or self.current_path is None:
            action = "Apply Full Resolution" if full_resolution else "Refresh Preview"
            self.status.set(f"{action}: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        threshold = int(self.threshold.get())
        existing = state.get("solar_data")
        reused = isinstance(existing, SolarData) and existing.threshold == threshold

        # Never expose SolarData from a different T as current while the new T is
        # waiting for its stage-2 rebuild. This also makes downstream state ownership
        # explicit if current-T establishment later fails.
        if isinstance(existing, SolarData) and existing.threshold != threshold:
            state["solar_data"] = None

        self.threshold_preview = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        if self.threshold_refinement_job is not None:
            self.root.after_cancel(self.threshold_refinement_job)
            self.threshold_refinement_job = None

        if full_resolution:
            self.status.set(
                f"Pure full-resolution threshold displayed at T={threshold}; refining current-T component."
            )
        elif changed_setting is not None:
            self.status.set(
                f"{changed_setting} committed; pure threshold displayed at T={threshold}; "
                "refining current preview."
            )
        else:
            self.status.set(
                f"Pure threshold displayed at T={threshold}; refining current preview."
            )

        self.threshold_refinement_job = self.root.after(
            THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS,
            self._finish_threshold_preview_refresh,
            self.current_path,
            threshold,
            reused,
            changed_setting,
            full_resolution,
        )

    def _finish_threshold_preview_refresh(
        self,
        path: str,
        threshold: int,
        reused: bool,
        changed_setting: str | None,
        full_resolution: bool,
    ):
        """Build current-T SolarData and publish the finalized refined preview."""
        self.threshold_refinement_job = None

        # A navigation or newer threshold change supersedes this queued stage.
        if self.current_path != path or self.gray_image is None:
            return
        if int(self.threshold.get()) != int(threshold):
            return

        state = self.image_state[path]
        try:
            refined_component = build_solar_data_at_threshold(
                self.gray_image,
                int(threshold),
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure threshold remains displayed at T={threshold}; SolarData could not be "
                f"established ({exc})."
            )
            return

        self.threshold_preview = refined_component
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        if full_resolution:
            self.status.set(
                "Full-resolution refined solar-component preview applied at "
                f"T={threshold}. Ellipse detector backend not implemented yet."
            )
        elif changed_setting is not None:
            self.status.set(
                f"{changed_setting} committed; refined current preview displayed at T={threshold}."
            )
        elif reused:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; existing SolarData reused."
            )
        else:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; SolarData rebuilt for this T."
            )

    def apply_full_resolution(self):
        if self.gray_image is None or self.current_path is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return
        self.refresh_preview(full_resolution=True)

    def _schedule_preview_redraw(self, _event=None):
        if self.preview_redraw_job is not None:
            self.root.after_cancel(self.preview_redraw_job)
        self.preview_redraw_job = self.root.after(
            PREVIEW_REDRAW_DELAY_MS, self._redraw_previews
        )

    def _redraw_previews(self):
        self.preview_redraw_job = None
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.display_on_canvas(self.color_canvas, image)

    def display_on_canvas(self, canvas, canvas_content):
        """Display supplied image/mask content on one canvas and flush pending repaint work."""
        if canvas_content is None:
            display = transparent_bgra()
        else:
            display = np.asarray(canvas_content)
            if display.dtype == bool:
                display = np.where(display, 255, 0).astype(np.uint8)
            elif display.dtype != np.uint8:
                display = np.clip(display, 0, 255).astype(np.uint8)

            if display.ndim == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGRA)
                display[:, :, 3] = 255
            elif display.ndim != 3 or display.shape[2] not in (3, 4):
                raise ValueError("canvas content must be a 2D mask or 3/4-channel image")

        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = display.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(display, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            raise ValueError("could not encode canvas content")

        photo = tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"),
            format="png",
        )
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2 + 1,
            canvas_height // 2 + 1,
            image=photo,
            anchor="center",
        )
        canvas._display_photo = photo

        # Let staged processing become visible without opening arbitrary event
        # callbacks while image-processing state is only partly updated.
        self.root.update_idletasks()

    def close(self):
        self.root.destroy()


def build_parser():
    parser = argparse.ArgumentParser(description="threshold-finder stage for the eclipse detector rebuild.")
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional ordered input image list; more images can be loaded in the GUI",
    )
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--min-radius", type=float, default=1000.0)
    parser.add_argument("--max-radius", type=float, default=1500.0)
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    return parser


def validate_args(args, parser):
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be 0..255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be > 0")
    if args.max_radius <= 0:
        parser.error("--max-radius must be > 0")
    if args.max_radius < args.min_radius:
        parser.error("--max-radius must be >= --min-radius")
    if args.max_error <= 0:
        parser.error("--max-error must be > 0")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be 0..1")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    root = tk.Tk()
    DetectorApp(root, args.images, args)
    root.mainloop()


if __name__ == "__main__":
    main()
