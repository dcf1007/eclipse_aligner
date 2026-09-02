"""Eclipse alignment GUI with grayscale automatic threshold selection.

This module combines the application's user-interface foundation with the tested
image-only automatic threshold finder. The GUI owns image navigation, per-image
processing settings, control interaction, preview lifecycle, and cached automatic
threshold results. Automatic threshold acquisition accepts the authoritative 8-bit
grayscale brightness image plus the current per-image state, stores its
``AutoThresholdResult`` there, and returns the selected integer T. Color-to-grayscale
conversion is an input-stage responsibility and is performed before the threshold
algorithm is called.

All processing controls are per-image. ``DetectorApp.image_state`` is keyed by the
absolute image path. ``settings.threshold`` is special: ``None`` means the threshold
has never been initialized, and once initialized its exact integer value is always
stored regardless of whether it came from Auto T or manual input. Other controls may
still use sparse overrides relative to application defaults.

Slider labels update continuously, but a setting is committed only after mouse
release or the final keyboard key release. Checkboxes and radio buttons commit
immediately. Every actual setting change passes through
``commit_setting_change(setting_name, value)``, which synchronizes the Tk variable,
persists the per-image setting, invalidates incompatible derived state, and invokes
``refresh_preview()`` once.

Automatic thresholding is only the T-acquisition stage. ``find_auto_threshold()``
writes ``AutoThresholdResult`` into the current image state and returns T; it does
not build SolarData. ``refresh_preview()`` first displays the pure full-resolution
``gray > T`` raster, then sequentially calls ``resolve_threshold()``. That one atomic
final-T function identifies the full-resolution solar component, selects the
authoritative seed from the unrefined component, applies 7x7 elliptical OPEN/CLOSE,
validates that the seed survives, constructs/stores SolarData from that same refined
mask, and returns the mask for final display. Auto, manual, and restored thresholds
therefore share exactly the same final-T path. ``full_resolution`` remains a future
placeholder; both preview modes use full resolution today.

The threshold algorithm uses authoritative 8-bit grayscale with fixed semantics
``dark = gray <= T`` and ``light = gray > T``. It derives a <=1200-pixel working
raster, starts from the left valley of the rightmost locally smoothed histogram mode,
establishes one 5x5-supported work-resolution solar seed, and tracks that same
8-connected component downward. The tracked work-resolution component is resized to
full resolution only to delimit the full-resolution seed search and a rounded 10%
guard. Auto-T then starts from the work-resolution T and searches only in the
monotonic direction needed to find the lowest full-resolution threshold whose seeded
component stays inside that fixed guard. If no supported work-resolution seed can be
established through T=0, Auto-T fails explicitly instead of substituting a histogram
fallback. No HSV/color thresholding, Otsu thresholding, ellipse-fit score,
bright-pixel dominance, competitor gain, or horizon special case is part of
automatic threshold selection.
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
CANVAS_REDRAW_DELAY_MS = 60


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
    """Per-image settings. Threshold ``None`` means never initialized; other ``None`` values use defaults."""

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
AUTO_T_GUARD_DILATION_FRACTION = 0.10
TOPOLOGY_OPTIMIZATION_STEPS = 5
TRACKING_SEED_KERNEL_SIZE = 5
SOLAR_COMPONENT_KERNEL_SIZE = 7
SOLAR_COMPONENT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (SOLAR_COMPONENT_KERNEL_SIZE, SOLAR_COMPONENT_KERNEL_SIZE),
)
REFINEMENT_ITERATIONS = 1


@dataclass(frozen=True)
class ThresholdTopology:
    threshold: int
    area: int
    contour_n: int
    perimeter: float
    roughness: float
    solidity: float
    internal_dark_fraction: float = 0.0


@dataclass(frozen=True)
class CleanupMetrics:
    """Unweighted cleanup benefit/cost terms measured against one raw component."""

    contour_cleanup: float
    roughness_cleanup: float
    solidity_gain: float
    internal_dark_cleanup: float
    area_loss: float
    solidity_loss: float


@dataclass(frozen=True)
class CleanupCandidateEvaluation:
    name: str
    mask: np.ndarray
    topology: ThresholdTopology
    metrics: CleanupMetrics


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

    # Internal dark fraction is measured in raster pixels, not contourArea. Fill
    # the selected external contour, then count original component pixels (N1) and
    # missing/dark pixels (N0) inside that fill.
    filled = np.zeros(component.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    filled_bool = filled != 0
    n1 = int(np.count_nonzero(component & filled_bool))
    n0 = int(np.count_nonzero(filled_bool & ~component))
    total = n0 + n1
    internal_dark_fraction = float(n0 / total) if total else 0.0

    return ThresholdTopology(
        threshold=int(threshold),
        area=area,
        contour_n=contour_n,
        perimeter=perimeter,
        roughness=roughness,
        solidity=solidity,
        internal_dark_fraction=internal_dark_fraction,
    )


def topology_trajectory_from_separated_component(
    full_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> tuple[ThresholdTopology, ...]:
    """Measure exact seeded topology for T..T+max_delta in the base-component crop."""
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
    """Select the first cleanup-versus-erosion knee in the topology trajectory."""
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
        selected_index = int(np.argmax(knee))

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
    """Optimize a proven separated threshold without lowering or invalidating it."""
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
class AutoThresholdResult:
    threshold: int | None
    histogram_start_threshold: int
    work_res_threshold: int | None
    full_res_seed_point: tuple[int, int] | None
    resolved: bool
    reason: str = ""


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert once to authoritative 8-bit grayscale."""
    if image.ndim == 2:
        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_img(
    img: np.ndarray,
    size: tuple[int | None, int | None],
) -> np.ndarray:
    """Resize raster to size, preserving aspect ratio if one dimension is omitted and dtype."""

    original_dtype = img.dtype

    h, w = img.shape[:2]
    width, height = size

    if width is None and height is None:
        raise ValueError("resize size must provide width, height, or both")
    if width is None:
        height = int(height)
        if height <= 0:
            raise ValueError("resize height must be positive")
        width = round(w * height / h)
    elif height is None:
        width = int(width)
        if width <= 0:
            raise ValueError("resize width must be positive")
        height = round(h * width / w)
    else:
        width = int(width)
        height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")

    if np.all((img == img.min()) | (img == img.max())):
        img = img.astype(np.uint8)
        interpolation = cv2.INTER_NEAREST_EXACT
    elif width < w:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4

    resized = cv2.resize(
        img,
        (width, height),
        interpolation=interpolation,
    )

    return resized.astype(original_dtype)


def find_histogram_start_threshold(work_res_gray: np.ndarray) -> int:
    """Return the left valley preceding the rightmost 3-bin-smoothed histogram mode."""
    histogram = np.bincount(work_res_gray.ravel(), minlength=256).astype(np.float64)
    signal = np.convolve(histogram, PEAK_KERNEL, mode="same")

    peaks: list[int] = []
    for index in range(1, len(signal) - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            peaks.append(index)
    if len(signal) >= 2 and signal[-1] > signal[-2]:
        peaks.append(len(signal) - 1)
    rightmost_peak = max(peaks or [int(np.argmax(signal))])

    for index in range(rightmost_peak - 1, 0, -1):
        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:
            return int(index)
    return 0


def brightest_supported_component_point(
    gray: np.ndarray,
    component: np.ndarray,
    support_kernel: np.ndarray,
) -> tuple[int, int] | None:
    """Return the brightest support-eligible component pixel; depth breaks ties.

    The caller owns support geometry and the meaning of an unavailable point. Empty
    or unsupported components return ``None``; malformed caller inputs remain errors.
    """
    source = (np.asarray(component) != 0).astype(np.uint8)
    support_kernel = np.asarray(support_kernel, dtype=np.uint8)
    if gray.shape != source.shape:
        raise ValueError("gray and component must have identical shapes")
    if support_kernel.ndim != 2 or support_kernel.size == 0 or not np.any(support_kernel):
        raise ValueError("support kernel must be a non-empty two-dimensional mask")
    if not np.any(source):
        return None

    supported = cv2.erode(source, support_kernel, iterations=1) != 0
    if not np.any(supported):
        return None

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


def find_work_res_solar_component(
    work_res_gray: np.ndarray,
    start_T: int,
    work_res_seed_kernel: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Establish one supported work-res seed, then track only that identity downward."""
    seed: tuple[int, int] | None = None
    work_res_T: int | None = None
    work_res_component: np.ndarray | None = None
    height, width = work_res_gray.shape

    for threshold in range(int(start_T), -1, -1):
        if seed is None:
            component = largest_enclosed_bright_component(work_res_gray > threshold)
            if component is not None:
                seed = brightest_supported_component_point(
                    work_res_gray,
                    component,
                    work_res_seed_kernel,
                )
                if seed is not None:
                    work_res_T = int(threshold)
                    work_res_component = component
        else:
            binary = work_res_gray > threshold
            seed_x, seed_y = seed
            if not (0 <= seed_x < width and 0 <= seed_y < height):
                raise ThresholdResolutionError("Work-resolution seed lies outside the image")
            if not bool(binary[seed_y, seed_x]):
                raise ThresholdResolutionError(
                    f"Work-resolution tracking seed is not light at T={threshold}"
                )

            flooded = np.where(binary, 255, 0).astype(np.uint8)
            cv2.floodFill(flooded, None, seed, 128, flags=8)
            component = flooded == 128
            touches_border = bool(
                np.any(component[0])
                or np.any(component[-1])
                or np.any(component[:, 0])
                or np.any(component[:, -1])
            )
            if touches_border:
                break

            work_res_T = int(threshold)
            work_res_component = component

    if seed is None or work_res_T is None or work_res_component is None:
        raise ThresholdResolutionError(
            "No 5x5-supported enclosed bright component exists through T=0"
        )

    return int(work_res_T), np.asarray(work_res_component, dtype=bool)


def dilate_component_mask(component_mask: np.ndarray, margin: float) -> np.ndarray:
    """Return a full-size rounded Euclidean dilation of one component mask."""
    component = np.asarray(component_mask, dtype=bool)
    if component.ndim != 2 or not np.any(component):
        raise ThresholdResolutionError("Cannot dilate empty solar component")

    ys, xs = np.nonzero(component)
    height, width = component.shape
    padding = max(0, int(math.ceil(float(margin))))
    x0 = max(0, int(xs.min()) - padding - 2)
    x1 = min(width, int(xs.max()) + padding + 3)
    y0 = max(0, int(ys.min()) - padding - 2)
    y1 = min(height, int(ys.max()) + padding + 3)

    component_crop = component[y0:y1, x0:x1]
    outside = np.where(component_crop, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    allowed = distance <= float(margin)

    full_mask = np.zeros(component.shape, dtype=bool)
    full_mask[y0:y1, x0:x1] = allowed
    return full_mask


def find_lowest_full_res_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Find the lowest enclosed full-res T by searching only the monotonic direction."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    guard = np.asarray(full_res_guard_mask, dtype=bool)
    if guard.shape != full_res_gray.shape:
        raise ValueError("full-resolution guard and grayscale image must have identical shapes")
    if not np.any(guard):
        raise ThresholdResolutionError("Full-resolution Auto-T guard is empty")

    seed_x, seed_y = map(int, full_res_seed)
    height, width = full_res_gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the image")
    if not guard[seed_y, seed_x]:
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the Auto-T guard")

    guard_u8 = np.where(guard, 255, 0).astype(np.uint8)
    boundary_kernel = generate_kernel(3, round_kernel=False)
    eroded_guard = cv2.erode(guard_u8, boundary_kernel, iterations=1) != 0
    full_res_guard_boundary = guard & ~eroded_guard
    full_res_guard_boundary[0, :] |= guard[0, :]
    full_res_guard_boundary[-1, :] |= guard[-1, :]
    full_res_guard_boundary[:, 0] |= guard[:, 0]
    full_res_guard_boundary[:, -1] |= guard[:, -1]

    start_T = int(start_T)
    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)
    cv2.bitwise_and(binary, guard_u8, dst=binary)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed is not light at start T={start_T}"
        )
    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    component = binary == 128
    enclosed = not bool(np.any(component & full_res_guard_boundary))

    if enclosed:
        best_T = start_T
        best_component = component
        for threshold in range(start_T - 1, -1, -1):
            binary = cv2.compare(full_res_gray, int(threshold), cv2.CMP_GT)
            cv2.bitwise_and(binary, guard_u8, dst=binary)
            cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
            component = binary == 128
            if np.any(component & full_res_guard_boundary):
                break
            best_T = int(threshold)
            best_component = component
        return best_T, best_component

    for threshold in range(start_T + 1, 256):
        binary = cv2.compare(full_res_gray, int(threshold), cv2.CMP_GT)
        cv2.bitwise_and(binary, guard_u8, dst=binary)
        if binary[seed_y, seed_x] == 0:
            break
        cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
        component = binary == 128
        if not np.any(component & full_res_guard_boundary):
            return int(threshold), component

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became enclosed by the 10% guard"
    )


def find_auto_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int:
    """Determine Auto T and cache only search state; ``resolve_threshold`` owns SolarData."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")

    # Auto-T limits the longest work-resolution dimension to 1200 pixels.
    # Compute the complete target raster size here so resize_img() receives explicit
    # dimensions and the exact work-res geometry is available for later mapping.
    full_res_height, full_res_width = full_res_gray.shape
    if max(full_res_height, full_res_width) > WORK_MAX_DIM:
        work_res_scale = WORK_MAX_DIM / float(max(full_res_height, full_res_width))
        work_res_size = (
            max(1, round(full_res_width * work_res_scale)),
            max(1, round(full_res_height * work_res_scale)),
        )
    else:
        work_res_size = (full_res_width, full_res_height)
    work_res_gray = resize_img(full_res_gray, work_res_size)

    histogram_start_T = find_histogram_start_threshold(work_res_gray)
    work_res_seed_kernel = generate_kernel(TRACKING_SEED_KERNEL_SIZE, round_kernel=False)

    try:
        work_res_T, work_res_component = find_work_res_solar_component(
            work_res_gray,
            histogram_start_T,
            work_res_seed_kernel,
        )

        full_res_search_mask = resize_img(
            work_res_component,
            (full_res_width, full_res_height),
        )

        mapped_kernel_size = (
            TRACKING_SEED_KERNEL_SIZE
            * max(full_res_gray.shape)
            / float(max(work_res_gray.shape))
        )
        low = max(1, int(math.floor(mapped_kernel_size)))
        if low % 2 == 0:
            low -= 1
        high = low + 2
        full_res_kernel_size = (
            low
            if abs(mapped_kernel_size - low) <= abs(high - mapped_kernel_size)
            else high
        )
        full_res_seed_kernel = generate_kernel(full_res_kernel_size, round_kernel=False)

        full_res_seed = brightest_supported_component_point(
            full_res_gray,
            full_res_search_mask,
            full_res_seed_kernel,
        )
        if full_res_seed is None:
            raise ThresholdResolutionError(
                f"Transferred solar component has no {full_res_kernel_size}x"
                f"{full_res_kernel_size}-supported full-resolution seed"
            )

        image_scale = math.sqrt(float(full_res_width) * float(full_res_height))
        full_res_guard_mask = dilate_component_mask(
            full_res_search_mask,
            AUTO_T_GUARD_DILATION_FRACTION * image_scale,
        )
        full_res_T, full_res_component = find_lowest_full_res_threshold(
            full_res_gray,
            work_res_T,
            full_res_seed,
            full_res_guard_mask,
        )

        topology_selection = optimize_separated_threshold(
            full_res_gray,
            full_res_T,
            full_res_seed,
            full_res_component,
        )
        result = AutoThresholdResult(
            threshold=topology_selection.threshold,
            histogram_start_threshold=histogram_start_T,
            work_res_threshold=work_res_T,
            full_res_seed_point=full_res_seed,
            resolved=True,
        )
    except ThresholdResolutionError as exc:
        result = AutoThresholdResult(
            threshold=None,
            histogram_start_threshold=histogram_start_T,
            work_res_threshold=None,
            full_res_seed_point=None,
            resolved=False,
            reason=str(exc),
        )
        image_state["auto_threshold_result"] = result
        raise

    image_state["auto_threshold_result"] = result
    return int(result.threshold)


# ---------------------------------------------------------------------------
# Cleanup morphology candidates
# ---------------------------------------------------------------------------
CLEANUP_KERNEL_SIZES = (3, 5, 7)
CLEANUP_CANDIDATE_ORDER = ("raw", "D3", "D5", "D7", "P35", "P357")


def generate_kernel(size: int, round_kernel: bool = False) -> np.ndarray:
    """Return a centered positive odd square or discrete Euclidean-disk kernel."""
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    if not round_kernel:
        return np.ones((size, size), dtype=np.uint8)
    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)


def open_close_component(component: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply exactly one binary OPEN followed by one CLOSE with ``kernel``."""
    component = np.asarray(component, dtype=bool)
    kernel = np.asarray(kernel, dtype=np.uint8)
    if component.ndim != 2 or not np.any(component):
        raise ValueError("component mask must be a non-empty two-dimensional mask")
    if kernel.ndim != 2 or kernel.size == 0 or not np.any(kernel):
        raise ValueError("cleanup kernel must be a non-empty two-dimensional mask")
    u8 = np.where(component, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed != 0


def cleanup_morphology_candidates(component: np.ndarray) -> dict[str, np.ndarray]:
    """Build raw, direct 3/5/7, and progressive 3->5 / 3->5->7 candidates.

    Cleanup geometry is fixed to Euclidean disks. Progressive candidates use the
    output of the preceding OPEN->CLOSE step rather than reapplying each kernel to
    the raw component. No candidate selection or weighting is performed here.
    """
    raw = np.asarray(component, dtype=bool)
    if raw.ndim != 2 or not np.any(raw):
        raise ValueError("component mask must be a non-empty two-dimensional mask")

    # Generate each reused morphology kernel once before applying any candidate path.
    k3 = generate_kernel(3, round_kernel=True)
    k5 = generate_kernel(5, round_kernel=True)
    k7 = generate_kernel(7, round_kernel=True)

    d3 = open_close_component(raw, k3)
    d5 = open_close_component(raw, k5)
    d7 = open_close_component(raw, k7)
    p35 = open_close_component(d3, k5)
    p357 = open_close_component(p35, k7)
    return {
        "raw": raw.copy(),
        "D3": d3,
        "D5": d5,
        "D7": d7,
        "P35": p35,
        "P357": p357,
    }


def seed_connected_cleanup_candidates(
    component: np.ndarray,
    seed_point: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Return only cleanup candidates whose seed survives, retaining its 8-component.

    OPEN can sever a thin bridge while leaving detached bright material elsewhere.
    Such material is not part of the authoritative solar component. Each surviving
    candidate is therefore reduced to the 8-connected component containing the
    authoritative seed. Morphology candidates that remove the seed are rejected.
    """
    raw = np.asarray(component, dtype=bool)
    if raw.ndim != 2 or not np.any(raw):
        raise ThresholdResolutionError("Raw cleanup component is empty")
    sx, sy = map(int, seed_point)
    height, width = raw.shape
    if not (0 <= sx < width and 0 <= sy < height):
        raise ThresholdResolutionError("Cleanup seed lies outside the component raster")
    if not raw[sy, sx]:
        raise ThresholdResolutionError("Cleanup seed lies outside the raw component")

    candidates = cleanup_morphology_candidates(raw)
    valid: dict[str, np.ndarray] = {}
    for name in CLEANUP_CANDIDATE_ORDER:
        candidate = candidates[name]
        if not candidate[sy, sx]:
            continue
        flood = np.where(candidate, 255, 0).astype(np.uint8)
        cv2.floodFill(flood, None, (sx, sy), 128, flags=8)
        connected = flood == 128
        if np.any(connected):
            valid[name] = connected

    if "raw" not in valid:
        raise ThresholdResolutionError("Raw cleanup component lost its authoritative seed")
    return valid


def evaluate_cleanup_candidates(
    component: np.ndarray,
    seed_point: tuple[int, int],
    threshold: int,
) -> tuple[CleanupCandidateEvaluation, ...]:
    """Measure every valid cleanup candidate against the same raw component.

    This stage intentionally computes no aggregate score and chooses no winner.
    ``internal_dark_cleanup`` is the agreed absolute reduction in internal-dark
    fraction; all other terms retain the existing dimensionless normalization.
    """
    candidates = seed_connected_cleanup_candidates(component, seed_point)
    raw_topology = _component_descriptor(candidates["raw"], int(threshold))
    base_area = max(float(raw_topology.area), 1.0)
    base_contour = max(float(raw_topology.contour_n), 1.0)
    base_roughness = max(float(raw_topology.roughness), 1e-12)
    base_solidity = max(float(raw_topology.solidity), 1e-12)
    solidity_headroom = max(1.0 - float(raw_topology.solidity), 1e-12)

    rows: list[CleanupCandidateEvaluation] = []
    for name in CLEANUP_CANDIDATE_ORDER:
        candidate = candidates.get(name)
        if candidate is None:
            continue
        topology = raw_topology if name == "raw" else _component_descriptor(candidate, int(threshold))
        contour_cleanup = max(0.0, 1.0 - float(topology.contour_n) / base_contour)
        roughness_cleanup = max(0.0, 1.0 - float(topology.roughness) / base_roughness)
        solidity_gain = max(
            0.0,
            (float(topology.solidity) - float(raw_topology.solidity)) / solidity_headroom,
        )
        internal_dark_cleanup = max(
            0.0,
            float(raw_topology.internal_dark_fraction) - float(topology.internal_dark_fraction),
        )
        area_loss = max(0.0, 1.0 - float(topology.area) / base_area)
        solidity_loss = max(
            0.0,
            (float(raw_topology.solidity) - float(topology.solidity)) / base_solidity,
        )
        rows.append(
            CleanupCandidateEvaluation(
                name=name,
                mask=candidate,
                topology=topology,
                metrics=CleanupMetrics(
                    contour_cleanup=float(contour_cleanup),
                    roughness_cleanup=float(roughness_cleanup),
                    solidity_gain=float(solidity_gain),
                    internal_dark_cleanup=float(internal_dark_cleanup),
                    area_loss=float(area_loss),
                    solidity_loss=float(solidity_loss),
                ),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Final-T full-resolution solar resolution and persistence
# ---------------------------------------------------------------------------
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


@dataclass(frozen=True)
class SolarData:
    """Refined solar geometry established at exactly one full-resolution threshold T."""

    threshold: int
    seed_point: tuple[int, int]
    component_mask: bytes
    roi_6_5_mask: bytes
    guard_19_5_mask: bytes
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


def resolve_threshold(
    full_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Resolve one already-selected T and atomically publish its authoritative SolarData.

    Auto, manual, and restored thresholds enter this exact same full-resolution
    path. The authoritative seed is selected from the unrefined current-T component
    using ``brightest_supported_component_point`` and must survive refinement.
    """
    threshold = int(threshold)
    if full_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")

    existing = image_state.get("solar_data")
    if isinstance(existing, SolarData) and existing.threshold == threshold:
        refined_component = decompress_full_mask(existing.component_mask, full_gray.shape)
        seed_x, seed_y = map(int, existing.seed_point)
        height, width = full_gray.shape
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError("Stored SolarData seed lies outside the image")
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Stored SolarData seed is outside its refined solar component"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                f"Stored SolarData seed is not light at T={threshold}"
            )
        return refined_component

    # Final-T resolution is deliberately full resolution for now. Auto-specific
    # search/tracking state is not used here, so equal T values resolve identically
    # whether they came from Auto, manual input, or restored image settings.
    raw_component = largest_enclosed_bright_component(full_gray > threshold)
    if raw_component is None:
        raise ThresholdResolutionError(
            f"No enclosed solar component exists at selected T={threshold}"
        )

    full_res_max = max(full_gray.shape)
    work_res_max = min(full_res_max, WORK_MAX_DIM)
    mapped_kernel_size = TRACKING_SEED_KERNEL_SIZE * full_res_max / float(work_res_max)
    low = max(1, int(math.floor(mapped_kernel_size)))
    if low % 2 == 0:
        low -= 1
    high = low + 2
    full_res_kernel_size = (
        low if abs(mapped_kernel_size - low) <= abs(high - mapped_kernel_size) else high
    )
    full_res_seed_kernel = generate_kernel(full_res_kernel_size, round_kernel=False)
    full_res_seed = brightest_supported_component_point(
        full_gray,
        raw_component,
        full_res_seed_kernel,
    )
    if full_res_seed is None:
        raise ThresholdResolutionError(
            f"Solar component has no {full_res_kernel_size}x{full_res_kernel_size}-"
            "supported authoritative seed"
        )
    seed_x, seed_y = full_res_seed

    if not raw_component[seed_y, seed_x]:
        raise ThresholdResolutionError(
            "Authoritative solar seed lies outside the unrefined solar component"
        )
    if int(full_gray[seed_y, seed_x]) <= threshold:
        raise ThresholdResolutionError(
            f"Authoritative solar seed is not light at T={threshold}"
        )

    refined_component = refine_solar_component_mask(raw_component)
    if not np.any(refined_component):
        raise ThresholdResolutionError(
            f"Solar component was removed by refinement at T={threshold}"
        )
    if not refined_component[seed_y, seed_x]:
        raise ThresholdResolutionError(
            "Authoritative solar seed did not survive 7x7 OPEN/CLOSE refinement"
        )

    height, width = full_gray.shape
    image_scale = math.sqrt(float(width) * float(height))
    roi_6_5 = dilate_component_mask(
        refined_component,
        ROI_DILATION_FRACTION * image_scale,
    )
    guard_19_5 = dilate_component_mask(
        refined_component,
        GUARD_DILATION_FRACTION * image_scale,
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

    # No partially constructed SolarData is published if any resolution step fails.
    image_state["solar_data"] = solar_data
    return refined_component


class DetectorApp:
    """Own GUI state, per-image settings, cached auto thresholds, and previews.

    Public methods represent application actions. Underscore-prefixed methods are
    Tk callback or rendering internals. Threshold acquisition and final-T resolution
    both write their agreed derived objects into the current per-image state; this
    class coordinates those operations with the processing settings shown by the GUI.
    """

    def __init__(self, root: tk.Tk, image_paths: list[str], args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path: str | None = None
        self.color_image = None
        self.gray_image = None

        # Per-image state keeps settings, the cached automatic threshold result,
        # and post-threshold SolarData when solar geometry has been built.
        # No ImageState wrapper class is needed: the outer dictionary directly
        # expresses the image-to-state hierarchy.
        self.image_state: dict[str, dict[str, object]] = {}

        self.canvas_redraw_job = None

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

        # Ordinary controls use these values as sparse baselines. Threshold is
        # different: once initialized, its exact current integer is always stored.
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
            if hasattr(self, "threshold_canvas"):
                self.render_canvas_content(self.threshold_canvas, transparent_bgra())
            if hasattr(self, "color_canvas"):
                self.render_canvas_content(self.color_canvas, transparent_bgra())
            self.status.set(f"Could not load image: {path}")
            self.update_navigation_state()
            return

        self.color_image = opaque_bgra(image)
        self.gray_image = to_gray(image)

        if path in self.image_state:
            state = self.image_state[path]
            state.setdefault("auto_threshold_result", None)
            state.setdefault("solar_data", None)
        else:
            state = {
                "settings": ImageSettings(),
                "auto_threshold_result": None,
                "solar_data": None,
            }
            self.image_state[path] = state

        settings = state["settings"]
        for setting_name, variable in self.setting_variables.items():
            if setting_name == "threshold":
                continue
            baseline = getattr(self.default_settings, setting_name)
            override = getattr(settings, setting_name)
            variable.set(baseline if override is None else override)

        self._update_center_preview_label()
        if hasattr(self, "color_canvas"):
            self.render_canvas_content(self.color_canvas, self.color_image)

        if settings.threshold is None:
            # None has one meaning only: this image has never had T initialized.
            try:
                threshold = find_auto_threshold(self.gray_image, state)
            except ThresholdResolutionError as exc:
                self.status.set(f"Automatic threshold could not be resolved ({exc}).")
            else:
                self.commit_setting_change("threshold", threshold)
        else:
            self.threshold.set(int(settings.threshold))
            self.refresh_preview(changed_setting="image load")

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
            command=lambda: self.commit_setting_change("morphology", self.morphology.get()),
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=lambda: self.commit_setting_change("outer_limb_assistance", self.outer_limb_assistance.get()),
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=lambda: self.commit_setting_change("use_horizon", self.use_horizon.get()),
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
            self.commit_setting_change(event.widget._setting_name, event.widget.get())

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
            self.commit_setting_change(widget._setting_name, widget.get())

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
        self.threshold_canvas.bind("<Configure>", self._schedule_canvas_redraw)
        self.color_canvas.bind("<Configure>", self._schedule_canvas_redraw)

    # ------------------------------------------------------------------
    # Application actions and threshold preview
    # ------------------------------------------------------------------
    def save_centered_images(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the threshold-finder stage."
        )

    def auto_select_threshold(self):
        """Restore/recompute image-only Auto T, then commit it like any other T change."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        result = state.get("auto_threshold_result")
        if (
            isinstance(result, AutoThresholdResult)
            and result.resolved
            and result.threshold is not None
        ):
            selected_threshold = int(result.threshold)
        else:
            try:
                selected_threshold = find_auto_threshold(self.gray_image, state)
            except ThresholdResolutionError as exc:
                self.status.set(f"Automatic threshold could not be resolved ({exc}).")
                return
            result = state["auto_threshold_result"]

        self.commit_setting_change("threshold", selected_threshold)

        solar_data = state.get("solar_data")
        if isinstance(solar_data, SolarData) and solar_data.threshold == selected_threshold:
            self.status.set(
                "Automatic grayscale threshold selected: "
                f"T={selected_threshold} (work-res T={result.work_res_threshold}, "
                f"histogram start={result.histogram_start_threshold}); refined component displayed."
            )


    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the threshold-finder stage."
        )




    def commit_setting_change(self, setting_name, value):
        """Synchronize one completed setting change, persist it, then refresh once."""
        variable = self.setting_variables[setting_name]
        variable.set(value)
        value = variable.get()

        if self.current_path is None or self.current_path not in self.image_state:
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]

        if setting_name == "threshold":
            # Threshold is never sparse once initialized. None means never initialized.
            threshold = int(value)
            settings.threshold = threshold
            existing = state.get("solar_data")
            if isinstance(existing, SolarData) and existing.threshold != threshold:
                state["solar_data"] = None
        else:
            baseline = getattr(self.default_settings, setting_name)
            setattr(settings, setting_name, None if value == baseline else value)

        if self.gray_image is not None:
            self.refresh_preview(changed_setting=setting_name)


    def _selected_center_target_name(self):
        """Return the user-facing name of the selected centering target."""
        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"

    def _handle_center_target_change(self):
        self._update_center_preview_label()
        self.commit_setting_change("center_target", self.center_target.get())
        self.status.set(
            f"Centering target set to {self._selected_center_target_name()}. "
            "Actual centering will be implemented with ellipse detection."
        )


    def _update_center_preview_label(self):
        target = self._selected_center_target_name()
        self.center_preview_label.set(f"Full-color image — center on {target}")

    def refresh_preview(self, changed_setting: str | None = None, full_resolution: bool = False):
        """Display raw current-T B/W, resolve/persist SolarData, then display refinement.

        ``full_resolution`` is intentionally a placeholder today: both modes use the
        same authoritative full-resolution grayscale and the same resolver.
        """
        if self.gray_image is None or self.current_path is None:
            action = "Apply Full Resolution" if full_resolution else "Refresh Preview"
            self.status.set(f"{action}: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        if settings.threshold is None:
            self.status.set("Threshold is not initialized for the current image.")
            return

        threshold = int(settings.threshold)
        existing = state.get("solar_data")
        reused = isinstance(existing, SolarData) and existing.threshold == threshold
        if isinstance(existing, SolarData) and existing.threshold != threshold:
            state["solar_data"] = None

        raw_component = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.render_canvas_content(self.threshold_canvas, raw_component)

        try:
            refined_component = resolve_threshold(
                self.gray_image,
                threshold,
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure threshold remains displayed at T={threshold}; SolarData could not be "
                f"established ({exc})."
            )
            return

        if hasattr(self, "threshold_canvas"):
            self.render_canvas_content(self.threshold_canvas, refined_component)

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


    def _schedule_canvas_redraw(self, _event=None):
        if self.canvas_redraw_job is not None:
            self.root.after_cancel(self.canvas_redraw_job)
        self.canvas_redraw_job = self.root.after(
            CANVAS_REDRAW_DELAY_MS, self._redraw_cached_canvases
        )

    def _redraw_cached_canvases(self):
        """Refit each canvas's retained unscaled raster after a canvas resize."""
        self.canvas_redraw_job = None
        for canvas_name in ("threshold_canvas", "color_canvas"):
            if not hasattr(self, canvas_name):
                continue
            canvas = getattr(self, canvas_name)
            unscaled_raster = getattr(canvas, "_unscaled_render_raster", None)
            if unscaled_raster is not None:
                self.render_canvas_content(canvas, unscaled_raster)


    def render_canvas_content(self, canvas, content):
        """Render supplied content, retain its unscaled raster, flush Tk repaint work."""
        if content is None:
            unscaled_render_raster = transparent_bgra()
        else:
            unscaled_render_raster = np.asarray(content)
            if unscaled_render_raster.dtype == bool:
                unscaled_render_raster = np.where(
                    unscaled_render_raster, 255, 0
                ).astype(np.uint8)
            elif unscaled_render_raster.dtype != np.uint8:
                unscaled_render_raster = np.clip(
                    unscaled_render_raster, 0, 255
                ).astype(np.uint8)

            if unscaled_render_raster.ndim == 2:
                unscaled_render_raster = cv2.cvtColor(
                    unscaled_render_raster, cv2.COLOR_GRAY2BGRA
                )
                unscaled_render_raster[:, :, 3] = 255
            elif (
                unscaled_render_raster.ndim == 3
                and unscaled_render_raster.shape[2] == 3
            ):
                unscaled_render_raster = cv2.cvtColor(
                    unscaled_render_raster, cv2.COLOR_BGR2BGRA
                )
                unscaled_render_raster[:, :, 3] = 255
            elif (
                unscaled_render_raster.ndim != 3
                or unscaled_render_raster.shape[2] != 4
            ):
                raise ValueError(
                    "canvas content must be a 2D mask or 3/4-channel image"
                )

        # Canvas-owned display cache. It retains exactly the normalized, unscaled
        # raster most recently supplied to this canvas, regardless of which
        # processing/display path produced it. Resize redraw uses this copy only;
        # downstream image processing never consumes it.
        canvas._unscaled_render_raster = unscaled_render_raster.copy()

        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        raster_height, raster_width = unscaled_render_raster.shape[:2]
        scale = max(
            min(canvas_width / raster_width, canvas_height / raster_height), 1e-6
        )
        fitted_size = (
            max(1, round(raster_width * scale)),
            max(1, round(raster_height * scale)),
        )
        scaled_raster = resize_img(unscaled_render_raster, fitted_size)
        ok, encoded_png = cv2.imencode(".png", scaled_raster)
        if not ok:
            raise ValueError("could not encode canvas content")

        tk_photo = tk.PhotoImage(
            data=base64.b64encode(encoded_png).decode("ascii"),
            format="png",
        )
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2 + 1,
            canvas_height // 2 + 1,
            image=tk_photo,
            anchor="center",
        )
        # Tk does not retain the Python PhotoImage object. Keep it alive for as
        # long as the canvas is displaying it.
        canvas._tk_photo_image = tk_photo

        # The raw threshold stage must become paintable before sequential final-T
        # processing continues; no arbitrary timer or nested root.update() is used.
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
