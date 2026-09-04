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
authoritative seed from the unrefined component, applies a 7x7 Euclidean OPEN/CLOSE,
validates that the seed survives, constructs/stores SolarData from that same refined
mask, and returns the mask for final display. Auto, manual, and restored thresholds
therefore share exactly the same final-T path. ``full_resolution`` remains a future
placeholder; both preview modes use full resolution today.

The threshold algorithm uses authoritative 8-bit grayscale with fixed semantics
``dark = gray <= T`` and ``light = gray > T``. It derives a <=1200-pixel working
raster, starts from the left valley of the rightmost locally smoothed histogram mode,
establishes one 5x5-supported work-resolution solar seed, and tracks that same
8-connected component downward. The tracked work-resolution component is resized to
full resolution only to delimit the full-resolution seed search and the fixed 10% L2-distance
guard. Auto-T then starts from the work-resolution T and searches only in the
monotonic direction needed to find the lowest full-resolution threshold whose
D7-cleaned seeded component stays inside that fixed guard. If no supported work-resolution seed can be
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


# ---------------------------------------------------------------------------
# Generic image and kernel utilities
# ---------------------------------------------------------------------------
def transparent_bgra(width: int = 1, height: int = 1) -> np.ndarray:
    """Return a BGRA frame whose pixels are fully transparent (alpha = 0)."""
    if width <= 0 or height <= 0:
        raise ValueError("transparent raster dimensions must be positive")
    return np.zeros((height, width, 4), dtype=np.uint8)


def opaque_bgra(bgr: np.ndarray) -> np.ndarray:
    """Convert a normal OpenCV BGR image to BGRA with fully opaque image pixels."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)


def generate_kernel(
    size: tuple[int, int],
    round_kernel: bool = False,
) -> np.ndarray:
    """Return a positive-odd rectangle or centered discrete L2 ellipse."""
    width, height = size
    if width <= 0 or height <= 0 or width % 2 == 0 or height % 2 == 0:
        raise ValueError("kernel width and height must be positive odd integers")

    if not round_kernel:
        return np.ones((height, width), dtype=np.uint8)

    # A one-pixel axis is the exact degenerate ellipse: a straight filled line.
    if width == 1 or height == 1:
        return np.ones((height, width), dtype=np.uint8)

    x_radius = width // 2
    y_radius = height // 2
    yy, xx = np.ogrid[-y_radius : y_radius + 1, -x_radius : x_radius + 1]
    ellipse = (xx / x_radius) ** 2 + (yy / y_radius) ** 2 <= 1.0
    return ellipse.astype(np.uint8)



def resize_img(
    img: np.ndarray,
    size: tuple[int, int],
    mask: bool = False,
) -> np.ndarray:
    """Resize to an explicit ``(width, height)``; masks always use exact nearest."""
    original_dtype = img.dtype
    original_height, original_width = img.shape[:2]
    width, height = size

    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    if (width, height) == (original_width, original_height):
        return img.copy()

    if mask:
        # OpenCV cannot resize bool directly; preserve mask membership with exact nearest.
        resize_source = img.astype(np.uint8) if img.dtype == bool else img
        interpolation = cv2.INTER_NEAREST_EXACT
    elif width < original_width or height < original_height:
        resize_source = img
        interpolation = cv2.INTER_AREA
    else:
        resize_source = img
        interpolation = cv2.INTER_LANCZOS4

    resized = cv2.resize(
        resize_source,
        (width, height),
        interpolation=interpolation,
    )
    return resized.astype(original_dtype, copy=False)


def normalize_master_bgra16(image: np.ndarray) -> np.ndarray:
    """Normalize an unchanged OpenCV load to lossless contiguous uint16 BGRA."""
    source = np.asarray(image)
    if source.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"master image dtype must be uint8 or uint16, got {source.dtype}")

    if source.ndim == 2:
        bgra = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    elif source.ndim == 3 and source.shape[2] == 3:
        bgra = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    elif source.ndim == 3 and source.shape[2] == 4:
        bgra = source
    else:
        raise ValueError(f"unsupported master image shape: {source.shape}")

    # Expanding uint8 by exactly 257 preserves every source code value in uint16.
    if bgra.dtype == np.uint8:
        bgra = bgra.astype(np.uint16) * 257
    return np.ascontiguousarray(bgra, dtype=np.uint16)


def master_bgra16_to_gray8(master_bgra16: np.ndarray) -> np.ndarray:
    """Derive authoritative uint8 gray directly from the lossless uint16 BGRA master."""
    master = np.asarray(master_bgra16)
    if master.dtype != np.uint16 or master.ndim != 3 or master.shape[2] != 4:
        raise ValueError("master image must be uint16 BGRA")

    gray16 = cv2.cvtColor(master, cv2.COLOR_BGRA2GRAY)
    # Fixed full-range mapping keeps T=0..255 comparable across all images.
    return ((gray16.astype(np.uint32) + 128) // 257).astype(np.uint8)


def master_bgra16_to_display_bgra8(master_bgra16: np.ndarray) -> np.ndarray:
    """Derive a fixed-range uint8 BGRA display raster without modifying the master."""
    master = np.asarray(master_bgra16)
    if master.dtype != np.uint16 or master.ndim != 3 or master.shape[2] != 4:
        raise ValueError("master image must be uint16 BGRA")
    return ((master.astype(np.uint32) + 128) // 257).astype(np.uint8)


def compress_master_bgra16(master_bgra16: np.ndarray) -> bytes:
    """Compress one contiguous uint16 BGRA master with fast lossless zlib level 1."""
    master = np.asarray(master_bgra16)
    if master.dtype != np.uint16 or master.ndim != 3 or master.shape[2] != 4:
        raise ValueError("master image must be uint16 BGRA")
    return zlib.compress(np.ascontiguousarray(master).tobytes(), level=1)


def decompress_master_bgra16(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    """Restore a compressed uint16 BGRA master as a read-only ndarray view."""
    if len(shape) != 3 or shape[2] != 4 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("master shape must be positive (height, width, 4)")
    raw = zlib.decompress(payload)
    expected_bytes = shape[0] * shape[1] * shape[2] * np.dtype(np.uint16).itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"compressed master has {len(raw)} bytes; expected {expected_bytes}"
        )
    return np.frombuffer(raw, dtype=np.uint16).reshape(shape)

def nearest_positive_odd(value: float) -> int:
    """Return the nearest positive odd integer; exact ties choose the lower odd."""
    if value <= 0:
        raise ValueError("value must be positive")
    return 2 * math.ceil(value / 2) - 1


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
# Automatic threshold: coarse separation and deterministic refinement
# ---------------------------------------------------------------------------
# The coarse search establishes one fixed full-resolution seed/guard pair and the
# lowest separated T. Fine refinement may only raise that T while preserving the
# same component identity and fixed guard.
WORK_RES_MAX_DIM = 1200
PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)
AUTO_T_GUARD_DILATION_FRACTION = 0.10


class ThresholdResolutionError(RuntimeError):
    """Expected inability to establish/track a separated solar component."""


@dataclass(frozen=True)
class AutoThresholdResult:
    threshold: int | None
    histogram_start_threshold: int
    work_res_threshold: int | None
    full_res_seed_point: tuple[int, int] | None
    # Currently equivalent to ``threshold is not None``. Kept explicit for now in
    # case resolution-state semantics become independent later.
    resolved: bool
    cleaned_component_mask: bytes | None = None
    reason: str = ""


def find_histogram_start_threshold(work_res_gray: np.ndarray) -> int:
    """Return the left valley preceding the rightmost 3-bin-smoothed histogram mode."""
    histogram = np.bincount(work_res_gray.ravel(), minlength=256).astype(np.float64)
    signal = np.convolve(histogram, PEAK_KERNEL, mode="same")

    rightmost_peak = None
    for index in range(1, len(signal) - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            rightmost_peak = index
    if signal[-1] > signal[-2]:
        rightmost_peak = len(signal) - 1
    if rightmost_peak is None:
        rightmost_peak = int(np.argmax(signal))

    for index in range(rightmost_peak - 1, 0, -1):
        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:
            return index
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

    supported = cv2.erode(
        source,
        support_kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
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
        (binary != 0).astype(np.uint8),
        connectivity=8,
    )
    height, width = binary.shape
    best_label = None
    best_area = None
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        ):
            continue
        if best_label is None or area > best_area:
            best_area = area
            best_label = label
    if best_label is None:
        return None
    return labels == best_label


MAX_T_REFINEMENT_STEPS = 10

EDGE_RADIUS = 25
EDGE_RECOVERY_FRACTION = 0.25
EDGE_RECOVERY_PERSISTENCE = 4
EDGE_MIN_SAMPLE_STRIDE = 8
EDGE_TARGET_PROFILE_COUNT = 2000
EDGE_TANGENT_SPAN = 12

GUARD_BOUNDARY_KERNEL = generate_kernel((3, 3), round_kernel=False)
SEPARATION_KERNEL = generate_kernel((7, 7), round_kernel=True)
SOLAR_CLEANUP_KERNELS = (
    generate_kernel((3, 3), round_kernel=True),
    generate_kernel((5, 5), round_kernel=True),
    SEPARATION_KERNEL,
)


@dataclass(frozen=True)
class ThresholdMeasurement:
    """All retained measurements and score terms for one cleaned threshold candidate."""

    threshold: int
    component_area: int
    filled_area: int
    roughness: float
    solidity: float
    internal_dark_fraction: float
    edge_alignment: float
    edge_credible_fraction: float
    edge_transition_monotonicity: float
    q_roughness: float = 0.0
    q_holes: float = 0.0
    q_area: float = 0.0
    q_solidity: float = 0.0
    q_edge: float = 0.0
    score: float = 0.0


@dataclass(frozen=True)
class ThresholdRefinementResult:
    """Selected fine threshold plus the authoritative full-resolution cleaned mask."""

    threshold: int
    base_threshold: int
    refinement_steps: int
    score: float
    edge_reliability: float
    edge_weight: float
    raw_reference_threshold: int
    cleaned_component_mask: bytes
    trajectory: tuple[ThresholdMeasurement, ...]


def extract_component(
    binary_mask: np.ndarray,
    seed_point: tuple[int, int],
) -> np.ndarray | None:
    """Return the 8-connected component of ``binary_mask`` containing ``seed_point``."""
    source = np.asarray(binary_mask)
    if source.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")

    seed_x, seed_y = seed_point
    height, width = source.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("seed lies outside binary-mask raster")
    if source[seed_y, seed_x] == 0:
        return None

    flood = np.where(source != 0, 255, 0).astype(np.uint8)
    cv2.floodFill(flood, None, (seed_x, seed_y), 128, flags=8)
    component = flood == 128
    return component if np.any(component) else None


def find_work_res_solar_component(
    work_res_gray: np.ndarray,
    start_T: int,
    work_res_seed_kernel: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Establish one supported work-resolution seed, then track that identity downward."""
    seed: tuple[int, int] | None = None
    work_res_T: int | None = None
    work_res_component: np.ndarray | None = None

    for threshold in range(start_T, -1, -1):
        if seed is None:
            component = largest_enclosed_bright_component(work_res_gray > threshold)
            if component is not None:
                seed = brightest_supported_component_point(
                    work_res_gray,
                    component,
                    work_res_seed_kernel,
                )
                if seed is not None:
                    work_res_T = threshold
                    work_res_component = component
        else:
            component = extract_component(work_res_gray > threshold, seed)
            if component is None:
                break
            if (
                np.any(component[0])
                or np.any(component[-1])
                or np.any(component[:, 0])
                or np.any(component[:, -1])
            ):
                break
            work_res_T = threshold
            work_res_component = component

    if seed is None or work_res_T is None or work_res_component is None:
        kernel_height, kernel_width = work_res_seed_kernel.shape
        raise ThresholdResolutionError(
            f"No {kernel_width}x{kernel_height}-supported enclosed bright component "
            "exists through T=0"
        )
    return work_res_T, work_res_component


def dilate_component_mask(component_mask: np.ndarray, margin: float) -> np.ndarray:
    """Return every raster pixel within ``margin`` L2 pixels of the component."""
    component = np.asarray(component_mask, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if not np.any(component):
        raise ThresholdResolutionError("Cannot dilate empty solar component")
    if margin < 0:
        raise ValueError("dilation margin must be non-negative")

    # distanceTransform measures each non-component pixel's L2 distance to the
    # nearest zero pixel, so encode the component itself as zero and threshold the
    # resulting full-frame distance field at the requested dilation margin.
    outside = np.where(component, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    return distance <= margin


def clean_solar_component(
    binary_mask: np.ndarray,
    seed_point: tuple[int, int],
    guard_mask: np.ndarray,
) -> np.ndarray | None:
    """Clean the full threshold mask, apply the fixed guard, and return its solar component.

    Each Euclidean kernel performs exactly one OPEN followed by one CLOSE.  Morphology
    operates on the unguarded threshold mask.  The fixed guard is applied only after the
    complete 3/5/7 cleanup, then component identity is resolved once by extracting the
    8-connected region containing the authoritative seed.
    """
    source = np.asarray(binary_mask)
    guard = np.asarray(guard_mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")
    if guard.shape != source.shape or not np.any(guard):
        raise ValueError("guard must be a non-empty mask matching binary mask")
    if not np.any(source):
        return None

    cleaned = np.where(source != 0, 255, 0).astype(np.uint8)
    for kernel in SOLAR_CLEANUP_KERNELS:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned[~guard] = 0
    return extract_component(cleaned, seed_point)


def find_guard_boundary(guard: np.ndarray) -> np.ndarray:
    source = np.asarray(guard, dtype=bool)
    if source.ndim != 2 or not np.any(source):
        raise ValueError("guard must be a non-empty two-dimensional mask")
    u8 = np.where(source, 255, 0).astype(np.uint8)
    eroded = cv2.erode(
        u8,
        GUARD_BOUNDARY_KERNEL,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
    return source & ~eroded


def find_lowest_full_res_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> int:
    """Return the lowest T whose D7-cleaned solar component is separated by the fixed guard.

    The guard semantics intentionally match :func:`refine_threshold`: threshold first,
    morphology on the unguarded full-resolution mask, then fixed-guard clipping, then
    :func:`extract_component`, then the common guard-boundary separation test.
    """
    gray = np.asarray(full_res_gray)
    guard = np.asarray(full_res_guard_mask, dtype=bool)
    if gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if gray.dtype != np.uint8:
        raise ValueError("separation search requires authoritative uint8 grayscale")
    if not 0 <= start_T <= 255:
        raise ValueError("start threshold must be 0..255")
    if guard.shape != gray.shape:
        raise ValueError("full-resolution guard and grayscale image must have identical shapes")
    if not np.any(guard):
        raise ThresholdResolutionError("Full-resolution Auto-T guard is empty")

    seed_x, seed_y = full_res_seed
    height, width = gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the image")
    if not guard[seed_y, seed_x]:
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the Auto-T guard")

    boundary = find_guard_boundary(guard)
    separation_kernel = SEPARATION_KERNEL

    def component_at(threshold: int) -> np.ndarray | None:
        binary = cv2.compare(gray, threshold, cv2.CMP_GT)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, separation_kernel, iterations=1
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, separation_kernel, iterations=1
        )
        binary[~guard] = 0
        return extract_component(binary, full_res_seed)

    component = component_at(start_T)
    if component is None:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed does not survive D7 cleanup at start T={start_T}"
        )

    if not np.any(component & boundary):
        best_threshold = start_T
        for threshold in range(start_T - 1, -1, -1):
            component = component_at(threshold)
            if component is None or np.any(component & boundary):
                break
            best_threshold = threshold
        return best_threshold

    for threshold in range(start_T + 1, 256):
        component = component_at(threshold)
        if component is None:
            break
        if not np.any(component & boundary):
            return threshold

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became separated after D7 cleanup"
    )


def find_external_contour(component: np.ndarray) -> np.ndarray:
    """Return the longest external CHAIN_APPROX_NONE contour of a non-empty component."""
    source = np.asarray(component, dtype=bool)
    contours, _ = cv2.findContours(
        np.where(source, 255, 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ThresholdResolutionError("solar component has no external contour")
    return max(contours, key=cv2.contourArea)


def _lattice_boundary_points(contour: np.ndarray) -> int:
    points = contour[:, 0, :].astype(np.int64)
    following = np.roll(points, -1, axis=0)
    distance = np.abs(following - points)
    return int(np.gcd(distance[:, 0], distance[:, 1]).sum())


def measure_filled_area(contour: np.ndarray) -> int:
    """Return raster-equivalent area enclosed by the external lattice contour."""
    polygon_area = float(cv2.contourArea(contour))
    boundary_points = _lattice_boundary_points(contour)
    filled_area = int(round(polygon_area + 0.5 * boundary_points + 1.0))
    if filled_area <= 0:
        raise ThresholdResolutionError("filled external contour area is empty")
    return filled_area


def measure_roughness(contour: np.ndarray, filled_area: int) -> float:
    """Return perimeter normalized by the equal-area circle using filled external area."""
    if filled_area <= 0:
        raise ValueError("filled area must be positive")
    perimeter = float(cv2.arcLength(contour, True))
    return perimeter / (2.0 * math.sqrt(math.pi * filled_area))


def measure_solidity(contour: np.ndarray, filled_area: int) -> float:
    """Return filled external area divided by raster-filled convex-hull area."""
    if filled_area <= 0:
        raise ValueError("filled area must be positive")
    hull = cv2.convexHull(contour)
    x, y, width, height = cv2.boundingRect(hull)
    local_hull = hull.astype(np.int32, copy=True)
    local_hull[:, 0, 0] -= x
    local_hull[:, 0, 1] -= y
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(raster, local_hull, 1)
    filled_hull_area = int(np.count_nonzero(raster))
    if filled_hull_area <= 0:
        raise ThresholdResolutionError("filled convex hull is empty")
    return filled_area / filled_hull_area


def measure_internal_dark_fraction(component_area: int, filled_area: int) -> float:
    """Return the fraction of the filled external silhouette absent from the component."""
    if filled_area <= 0:
        raise ValueError("filled area must be positive")
    if component_area < 0 or component_area > filled_area:
        raise ValueError("component area must be between zero and filled area")
    return (filled_area - component_area) / filled_area


def _nearest_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = np.rint(xy[:, 0]).astype(np.int32)
    y = np.rint(xy[:, 1]).astype(np.int32)
    height, width = mask.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    output = np.zeros(len(x), dtype=bool)
    valid_indices = np.flatnonzero(valid)
    output[valid_indices] = mask[y[valid_indices], x[valid_indices]]
    return output


def _contour_normals(
    component: np.ndarray,
    contour: np.ndarray,
    sample_stride: int,
    tangent_span: int = EDGE_TANGENT_SPAN,
) -> tuple[np.ndarray, np.ndarray]:
    points = contour[:, 0, :].astype(np.float32)
    count = len(points)
    if count < 2 * tangent_span + 3:
        tangent_span = max(1, count // 8)

    indices = np.arange(0, count, max(1, sample_stride), dtype=np.int32)
    sampled = points[indices]
    previous = points[(indices - tangent_span) % count]
    following = points[(indices + tangent_span) % count]
    tangent = following - previous
    lengths = np.linalg.norm(tangent, axis=1)
    usable = lengths > 1e-6
    sampled = sampled[usable]
    tangent = tangent[usable] / lengths[usable, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)

    positive_inside = np.zeros(len(sampled), np.int16)
    negative_inside = np.zeros(len(sampled), np.int16)
    for distance in (1.0, 2.0, 3.0, 4.0, 5.0):
        positive_inside += _nearest_mask(component, sampled + normal * distance).astype(np.int16)
        negative_inside += _nearest_mask(component, sampled - normal * distance).astype(np.int16)

    positive_is_inward = positive_inside > negative_inside
    tied = positive_inside == negative_inside
    if np.any(tied):
        positive2 = _nearest_mask(component, sampled[tied] + normal[tied] * 2.0)
        negative2 = _nearest_mask(component, sampled[tied] - normal[tied] * 2.0)
        tie_choice = positive2 & ~negative2
        tie_resolved = positive2 ^ negative2
        tied_indices = np.flatnonzero(tied)
        positive_is_inward[tied_indices[tie_resolved]] = tie_choice[tie_resolved]
        keep = np.ones(len(sampled), dtype=bool)
        keep[tied_indices[~tie_resolved]] = False
        sampled = sampled[keep]
        normal = normal[keep]
        positive_is_inward = positive_is_inward[keep]

    outward = normal.copy()
    outward[positive_is_inward] *= -1.0
    return sampled, outward


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return math.nan, math.nan
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator <= 1e-12:
        return 0.0, y_mean
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denominator)
    return slope, y_mean - slope * x_mean


def _sample_edge_profiles(
    gray: np.ndarray,
    component: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(
        EDGE_MIN_SAMPLE_STRIDE,
        math.ceil(max(1, len(contour)) / EDGE_TARGET_PROFILE_COUNT),
    )
    points, outward = _contour_normals(component, contour, stride)
    if len(points) == 0:
        raise ThresholdResolutionError("no oriented contour samples for photometric edge")

    distances = np.arange(-EDGE_RADIUS, EDGE_RADIUS + 1, dtype=np.float32)
    xy = points[:, None, :] + outward[:, None, :] * distances[None, :, None]
    height, width = gray.shape
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    complete = (
        (x.min(axis=1) >= 0)
        & (x.max(axis=1) <= width - 1)
        & (y.min(axis=1) >= 0)
        & (y.max(axis=1) <= height - 1)
    )
    xy = xy[complete]
    if len(xy) == 0:
        raise ThresholdResolutionError("all photometric edge profiles leave the image")

    source = gray.astype(np.float32, copy=False)
    map_x = xy[:, :, 0].astype(np.float32)
    map_y = xy[:, :, 1].astype(np.float32)
    profile_chunks = []
    for start in range(0, len(map_x), 15000):
        stop = min(len(map_x), start + 15000)
        profile_chunks.append(
            cv2.remap(
                source,
                map_x[start:stop],
                map_y[start:stop],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    profiles = np.vstack(profile_chunks)
    profiles = cv2.GaussianBlur(profiles, (7, 1), sigmaX=1.25, sigmaY=0)
    return profiles, distances


def measure_edge_alignment(
    gray: np.ndarray,
    component: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, float, float]:
    """Measure edge quality plus credibility and transition-monotonicity confidence.

    Returns ``(alignment, credible_fraction, transition_monotonicity)``.  Alignment
    is 1 at zero median absolute outer-edge offset and falls linearly to 0 at
    ``EDGE_RADIUS`` pixels.  The endpoint is the 25%-of-peak derivative-recovery
    location with four-pixel persistence.
    """
    profiles, distances = _sample_edge_profiles(gray, component, contour)
    profile_count = len(profiles)
    drop = (profiles[:, :-2] - profiles[:, 2:]) * 0.5
    derivative_positions = distances[1:-1]
    peak_indices = np.argmax(drop, axis=1)
    rows = np.arange(profile_count)
    peak_drop = drop[rows, peak_indices]
    peak_position = distances[peak_indices + 1]

    inner = np.median(profiles[:, :6], axis=1)
    outer = np.median(profiles[:, -6:], axis=1)
    total_drop = inner - outer
    credible = (total_drop >= 1.0) & (peak_drop >= 0.12)
    credible_count = int(np.count_nonzero(credible))
    credible_fraction = credible_count / profile_count if profile_count else 0.0
    if credible_count == 0:
        return 0.0, float(credible_fraction), 0.0

    endpoints = np.full(profile_count, np.nan, dtype=np.float32)
    monotonicity = np.full(profile_count, np.nan, dtype=np.float32)

    for index in np.flatnonzero(credible):
        peak_derivative_index = int(peak_indices[index])
        recovery_limit = max(0.02, EDGE_RECOVERY_FRACTION * float(peak_drop[index]))
        for derivative_index in range(
            peak_derivative_index + 1,
            len(derivative_positions) - EDGE_RECOVERY_PERSISTENCE + 1,
        ):
            segment = drop[
                index,
                derivative_index : derivative_index + EDGE_RECOVERY_PERSISTENCE,
            ]
            if (
                len(segment) == EDGE_RECOVERY_PERSISTENCE
                and np.all(segment <= recovery_limit)
            ):
                endpoints[index] = derivative_positions[derivative_index]
                break

        peak_profile_index = peak_derivative_index + 1
        transition_low = max(0, peak_profile_index - 3)
        transition_high = min(len(distances), peak_profile_index + 4)
        transition_slope, transition_intercept = _linear_fit(
            distances[transition_low:transition_high],
            profiles[index, transition_low:transition_high],
        )
        outer_slope, outer_intercept = _linear_fit(
            distances[-7:], profiles[index, -7:]
        )
        inner_slope, inner_intercept = _linear_fit(
            distances[:7], profiles[index, :7]
        )

        tangent_outer = math.nan
        tangent_inner = math.nan
        if (
            math.isfinite(transition_slope)
            and transition_slope < -1e-3
            and math.isfinite(outer_slope)
            and abs(transition_slope - outer_slope) > 1e-4
        ):
            value = (outer_intercept - transition_intercept) / (
                transition_slope - outer_slope
            )
            if distances[0] - 5 <= value <= distances[-1] + 5:
                tangent_outer = value
        if (
            math.isfinite(transition_slope)
            and transition_slope < -1e-3
            and math.isfinite(inner_slope)
            and abs(transition_slope - inner_slope) > 1e-4
        ):
            value = (inner_intercept - transition_intercept) / (
                transition_slope - inner_slope
            )
            if distances[0] - 5 <= value <= distances[-1] + 5:
                tangent_inner = value

        x0 = tangent_inner if math.isfinite(tangent_inner) else peak_position[index] - 6
        x1 = tangent_outer if math.isfinite(tangent_outer) else peak_position[index] + 6
        if x1 > x0:
            transition = (derivative_positions >= x0) & (derivative_positions <= x1)
            if np.any(transition):
                monotonicity[index] = float(
                    np.mean(drop[index, transition] >= -0.03)
                )

    valid_endpoints = endpoints[np.isfinite(endpoints) & credible]
    valid_monotonicity = monotonicity[np.isfinite(monotonicity) & credible]
    median_abs_offset = (
        float(np.median(np.abs(valid_endpoints)))
        if len(valid_endpoints)
        else float(EDGE_RADIUS)
    )
    transition_monotonicity = (
        float(np.median(valid_monotonicity))
        if len(valid_monotonicity)
        else 0.0
    )
    alignment = 1.0 - min(median_abs_offset / EDGE_RADIUS, 1.0)
    return (
        float(alignment),
        float(credible_fraction),
        float(transition_monotonicity),
    )


def refine_threshold(
    full_res_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
    max_steps: int = MAX_T_REFINEMENT_STEPS,
) -> ThresholdRefinementResult:
    """Fine-tune one separated threshold using deterministic full-resolution cleanup."""
    gray = np.asarray(full_res_gray)
    guard = np.asarray(full_res_guard_mask, dtype=bool)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("threshold refinement requires authoritative uint8 grayscale")
    if guard.shape != gray.shape or not np.any(guard):
        raise ValueError("guard must be a non-empty mask matching grayscale")
    if not 0 <= base_threshold <= 255:
        raise ValueError("base threshold must be 0..255")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    seed_x, seed_y = seed_point
    height, width = gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("seed lies outside grayscale raster")
    if not guard[seed_y, seed_x]:
        raise ValueError("seed must lie inside the fixed guard")

    boundary = find_guard_boundary(guard)

    preliminary: list[ThresholdMeasurement] = []
    compressed_masks: dict[int, bytes] = {}
    raw_reference_threshold: int | None = None
    raw_reference_area: int | None = None
    raw_reference_roughness: float | None = None

    max_threshold = min(255, base_threshold + max_steps)
    for threshold in range(base_threshold, max_threshold + 1):
        threshold_mask = cv2.compare(gray, threshold, cv2.CMP_GT)

        cleaned = clean_solar_component(threshold_mask, seed_point, guard)
        if cleaned is not None and not np.any(cleaned & boundary):
            contour = find_external_contour(cleaned)
            filled_area = measure_filled_area(contour)
            roughness = measure_roughness(contour, filled_area)
            solidity = measure_solidity(contour, filled_area)
            component_area = int(np.count_nonzero(cleaned))
            internal_dark_fraction = measure_internal_dark_fraction(
                component_area, filled_area
            )
            edge_alignment, credible_fraction, transition_monotonicity = (
                measure_edge_alignment(gray, cleaned, contour)
            )

            preliminary.append(
                ThresholdMeasurement(
                    threshold=threshold,
                    component_area=component_area,
                    filled_area=filled_area,
                    roughness=roughness,
                    solidity=solidity,
                    internal_dark_fraction=internal_dark_fraction,
                    edge_alignment=edge_alignment,
                    edge_credible_fraction=credible_fraction,
                    edge_transition_monotonicity=transition_monotonicity,
                )
            )
            compressed_masks[threshold] = compress_full_mask(cleaned)

        # Raw geometry exists only to anchor the large/rough end of the score scale.
        # Find the first separated raw component once, measure area/roughness, then
        # never extract or measure another raw component in this refinement pass.
        if raw_reference_threshold is None:
            raw_mask = threshold_mask.copy()
            raw_mask[~guard] = 0
            raw_component = extract_component(raw_mask, seed_point)
            if raw_component is not None and not np.any(raw_component & boundary):
                raw_contour = find_external_contour(raw_component)
                raw_reference_area = measure_filled_area(raw_contour)
                raw_reference_roughness = measure_roughness(
                    raw_contour, raw_reference_area
                )
                raw_reference_threshold = threshold

    if not preliminary:
        raise ThresholdResolutionError(
            "no separated cleaned solar component exists in the refinement window"
        )
    if raw_reference_threshold is None:
        raise ThresholdResolutionError(
            "no separated raw reference exists in the refinement window"
        )
    assert raw_reference_area is not None
    assert raw_reference_roughness is not None

    max_roughness = max(
        raw_reference_roughness,
        *(measurement.roughness for measurement in preliminary),
    )
    max_area = max(
        raw_reference_area,
        *(measurement.filled_area for measurement in preliminary),
    )
    if not math.isfinite(max_roughness) or max_roughness <= 0.0 or max_area <= 0:
        raise ThresholdResolutionError("invalid within-image score scale")

    reliability_samples = [
        measurement.edge_credible_fraction
        * measurement.edge_transition_monotonicity
        for measurement in preliminary
    ]
    edge_reliability = float(np.median(reliability_samples))
    edge_reliability = (
        float(np.clip(edge_reliability, 0.0, 1.0))
        if math.isfinite(edge_reliability)
        else 0.0
    )
    edge_weight = edge_reliability**2

    scored: list[ThresholdMeasurement] = []
    for measurement in preliminary:
        q_roughness = 1.0 - measurement.roughness / max_roughness
        q_holes = 1.0 - measurement.internal_dark_fraction
        q_area = measurement.filled_area / max_area
        q_solidity = measurement.solidity
        q_edge = measurement.edge_alignment
        score = (
            q_roughness
            + q_holes
            + 0.5 * q_area
            + 0.5 * q_solidity
            + edge_weight * q_edge
        )
        scored.append(
            ThresholdMeasurement(
                threshold=measurement.threshold,
                component_area=measurement.component_area,
                filled_area=measurement.filled_area,
                roughness=measurement.roughness,
                solidity=measurement.solidity,
                internal_dark_fraction=measurement.internal_dark_fraction,
                edge_alignment=measurement.edge_alignment,
                edge_credible_fraction=measurement.edge_credible_fraction,
                edge_transition_monotonicity=measurement.edge_transition_monotonicity,
                q_roughness=q_roughness,
                q_holes=q_holes,
                q_area=q_area,
                q_solidity=q_solidity,
                q_edge=q_edge,
                score=score,
            )
        )

    best_score = max(measurement.score for measurement in scored)
    # Trajectory order is ascending T, so the first exact maximum is the lower-T
    # deterministic tie-break without introducing any tolerance rule.
    chosen = next(measurement for measurement in scored if measurement.score == best_score)

    return ThresholdRefinementResult(
        threshold=chosen.threshold,
        base_threshold=base_threshold,
        refinement_steps=chosen.threshold - base_threshold,
        score=chosen.score,
        edge_reliability=edge_reliability,
        edge_weight=edge_weight,
        raw_reference_threshold=raw_reference_threshold,
        cleaned_component_mask=compressed_masks[chosen.threshold],
        trajectory=tuple(scored),
    )

def find_auto_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int:
    """Determine automatic T, retain its seed, and cache the winning cleaned mask."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if full_res_gray.dtype != np.uint8:
        raise ValueError("automatic thresholding requires authoritative uint8 grayscale")

    full_res_height, full_res_width = full_res_gray.shape
    full_res_max_dim = max(full_res_height, full_res_width)
    if full_res_max_dim > WORK_RES_MAX_DIM:
        work_res_scale = WORK_RES_MAX_DIM / full_res_max_dim
        work_res_size = (
            round(full_res_width * work_res_scale),
            round(full_res_height * work_res_scale),
        )
    else:
        work_res_size = (full_res_width, full_res_height)

    work_res_gray = resize_img(full_res_gray, work_res_size)
    histogram_start_T = find_histogram_start_threshold(work_res_gray)
    work_res_seed_kernel = generate_kernel((5, 5), round_kernel=False)

    try:
        # Coarse search: establish and track one supported work-resolution identity.
        work_res_T, work_res_component = find_work_res_solar_component(
            work_res_gray,
            histogram_start_T,
            work_res_seed_kernel,
        )

        # Transfer the mature component only to delimit full-resolution seed/guard state.
        full_res_search_mask = resize_img(
            work_res_component,
            (full_res_width, full_res_height),
            mask=True,
        )

        work_res_support_size = max(work_res_seed_kernel.shape)
        mapped_kernel_size = (
            work_res_support_size
            * max(full_res_gray.shape)
            / max(work_res_gray.shape)
        )
        full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)
        full_res_seed_kernel = generate_kernel(
            (full_res_kernel_size, full_res_kernel_size),
            round_kernel=False,
        )

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

        # Construct the fixed 10% L2 guard once. Both coarse separation and fine
        # refinement consume this exact immutable guard with identical clipping semantics.
        image_scale = math.sqrt(full_res_width * full_res_height)
        full_res_guard_mask = dilate_component_mask(
            full_res_search_mask,
            AUTO_T_GUARD_DILATION_FRACTION * image_scale,
        )

        separation_T = find_lowest_full_res_threshold(
            full_res_gray,
            work_res_T,
            full_res_seed,
            full_res_guard_mask,
        )

    except ThresholdResolutionError as exc:
        result = AutoThresholdResult(
            threshold=None,
            histogram_start_threshold=histogram_start_T,
            work_res_threshold=None,
            full_res_seed_point=None,
            resolved=False,
            cleaned_component_mask=None,
            reason=str(exc),
        )
        image_state["auto_threshold_result"] = result
        raise

    refinement = refine_threshold(
        full_res_gray,
        separation_T,
        full_res_seed,
        full_res_guard_mask,
    )

    result = AutoThresholdResult(
        threshold=refinement.threshold,
        histogram_start_threshold=histogram_start_T,
        work_res_threshold=work_res_T,
        full_res_seed_point=full_res_seed,
        resolved=True,
        cleaned_component_mask=refinement.cleaned_component_mask,
    )
    image_state["auto_threshold_result"] = result
    return refinement.threshold


# ---------------------------------------------------------------------------
# Final-T full-resolution solar resolution and persistence
# ---------------------------------------------------------------------------
# These constants belong to downstream SolarData construction, not automatic threshold search.
ROI_DILATION_FRACTION = 0.065
GUARD_DILATION_FRACTION = 0.195

# Build the fixed final-T Euclidean cleanup kernel once and reuse it.
SOLAR_COMPONENT_KERNEL = generate_kernel((7, 7), round_kernel=True)


def refine_solar_component_mask(component: np.ndarray) -> np.ndarray:
    """Apply the agreed 7x7 Euclidean OPEN then CLOSE to one solar component."""
    component = np.asarray(component, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    if not np.any(component):
        raise ValueError("component mask is empty")

    component_u8 = np.where(component, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(
        component_u8,
        cv2.MORPH_OPEN,
        SOLAR_COMPONENT_KERNEL,
        iterations=1,
    )
    refined = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        SOLAR_COMPONENT_KERNEL,
        iterations=1,
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
    height, width = shape
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
    return bits.reshape((height, width)) != 0


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
    full_res_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Resolve one selected T and atomically publish its authoritative SolarData."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be 0..255")

    existing = image_state.get("solar_data")
    if isinstance(existing, SolarData) and existing.threshold == threshold:
        # Restore and validate the cached authoritative component before reuse.
        refined_component = decompress_full_mask(existing.component_mask, full_res_gray.shape)
        seed_x, seed_y = existing.seed_point
        height, width = full_res_gray.shape
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError("Stored SolarData seed lies outside the image")
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Stored SolarData seed is outside its refined solar component"
            )
        if full_res_gray[seed_y, seed_x] <= threshold:
            raise ThresholdResolutionError(
                f"Stored SolarData seed is not light at T={threshold}"
            )
        return refined_component

    # Identify the largest enclosed source component at exactly the selected threshold.
    raw_component = largest_enclosed_bright_component(full_res_gray > threshold)
    if raw_component is None:
        raise ThresholdResolutionError(
            f"No enclosed solar component exists at selected T={threshold}"
        )

    # Match the 5-pixel work-resolution support footprint at this image's source scale.
    full_res_max_dim = max(full_res_gray.shape)
    work_res_max_dim = min(full_res_max_dim, WORK_RES_MAX_DIM)
    mapped_kernel_size = 5 * full_res_max_dim / work_res_max_dim
    full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)

    # Generate the source support kernel once before selecting the authoritative final-T seed.
    full_res_seed_kernel = generate_kernel(
        (full_res_kernel_size, full_res_kernel_size),
        round_kernel=False,
    )

    # Select the brightest deeply supported pixel inside the unrefined final-T component.
    full_res_seed = brightest_supported_component_point(
        full_res_gray,
        raw_component,
        full_res_seed_kernel,
    )
    if full_res_seed is None:
        raise ThresholdResolutionError(
            f"Solar component has no {full_res_kernel_size}x{full_res_kernel_size}-"
            "supported authoritative seed"
        )
    seed_x, seed_y = full_res_seed

    # Apply the fixed 7x7 Euclidean OPEN/CLOSE refinement to the authoritative component.
    refined_component = refine_solar_component_mask(raw_component)
    if not np.any(refined_component):
        raise ThresholdResolutionError(
            f"Solar component was removed by refinement at T={threshold}"
        )
    if not refined_component[seed_y, seed_x]:
        raise ThresholdResolutionError(
            "Authoritative solar seed did not survive 7x7 OPEN/CLOSE refinement"
        )

    height, width = full_res_gray.shape
    image_scale = math.sqrt(width * height)

    # Expand the refined component by the fixed 6.5% analysis margin.
    roi_6_5_mask = dilate_component_mask(
        refined_component,
        ROI_DILATION_FRACTION * image_scale,
    )

    # Expand the same refined component independently by the fixed 19.5% guard margin.
    guard_19_5_mask = dilate_component_mask(
        refined_component,
        GUARD_DILATION_FRACTION * image_scale,
    )

    # Preserve the ordered external contour from this exact authoritative refined mask.
    contour = _ordered_external_component_contour(refined_component)

    solar_data = SolarData(
        threshold=threshold,
        seed_point=(seed_x, seed_y),
        component_mask=compress_full_mask(refined_component),
        roi_6_5_mask=compress_full_mask(roi_6_5_mask),
        guard_19_5_mask=compress_full_mask(guard_19_5_mask),
        component_contour=contour,
    )

    # Publish only after every final-T resolution step has succeeded.
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
        self.master_image_payload: bytes | None = None
        self.master_image_shape: tuple[int, int, int] | None = None
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
            min_radius=self.min_radius.get(),
            max_radius=self.max_radius.get(),
            max_error=self.max_error.get(),
            min_coverage=self.min_coverage.get(),
            morphology=self.morphology.get(),
            outer_limb_assistance=self.outer_limb_assistance.get(),
            use_horizon=self.use_horizon.get(),
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
        root.bind("<Return>", self.apply_full_resolution)
        root.bind("<Escape>", self.close)

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
        self.master_image_payload = None
        self.master_image_shape = None
        self.gray_image = None
        self.load_image_at(0)


    def load_image_at(self, index: int):
        if not 0 <= index < len(self.image_paths):
            return

        path = self.image_paths[index]

        # Load source pixels without changing their channel count or integer depth.
        unchanged_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.current_index = index
        self.current_path = path

        if unchanged_image is None:
            self.master_image_payload = None
            self.master_image_shape = None
            self.gray_image = None
            if hasattr(self, "threshold_canvas"):
                self.render_canvas_content(self.threshold_canvas, transparent_bgra())
            if hasattr(self, "color_canvas"):
                self.render_canvas_content(self.color_canvas, transparent_bgra())
            self.status.set(f"Could not load image: {path}")
            self.update_navigation_state()
            return

        # Normalize once to the lossless uint16 BGRA master used by future transforms.
        master_image = normalize_master_bgra16(unchanged_image)

        # Derive the authoritative uint8 processing grayscale directly from that master.
        self.gray_image = master_bgra16_to_gray8(master_image)

        # Derive a temporary uint8 color raster only for the Tk canvas display path.
        display_image = master_bgra16_to_display_bgra8(master_image)

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
            self.render_canvas_content(self.color_canvas, display_image)

        # The canvas retains its own display raster. Keep only a compressed exact master
        # until full-color transform/export code starts consuming the master directly.
        self.master_image_shape = master_image.shape
        self.master_image_payload = compress_master_bgra16(master_image)

        if settings.threshold is None:
            # None has one meaning only: this image has never had T initialized.
            try:
                threshold = find_auto_threshold(self.gray_image, state)
            except ThresholdResolutionError as exc:
                self.status.set(f"Automatic threshold could not be resolved ({exc}).")
            else:
                self.commit_setting_change("threshold", threshold)
        else:
            self.threshold.set(settings.threshold)
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
        readable = (
            has_current
            and self.gray_image is not None
            and self.master_image_payload is not None
            and self.master_image_shape is not None
        )

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

        # Keep the existing GUI policy that the radius sliders extend to at least 1600 px.
        radius_limit = max(1600, round(max(self.args.max_radius, self.args.min_radius) * 1.5))

        slider_specs = [
            ("threshold", "Brightness threshold (dark <= T, light > T)", self.threshold, 0, 255, 1),
            ("min_radius", "Minimum FINAL fitted semi-axis radius (px)", self.min_radius, 1, radius_limit, 1),
            ("max_radius", "Maximum FINAL fitted semi-axis radius (px)", self.max_radius, 1, radius_limit, 1),
            ("max_error", "Maximum average normalized ellipse error (%)", self.max_error, 0.5, 50, 0.1),
            ("min_coverage", "Minimum TOTAL supported ellipse arc (%)", self.min_coverage, 0, 100, 1),
        ]
        for row, spec in enumerate(slider_specs):
            # Build each slider from the same declarative specification.
            self._add_slider(frame, row, *spec)

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
            command=self._handle_morphology_change,
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=self._handle_outer_limb_assistance_change,
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=self._handle_horizon_change,
            state=tk.DISABLED,
        )
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")

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

    def _add_slider(
        self,
        parent,
        row,
        setting_name,
        text,
        variable,
        low,
        high,
        resolution,
    ):
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
            # Format the current typed Tk variable value without conversion lambdas.
            value_label.config(
                text=self._format_slider_value(setting_name, variable.get())
            )

        variable.trace_add("write", update_value)
        update_value()

    def _format_slider_value(self, setting_name, value) -> str:
        """Format one slider's typed Tk variable value for its adjacent label."""
        if setting_name == "threshold":
            return str(value)
        if setting_name in ("min_radius", "max_radius"):
            return f"{value} px"
        if setting_name == "max_error":
            return f"{value:.1f}%"
        if setting_name == "min_coverage":
            return f"{value}% (~{value * 3.6:.0f}°)"
        raise ValueError(f"unknown slider setting: {setting_name}")

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
            selected_threshold = result.threshold
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
            threshold = value
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

    def _handle_morphology_change(self):
        """Commit the morphology checkbox immediately."""
        self.commit_setting_change("morphology", self.morphology.get())

    def _handle_outer_limb_assistance_change(self):
        """Commit the outer-limb-assistance checkbox immediately."""
        self.commit_setting_change(
            "outer_limb_assistance",
            self.outer_limb_assistance.get(),
        )

    def _handle_horizon_change(self):
        """Commit the horizon checkbox immediately."""
        self.commit_setting_change("use_horizon", self.use_horizon.get())

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

        threshold = settings.threshold
        existing = state.get("solar_data")
        reused = isinstance(existing, SolarData) and existing.threshold == threshold
        if isinstance(existing, SolarData) and existing.threshold != threshold:
            state["solar_data"] = None

        threshold_mask = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.render_canvas_content(self.threshold_canvas, threshold_mask)

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


    def apply_full_resolution(self, _event=None):
        # Reuse the same authoritative preview/resolution path with full-resolution status text.
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
            # Use the canonical transparent placeholder when no raster is supplied.
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
            elif (
                unscaled_render_raster.ndim == 3
                and unscaled_render_raster.shape[2] == 3
            ):
                unscaled_render_raster = cv2.cvtColor(
                    unscaled_render_raster, cv2.COLOR_BGR2BGRA
                )
            elif (
                unscaled_render_raster.ndim != 3
                or unscaled_render_raster.shape[2] != 4
            ):
                raise ValueError(
                    "canvas content must be a 2D mask or 3/4-channel image"
                )

        # Canvas-owned display cache retains exactly the normalized unscaled raster.
        canvas._unscaled_render_raster = unscaled_render_raster.copy()

        # Tk may transiently report a 1-pixel unrealized canvas during initial layout;
        # this clamp is GUI lifecycle handling, not image-geometry repair.
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        raster_height, raster_width = unscaled_render_raster.shape[:2]
        scale = min(canvas_width / raster_width, canvas_height / raster_height)
        fitted_size = (
            round(raster_width * scale),
            round(raster_height * scale),
        )

        # Resize the retained raster to the exact fitted canvas dimensions.
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
        # Tk does not retain the Python PhotoImage object. Keep it alive while displayed.
        canvas._tk_photo_image = tk_photo

        # Make the raw threshold stage paintable before sequential final-T processing continues.
        self.root.update_idletasks()


    def close(self, _event=None):
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
