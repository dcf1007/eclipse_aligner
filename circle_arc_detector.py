"""Eclipse alignment GUI with grayscale automatic threshold selection.

This module combines the application's user-interface foundation with the tested
image-only automatic threshold stages. The GUI owns image navigation, per-image
processing settings, control interaction, explicit processing-stage orchestration,
and cached automatic-threshold results. Color-to-grayscale conversion is an
input-stage responsibility and is performed before the threshold algorithm is called.

All processing controls are per-image. ``DetectorApp.image_state`` is keyed by the
absolute image path. ``settings.threshold`` is special: ``None`` means the threshold
has never been initialized, and once initialized its exact integer value is always
stored regardless of whether it came from Auto T or manual input. Other controls may
still use sparse overrides relative to application defaults.

Slider labels update continuously, but a setting is committed only after mouse
release or the final keyboard key release. Checkboxes and radio buttons commit
immediately. Every actual setting change passes through
``commit_setting_change(setting_name, value)``, which synchronizes the Tk variable,
persists the per-image setting, invalidates incompatible derived state, and clears
stale threshold-canvas content. Processing is started only by explicit processing
actions such as Auto T or Refresh Preview.

Automatic threshold selection has two explicit algorithmic stages. Stage A,
``find_separation_threshold()``, progressively fills the current image's single
``AutoThresholdResult`` through work-resolution and full-resolution separation.
The GUI renders the completed Stage-A component, then Stage B,
``refine_threshold()``, fills the refined threshold and component in that same
object and returns the final T. Auto T stops after rendering this refined component;
it does not build SolarData.

SolarData integration of the cached Auto-T component is intentionally deferred.
``resolve_threshold()`` therefore remains the existing downstream selected-T path in
this revision: it independently establishes the selected-T component/seed, applies
its existing 7x7 Euclidean OPEN/CLOSE, constructs/stores SolarData, and returns that
mask for final display.

The automatic-threshold algorithm uses authoritative 8-bit grayscale with fixed
semantics ``dark = gray <= T`` and ``light = gray > T``. It derives a <=1200-pixel
working raster, starts from the left valley of the rightmost locally smoothed
histogram mode, establishes one 5x5-square-supported work-resolution solar seed, and
tracks that same 8-connected component downward. The mature work-resolution
component is resized to full resolution only to delimit the full-resolution seed
search and fixed 10% L2-distance guard. The equivalent full-resolution seed-support
kernel remains square and scales from the realized work/full-resolution ratio.

The coarse full-resolution search uses a 7x7 Euclidean OPEN/CLOSE and finds the
lowest defensible T whose component containing the authoritative seed stays inside
the fixed guard. Fine refinement may only raise that T. It evaluates T through T+10
at full resolution using progressive 3x3, 5x5, and 7x7 Euclidean OPEN/CLOSE cleanup,
the same seed and fixed guard, one first-separated raw normalization reference, and
the agreed geometry/photometric score. The exact maximum score wins; there is no
rounding plateau, epsilon, crop coordinate system, second seed, metadata prior, or
horizon special case in automatic threshold selection.
"""


import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
import struct
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


def morphological_cleanup(
    source: np.ndarray,
    kernel: np.ndarray,
    threshold: int | None = None,
) -> np.ndarray:
    """Apply one OPEN->CLOSE cleanup to grayscale-at-T or an existing binary mask."""
    source = np.asarray(source)
    kernel = np.asarray(kernel, dtype=np.uint8)
    if kernel.ndim != 2 or kernel.size == 0 or not np.any(kernel):
        raise ValueError("kernel must be a non-empty two-dimensional mask")

    if threshold is not None:
        if source.ndim != 2 or source.dtype != np.uint8:
            raise ValueError(
                "thresholded morphology requires authoritative 2D uint8 grayscale"
            )
        if not isinstance(threshold, (int, np.integer)) or not 0 <= int(threshold) <= 255:
            raise ValueError("threshold must be an integer from 0 to 255")
        cleaned = cv2.compare(source, int(threshold), cv2.CMP_GT)
    else:
        if source.ndim != 2 or source.dtype not in (bool, np.uint8):
            raise ValueError("binary morphology requires a 2D bool or uint8 mask")
        cleaned = np.where(source != 0, 255, 0).astype(np.uint8)

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )
    return cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )


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


def compress_array(array: np.ndarray) -> bytes:
    """Compress one supported 1-, 8-, or 16-bit internal image array."""
    array = np.asarray(array)
    if array.ndim not in (2, 3):
        raise ValueError("array must be two- or three-dimensional")

    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("array dimensions must be positive")

    channels = 1 if array.ndim == 2 else array.shape[2]
    if channels not in (1, 3, 4):
        raise ValueError("array channel count must be 1, 3, or 4")

    if array.dtype == bool:
        if array.ndim != 2:
            raise ValueError("1-bit arrays must be two-dimensional")
        bit_depth = 1
        encoded_pixels = np.packbits(array.reshape(-1)).tobytes()
    elif array.dtype == np.uint8:
        bit_depth = 8
        encoded_pixels = np.ascontiguousarray(array).tobytes()
    elif array.dtype == np.uint16:
        bit_depth = 16
        # Persist uint16 samples in an explicit byte order rather than native endian.
        encoded_pixels = np.ascontiguousarray(
            array.astype(np.dtype("<u2"), copy=False)
        ).tobytes()
    else:
        raise ValueError("array dtype must be bool, uint8, or uint16")

    header = struct.pack("<IIB", height, width, bit_depth)
    return zlib.compress(header + encoded_pixels, level=1)


def decompress_array(payload: bytes) -> np.ndarray:
    """Restore one self-describing array produced by ``compress_array``."""
    raw = zlib.decompress(payload)
    header_size = struct.calcsize("<IIB")
    if len(raw) < header_size:
        raise ValueError("compressed array payload is truncated")

    height, width, bit_depth = struct.unpack("<IIB", raw[:header_size])
    if height <= 0 or width <= 0:
        raise ValueError("compressed array dimensions must be positive")

    encoded_pixels = raw[header_size:]
    pixel_count = height * width

    if bit_depth == 1:
        expected_bytes = (pixel_count + 7) // 8
        if len(encoded_pixels) != expected_bytes:
            raise ValueError(
                f"compressed 1-bit array has {len(encoded_pixels)} bytes; "
                f"expected {expected_bytes}"
            )
        packed = np.frombuffer(encoded_pixels, dtype=np.uint8)
        return np.unpackbits(packed, count=pixel_count).reshape((height, width)) != 0

    if bit_depth == 8:
        bytes_per_channel = pixel_count
        dtype = np.uint8
    elif bit_depth == 16:
        bytes_per_channel = pixel_count * 2
        dtype = np.dtype("<u2")
    else:
        raise ValueError(f"unsupported compressed array bit depth: {bit_depth}")

    if len(encoded_pixels) % bytes_per_channel != 0:
        raise ValueError("compressed array pixel payload has invalid length")
    channels = len(encoded_pixels) // bytes_per_channel
    if channels not in (1, 3, 4):
        raise ValueError(
            f"compressed array has unsupported inferred channel count: {channels}"
        )

    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.frombuffer(encoded_pixels, dtype=dtype).reshape(shape)


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


@dataclass
class AutoThresholdResult:
    """Progressively accumulated authoritative results for one Auto-T run."""

    histogram_start_threshold: int | None = None

    work_res_seed_point: tuple[int, int] | None = None
    work_res_separation_threshold: int | None = None
    work_res_separation_component_mask: bytes | None = None

    full_res_seed_point: tuple[int, int] | None = None

    full_res_separation_threshold: int | None = None
    full_res_separation_component_mask: bytes | None = None
    full_res_separation_guard_mask: bytes | None = None

    full_res_refined_threshold: int | None = None
    full_res_refined_component_mask: bytes | None = None

    failure_reason: str | None = None



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

# Photometric edge profiles retain a fixed +/-25-pixel observation radius so each
# profile normally contains the full limb transition plus interior/exterior context.
# Unlike the previous implementation, this acquisition radius does not define the
# edge score scale.
EDGE_PROFILE_RADIUS_PX = 25

GUARD_BOUNDARY_KERNEL = generate_kernel((3, 3), round_kernel=False)
SEPARATION_KERNEL = generate_kernel((7, 7), round_kernel=True)
SOLAR_CLEANUP_KERNELS = (
    generate_kernel((3, 3), round_kernel=True),
    generate_kernel((5, 5), round_kernel=True),
    SEPARATION_KERNEL,
)


@dataclass(frozen=True)
class ThresholdMeasurement:
    """Measurements retained for one valid cleaned threshold candidate."""

    threshold: int
    filled_area: int
    roughness: float
    hole_quality: float
    edge_distance: float
    edge_reliability: float







def extract_component(
    binary_mask: np.ndarray,
    seed_point: tuple[int, int],
) -> np.ndarray | None:
    """Return the 8-connected component of ``binary_mask`` containing ``seed_point``."""
    if binary_mask.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")

    seed_x, seed_y = seed_point
    height, width = binary_mask.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ValueError("seed lies outside binary-mask raster")
    if binary_mask[seed_y, seed_x] == 0:
        return None

    flood = np.where(binary_mask != 0, 255, 0).astype(np.uint8)
    cv2.floodFill(flood, None, (seed_x, seed_y), 128, flags=8)
    component = flood == 128
    return component if np.any(component) else None



def find_work_res_separation_threshold(
    work_res_gray: np.ndarray,
    start_T: int,
    work_res_seed_kernel: np.ndarray,
) -> tuple[tuple[int, int], int, np.ndarray]:
    """Return the work seed, lowest separated T, and tracked component."""
    work_res_seed_point: tuple[int, int] | None = None
    work_res_T: int | None = None
    work_res_component: np.ndarray | None = None

    for threshold in range(start_T, -1, -1):
        if work_res_seed_point is None:
            # Before identity is established, propose the largest enclosed bright component.
            component = largest_enclosed_bright_component(work_res_gray > threshold)
            if component is not None:
                # Accept the first candidate with the required full seed-support footprint.
                work_res_seed_point = brightest_supported_component_point(
                    work_res_gray,
                    component,
                    work_res_seed_kernel,
                )
                if work_res_seed_point is not None:
                    work_res_T = threshold
                    work_res_component = component
        else:
            component = extract_component(work_res_gray > threshold, work_res_seed_point)
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

    if work_res_seed_point is None or work_res_T is None or work_res_component is None:
        kernel_height, kernel_width = work_res_seed_kernel.shape
        raise ThresholdResolutionError(
            f"No {kernel_width}x{kernel_height}-supported enclosed bright component "
            "exists through T=0"
        )
    return work_res_seed_point, work_res_T, work_res_component



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


def find_guard_boundary(guard_mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel inner boundary of a non-empty boolean guard mask."""
    if guard_mask.ndim != 2 or not np.any(guard_mask):
        raise ValueError("guard must be a non-empty two-dimensional mask")

    guard_u8 = np.where(guard_mask, 255, 0).astype(np.uint8)
    eroded_guard = cv2.erode(
        guard_u8,
        GUARD_BOUNDARY_KERNEL,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
    return guard_mask & ~eroded_guard



def find_full_res_separation_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Return the lowest separated T and its already-computed D7 component."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if not 0 <= start_T <= 255:
        raise ValueError("start threshold must be 0..255")
    if full_res_guard_mask.ndim != 2 or full_res_guard_mask.shape != full_res_gray.shape:
        raise ValueError("full-resolution guard and grayscale image must have identical shapes")
    if not np.any(full_res_guard_mask):
        raise ThresholdResolutionError("Full-resolution Auto-T guard is empty")

    seed_x, seed_y = full_res_seed_point
    full_res_height, full_res_width = full_res_gray.shape
    if not (0 <= seed_x < full_res_width and 0 <= seed_y < full_res_height):
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the image")
    if not full_res_guard_mask[seed_y, seed_x]:
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the Auto-T guard")

    full_res_guard_boundary = find_guard_boundary(full_res_guard_mask)

    # Evaluate the starting T after the fixed D7 cleanup used by coarse separation.
    binary = morphological_cleanup(full_res_gray, SEPARATION_KERNEL, start_T)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed does not survive D7 cleanup at start T={start_T}"
        )
    binary[~full_res_guard_mask] = 0
    component = extract_component(binary, full_res_seed_point)
    if component is None:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed is unavailable after guard clipping at start T={start_T}"
        )

    if not np.any(component & full_res_guard_boundary):
        best_T = start_T
        best_component = component
        for threshold in range(start_T - 1, -1, -1):
            binary = morphological_cleanup(
                full_res_gray,
                SEPARATION_KERNEL,
                threshold,
            )
            binary[~full_res_guard_mask] = 0
            component = extract_component(binary, full_res_seed_point)
            if component is None or np.any(component & full_res_guard_boundary):
                break
            best_T = threshold
            best_component = component
        return best_T, best_component

    for threshold in range(start_T + 1, 256):
        binary = morphological_cleanup(full_res_gray, SEPARATION_KERNEL, threshold)
        if binary[seed_y, seed_x] == 0:
            break
        binary[~full_res_guard_mask] = 0
        component = extract_component(binary, full_res_seed_point)
        if component is not None and not np.any(component & full_res_guard_boundary):
            return threshold, component

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became separated after D7 cleanup"
    )



def find_separation_threshold(
    full_res_gray: np.ndarray,
    auto_threshold_result: AutoThresholdResult,
) -> None:
    """Run Auto-T Stage A and fill separation results in the supplied state object."""
    if full_res_gray.ndim != 2 or full_res_gray.dtype != np.uint8:
        raise ValueError("automatic thresholding requires authoritative 2D uint8 grayscale")
    if not isinstance(auto_threshold_result, AutoThresholdResult):
        raise ValueError("Stage A requires an AutoThresholdResult")

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
    auto_threshold_result.histogram_start_threshold = (
        find_histogram_start_threshold(work_res_gray)
    )
    work_res_seed_kernel = generate_kernel((5, 5), round_kernel=False)

    resolution_step = "work-resolution separation"
    try:
        (
            auto_threshold_result.work_res_seed_point,
            auto_threshold_result.work_res_separation_threshold,
            work_res_component,
        ) = find_work_res_separation_threshold(
            work_res_gray,
            auto_threshold_result.histogram_start_threshold,
            work_res_seed_kernel,
        )
        auto_threshold_result.work_res_separation_component_mask = compress_array(
            work_res_component
        )

        # Transfer only the mature work component geometry onto the exact source raster.
        full_res_search_mask = resize_img(
            work_res_component,
            (full_res_width, full_res_height),
            mask=True,
        )

        # Scale the fixed 5x5 square work support to the realized full-resolution scale.
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

        resolution_step = "full-resolution seed selection"
        auto_threshold_result.full_res_seed_point = brightest_supported_component_point(
            full_res_gray,
            full_res_search_mask,
            full_res_seed_kernel,
        )
        if auto_threshold_result.full_res_seed_point is None:
            raise ThresholdResolutionError(
                f"Transferred solar component has no {full_res_kernel_size}x"
                f"{full_res_kernel_size}-supported full-resolution seed"
            )

        resolution_step = "full-resolution guard construction"
        image_scale = math.sqrt(full_res_width * full_res_height)
        full_res_guard_mask = dilate_component_mask(
            full_res_search_mask,
            AUTO_T_GUARD_DILATION_FRACTION * image_scale,
        )
        auto_threshold_result.full_res_separation_guard_mask = compress_array(
            full_res_guard_mask
        )

        resolution_step = "full-resolution separation"
        (
            auto_threshold_result.full_res_separation_threshold,
            full_res_separation_component,
        ) = find_full_res_separation_threshold(
            full_res_gray,
            auto_threshold_result.work_res_separation_threshold,
            auto_threshold_result.full_res_seed_point,
            full_res_guard_mask,
        )
        auto_threshold_result.full_res_separation_component_mask = compress_array(
            full_res_separation_component
        )
    except ThresholdResolutionError as exc:
        auto_threshold_result.failure_reason = f"{resolution_step}: {exc}"
        raise


def find_external_contour(component: np.ndarray) -> np.ndarray:
    """Return the largest-area external CHAIN_APPROX_NONE contour of a non-empty component."""
    if component.ndim != 2 or not np.any(component):
        raise ThresholdResolutionError("solar component is empty or not two-dimensional")
    component_u8 = np.where(component != 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ThresholdResolutionError("solar component has no external contour")
    return max(contours, key=cv2.contourArea)



def measure_filled_area(contour: np.ndarray) -> int:
    """Return raster-equivalent area enclosed by the external lattice contour."""
    points = contour[:, 0, :].astype(np.int64)
    following = np.roll(points, -1, axis=0)
    distance = np.abs(following - points)
    boundary_points = int(np.gcd(distance[:, 0], distance[:, 1]).sum())
    polygon_area = float(cv2.contourArea(contour))
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


def measure_hole_quality(
    contour: np.ndarray,
    component_area: int,
    filled_area: int,
) -> float:
    """Return external-boundary share after converting internal dark area to perimeter."""
    if filled_area <= 0:
        raise ValueError("filled area must be positive")
    if component_area < 0 or component_area > filled_area:
        raise ValueError("component area must be between zero and filled area")
    external_perimeter = float(cv2.arcLength(contour, True))
    if not math.isfinite(external_perimeter) or external_perimeter <= 0.0:
        raise ThresholdResolutionError("external contour perimeter is empty")
    hole_area = filled_area - component_area
    minimum_hole_perimeter = 2.0 * math.sqrt(math.pi * hole_area)
    return external_perimeter / (external_perimeter + minimum_hole_perimeter)


def _sample_grayscale_profiles(
    full_res_gray: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one averaged outward grayscale profile per raster-scale polygon side."""
    radius = EDGE_PROFILE_RADIUS_PX
    profile_width = 2 * radius + 1
    empty = (
        np.empty((0, profile_width), dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )

    # Integer contour coordinates represent pixel-cell locations. Simplify only
    # deviations no larger than the half-pixel cell diagonal, so raster stair-steps
    # do not create thousands of nearly duplicate local directions.
    raster_tolerance = math.hypot(0.5, 0.5)
    polygon = cv2.approxPolyDP(contour, raster_tolerance, True)
    points = polygon[:, 0, :].astype(np.float64)
    if len(points) < 3:
        return empty

    oriented_area = float(cv2.contourArea(polygon, oriented=True))
    if oriented_area == 0.0:
        return empty
    orientation = 1.0 if oriented_area > 0.0 else -1.0

    starts = points
    vectors = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    usable = np.isfinite(lengths) & (lengths > np.finfo(np.float64).eps)
    starts = starts[usable]
    vectors = vectors[usable]
    lengths = lengths[usable]
    if len(lengths) == 0:
        return empty
    outwards = orientation * np.column_stack((vectors[:, 1], -vectors[:, 0]))
    outwards /= lengths[:, None]
    counts = np.maximum(1, np.ceil(lengths).astype(np.int32))

    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    height, width = full_res_gray.shape
    full_res_gray_float = full_res_gray.astype(np.float32, copy=False)
    bases: list[np.ndarray] = []
    kept_outwards: list[np.ndarray] = []
    sample_counts: list[int] = []
    segment_lengths: list[float] = []

    for start, vector, segment_length, outward, sample_count in zip(
        starts, vectors, lengths, outwards, counts
    ):
        fraction = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_count
        base = start[None, :] + fraction[:, None] * vector[None, :]

        x_min = base[:, 0].min() - radius * abs(outward[0])
        x_max = base[:, 0].max() + radius * abs(outward[0])
        y_min = base[:, 1].min() - radius * abs(outward[1])
        y_max = base[:, 1].max() + radius * abs(outward[1])
        if x_min < 0 or x_max > width - 1 or y_min < 0 or y_max > height - 1:
            continue

        bases.append(base.astype(np.float32))
        kept_outwards.append(outward.astype(np.float32))
        sample_counts.append(int(sample_count))
        segment_lengths.append(float(segment_length))

    if not bases:
        return empty

    sample_counts_array = np.asarray(sample_counts, dtype=np.int32)
    base_points = np.vstack(bases)
    outward_normals = np.repeat(
        np.asarray(kept_outwards, dtype=np.float32),
        sample_counts_array,
        axis=0,
    )
    segment_ids = np.repeat(
        np.arange(len(sample_counts), dtype=np.int32),
        sample_counts_array,
    )
    profile_sums = np.zeros((len(sample_counts), profile_width), dtype=np.float64)

    # cv::remap stores map dimensions in signed 16-bit coordinates internally.
    # Chunk only for that library limit; it does not alter the sampled geometry.
    max_remap_rows = np.iinfo(np.int16).max - 1
    for first in range(0, len(base_points), max_remap_rows):
        last = min(len(base_points), first + max_remap_rows)
        xy = (
            base_points[first:last, None, :]
            + outward_normals[first:last, None, :] * offsets[None, :, None]
        )
        sampled = cv2.remap(
            full_res_gray_float,
            xy[:, :, 0],
            xy[:, :, 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float64, copy=False)
        np.add.at(profile_sums, segment_ids[first:last], sampled)

    return (
        profile_sums / sample_counts_array[:, None],
        np.asarray(segment_lengths, dtype=np.float64),
    )


def measure_edge_alignment(
    full_res_gray: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, float]:
    """Return photometric edge distance and image-local profile reliability."""
    profiles, segment_lengths = _sample_grayscale_profiles(full_res_gray, contour)
    if len(profiles) == 0:
        return math.nan, 0.0

    positions = np.arange(
        -EDGE_PROFILE_RADIUS_PX,
        EDGE_PROFILE_RADIUS_PX + 1,
        dtype=np.float64,
    )
    falling_derivative = -np.gradient(profiles, positions, axis=1)
    edge_positions = np.full(len(profiles), np.nan, dtype=np.float64)
    model_centers = np.full(len(profiles), np.nan, dtype=np.float64)
    model_sigmas = np.full(len(profiles), np.nan, dtype=np.float64)
    profile_reliability = np.zeros(len(profiles), dtype=np.float64)
    machine_epsilon = np.finfo(np.float64).eps

    for index, profile in enumerate(profiles):
        derivative = falling_derivative[index]
        peak_index = int(np.argmax(derivative))
        peak_value = float(derivative[peak_index])
        if not math.isfinite(peak_value) or peak_value <= 0.0:
            continue

        transition_start = peak_index
        transition_stop = peak_index
        while transition_start > 0 and derivative[transition_start - 1] > 0.0:
            transition_start -= 1
        while (
            transition_stop + 1 < len(derivative)
            and derivative[transition_stop + 1] > 0.0
        ):
            transition_stop += 1

        # The observed edge is where the approximately linear falling transition
        # meets the observed exterior trend. Both intervals come directly from the
        # dominant positive derivative lobe; no fixed fit width or slope fraction is used.
        if (
            transition_stop - transition_start + 1 >= 2
            and transition_stop + 1 < len(positions) - 1
        ):
            transition_x = positions[transition_start : transition_stop + 1]
            transition_y = profile[transition_start : transition_stop + 1]
            count = len(transition_x)
            sum_x = float(np.sum(transition_x))
            sum_y = float(np.sum(transition_y))
            sum_xx = float(np.dot(transition_x, transition_x))
            sum_xy = float(np.dot(transition_x, transition_y))
            denominator = count * sum_xx - sum_x * sum_x
            denominator_limit = machine_epsilon * max(
                1.0,
                abs(count * sum_xx),
                abs(sum_x * sum_x),
            )
            if abs(denominator) > denominator_limit:
                transition_slope = (count * sum_xy - sum_x * sum_y) / denominator
                transition_intercept = (sum_y - transition_slope * sum_x) / count

                outer_x = positions[transition_stop + 1 :]
                outer_y = profile[transition_stop + 1 :]
                count = len(outer_x)
                sum_x = float(np.sum(outer_x))
                sum_y = float(np.sum(outer_y))
                sum_xx = float(np.dot(outer_x, outer_x))
                sum_xy = float(np.dot(outer_x, outer_y))
                denominator = count * sum_xx - sum_x * sum_x
                denominator_limit = machine_epsilon * max(
                    1.0,
                    abs(count * sum_xx),
                    abs(sum_x * sum_x),
                )
                if abs(denominator) > denominator_limit:
                    outer_slope = (count * sum_xy - sum_x * sum_y) / denominator
                    outer_intercept = (sum_y - outer_slope * sum_x) / count
                    slope_difference = transition_slope - outer_slope
                    slope_limit = machine_epsilon * max(
                        1.0,
                        abs(transition_slope),
                        abs(outer_slope),
                    )
                    if transition_slope < 0.0 and abs(slope_difference) > slope_limit:
                        edge_positions[index] = (
                            outer_intercept - transition_intercept
                        ) / slope_difference

        # An ideal blurred step has a Gaussian derivative and therefore an
        # error-function intensity profile. Derive its center/width from the same
        # dominant derivative lobe. R^2 is computed for all usable profiles together
        # below so the identical model does not pay Python overhead once per sample.
        transition_derivative = np.maximum(
            derivative[transition_start : transition_stop + 1],
            0.0,
        )
        derivative_mass = float(np.sum(transition_derivative))
        if not math.isfinite(derivative_mass) or derivative_mass <= machine_epsilon:
            continue
        transition_x = positions[transition_start : transition_stop + 1]
        center = float(np.dot(transition_derivative, transition_x) / derivative_mass)
        variance = float(
            np.dot(transition_derivative, (transition_x - center) ** 2)
            / derivative_mass
        )
        sigma = math.sqrt(max(0.0, variance))
        if not math.isfinite(sigma) or sigma <= machine_epsilon:
            continue
        model_centers[index] = center
        model_sigmas[index] = sigma

    model_valid = np.isfinite(model_centers) & np.isfinite(model_sigmas)
    if np.any(model_valid):
        valid_profiles = profiles[model_valid]
        valid_centers = model_centers[model_valid]
        valid_sigmas = model_sigmas[model_valid]
        sqrt_two = math.sqrt(2.0)
        basis = np.fromiter(
            (
                0.5 * (1.0 + math.erf((center - position) / (sigma * sqrt_two)))
                for center, sigma in zip(valid_centers, valid_sigmas)
                for position in positions
            ),
            dtype=np.float64,
            count=len(valid_profiles) * len(positions),
        ).reshape(len(valid_profiles), len(positions))

        profile_means = np.mean(valid_profiles, axis=1)
        basis_means = np.mean(basis, axis=1)
        centered_profiles = valid_profiles - profile_means[:, None]
        centered_basis = basis - basis_means[:, None]
        basis_energy = np.sum(centered_basis * centered_basis, axis=1)
        profile_energy = np.sum(centered_profiles * centered_profiles, axis=1)
        covariance = np.sum(centered_basis * centered_profiles, axis=1)
        amplitudes = np.divide(
            covariance,
            basis_energy,
            out=np.zeros_like(covariance),
            where=basis_energy > machine_epsilon,
        )
        reliable = (
            (basis_energy > machine_epsilon)
            & (profile_energy > machine_epsilon)
            & np.isfinite(amplitudes)
            & (amplitudes > 0.0)
        )
        if np.any(reliable):
            backgrounds = profile_means - amplitudes * basis_means
            fitted = backgrounds[:, None] + amplitudes[:, None] * basis
            residual_energy = np.sum((valid_profiles - fitted) ** 2, axis=1)
            r_squared = np.zeros(len(valid_profiles), dtype=np.float64)
            r_squared[reliable] = np.clip(
                1.0 - residual_energy[reliable] / profile_energy[reliable],
                0.0,
                1.0,
            )
            profile_reliability[model_valid] = r_squared

    weights = segment_lengths * profile_reliability
    valid = np.isfinite(edge_positions) & np.isfinite(weights) & (weights > 0.0)
    if np.any(valid):
        values = np.abs(edge_positions[valid])
        valid_weights = weights[valid]
        order = np.argsort(values)
        values = values[order]
        valid_weights = valid_weights[order]
        cutoff = 0.5 * float(np.sum(valid_weights))
        edge_distance = float(
            values[
                np.searchsorted(
                    np.cumsum(valid_weights),
                    cutoff,
                    side="left",
                )
            ]
        )
    else:
        edge_distance = math.nan

    total_length = float(np.sum(segment_lengths))
    edge_reliability = (
        float(np.dot(segment_lengths, profile_reliability) / total_length)
        if total_length > 0.0
        else 0.0
    )
    return edge_distance, edge_reliability


def refine_threshold(
    full_res_gray: np.ndarray,
    auto_threshold_result: AutoThresholdResult,
) -> int:
    """Run Auto-T Stage B, fill the supplied result, and return its winning T."""
    if full_res_gray.ndim != 2 or full_res_gray.dtype != np.uint8:
        raise ValueError("threshold refinement requires authoritative uint8 grayscale")
    if not isinstance(auto_threshold_result, AutoThresholdResult):
        raise ValueError("Stage B requires an AutoThresholdResult")
    if auto_threshold_result.failure_reason is not None:
        raise ValueError("cannot refine an Auto-T result that already failed")
    if (
        auto_threshold_result.full_res_separation_threshold is None
        or auto_threshold_result.full_res_seed_point is None
        or auto_threshold_result.full_res_separation_guard_mask is None
    ):
        raise ValueError("threshold refinement requires a completed Stage-A separation")

    base_threshold = auto_threshold_result.full_res_separation_threshold
    full_res_seed_point = auto_threshold_result.full_res_seed_point
    full_res_guard_mask = decompress_array(
        auto_threshold_result.full_res_separation_guard_mask
    )
    if (
        full_res_guard_mask.dtype != bool
        or full_res_guard_mask.ndim != 2
        or full_res_guard_mask.shape != full_res_gray.shape
        or not np.any(full_res_guard_mask)
    ):
        raise ValueError("stored Stage-A guard must be a non-empty mask matching grayscale")
    if not 0 <= base_threshold <= 255:
        raise ValueError("base threshold must be 0..255")

    seed_x, seed_y = full_res_seed_point
    full_res_height, full_res_width = full_res_gray.shape
    if not (0 <= seed_x < full_res_width and 0 <= seed_y < full_res_height):
        raise ValueError("full-resolution seed lies outside grayscale raster")
    if not full_res_guard_mask[seed_y, seed_x]:
        raise ValueError("full-resolution seed must lie inside the fixed guard")

    try:
        full_res_guard_boundary = find_guard_boundary(full_res_guard_mask)
        measurements: list[ThresholdMeasurement] = []
        compressed_masks: dict[int, bytes] = {}
        raw_reference_area: int | None = None
        raw_reference_roughness: float | None = None

        max_threshold = min(255, base_threshold + MAX_T_REFINEMENT_STEPS)
        for threshold in range(base_threshold, max_threshold + 1):
            threshold_mask = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)

            cleaned = threshold_mask
            for kernel in SOLAR_CLEANUP_KERNELS:
                cleaned = morphological_cleanup(cleaned, kernel)
            cleaned[~full_res_guard_mask] = 0
            cleaned_component = extract_component(cleaned, full_res_seed_point)

            if (
                cleaned_component is not None
                and not np.any(cleaned_component & full_res_guard_boundary)
            ):
                contour = find_external_contour(cleaned_component)
                filled_area = measure_filled_area(contour)
                roughness = measure_roughness(contour, filled_area)
                component_area = int(np.count_nonzero(cleaned_component))
                hole_quality = measure_hole_quality(
                    contour,
                    component_area,
                    filled_area,
                )
                edge_distance, edge_reliability = measure_edge_alignment(
                    full_res_gray,
                    contour,
                )
                measurements.append(
                    ThresholdMeasurement(
                        threshold=threshold,
                        filled_area=filled_area,
                        roughness=roughness,
                        hole_quality=hole_quality,
                        edge_distance=edge_distance,
                        edge_reliability=edge_reliability,
                    )
                )
                compressed_masks[threshold] = compress_array(cleaned_component)

            # Raw geometry only anchors the largest/roughest end of the score scale.
            # Measure the first separated raw component once, then stop evaluating raw.
            if raw_reference_area is None:
                raw_mask = threshold_mask.copy()
                raw_mask[~full_res_guard_mask] = 0
                raw_component = extract_component(raw_mask, full_res_seed_point)
                if (
                    raw_component is not None
                    and not np.any(raw_component & full_res_guard_boundary)
                ):
                    raw_contour = find_external_contour(raw_component)
                    raw_reference_area = measure_filled_area(raw_contour)
                    raw_reference_roughness = measure_roughness(
                        raw_contour,
                        raw_reference_area,
                    )

        if not measurements:
            raise ThresholdResolutionError(
                "no separated cleaned solar component exists in the refinement window"
            )
        if raw_reference_area is None or raw_reference_roughness is None:
            raise ThresholdResolutionError(
                "no separated raw reference exists in the refinement window"
            )

        max_roughness = max(
            raw_reference_roughness,
            *(measurement.roughness for measurement in measurements),
        )
        max_area = max(
            raw_reference_area,
            *(measurement.filled_area for measurement in measurements),
        )
        if not math.isfinite(max_roughness) or max_roughness <= 0.0 or max_area <= 0:
            raise ThresholdResolutionError("invalid within-image score scale")

        edge_reliability = float(
            np.median([measurement.edge_reliability for measurement in measurements])
        )
        if not math.isfinite(edge_reliability):
            edge_reliability = 0.0

        best_threshold: int | None = None
        best_score = -math.inf
        for measurement in measurements:
            q_roughness = 1.0 - measurement.roughness / max_roughness
            q_area = measurement.filled_area / max_area
            q_edge = (
                1.0 / (1.0 + measurement.edge_distance)
                if math.isfinite(measurement.edge_distance)
                else 0.0
            )
            score = (
                q_roughness
                + measurement.hole_quality
                + 0.5 * q_area
                + edge_reliability * q_edge
            )
            # Measurements are in ascending T order. Strict '>' therefore keeps the
            # lower T only when two floating-point scores are exactly equal.
            if score > best_score:
                best_score = score
                best_threshold = measurement.threshold

        if best_threshold is None:
            raise ThresholdResolutionError("threshold refinement produced no score winner")

        auto_threshold_result.full_res_refined_threshold = best_threshold
        auto_threshold_result.full_res_refined_component_mask = compressed_masks[
            best_threshold
        ]
        return best_threshold
    except ThresholdResolutionError as exc:
        auto_threshold_result.failure_reason = f"fine refinement: {exc}"
        raise


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

    return morphological_cleanup(component, SOLAR_COMPONENT_KERNEL) != 0


@dataclass(frozen=True)
class SolarData:
    """Refined solar geometry established at exactly one full-resolution threshold T."""

    threshold: int
    seed_point: tuple[int, int]
    component_mask: bytes
    roi_6_5_mask: bytes
    guard_19_5_mask: bytes
    component_contour: np.ndarray


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
        refined_component = decompress_array(existing.component_mask)
        if refined_component.dtype != bool or refined_component.shape != full_res_gray.shape:
            raise ThresholdResolutionError(
                "Stored SolarData component mask does not match the current image"
            )
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
        component_mask=compress_array(refined_component),
        roi_6_5_mask=compress_array(roi_6_5_mask),
        guard_19_5_mask=compress_array(guard_19_5_mask),
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
        self.gray_image = None

        # Per-image state keeps settings, the cached automatic threshold result,
        # and post-threshold SolarData when solar geometry has been built.
        # No ImageState wrapper class is needed: the outer dictionary directly
        # expresses the image-to-state hierarchy.
        self.image_state: dict[str, dict[str, object]] = {}


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
        self.gray_image = None
        self.load_image_at(0)


    def load_image_at(self, index: int):
        if not 0 <= index < len(self.image_paths):
            return

        path = self.image_paths[index]
        self.current_index = index
        self.current_path = path

        # Navigation/filename is the first visible change for the newly selected image.
        self.update_navigation_state()
        self.root.update_idletasks()

        try:
            with self.processing_ui():
                # Load source pixels without changing their channel count or integer depth.
                unchanged_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                if unchanged_image is None:
                    self.master_image_payload = None
                    self.gray_image = None
                    if hasattr(self, "threshold_canvas"):
                        self.render_canvas_content(self.threshold_canvas, transparent_bgra())
                    if hasattr(self, "color_canvas"):
                        self.render_canvas_content(self.color_canvas, transparent_bgra())
                    self.status.set(f"Could not load image: {path}")
                    return

                # Normalize the unchanged load directly to the agreed lossless uint16
                # BGRA master. uint8 inputs are expanded by exactly 257 so every source
                # code value is preserved in the uint16 representation.
                source = np.asarray(unchanged_image)
                if source.dtype not in (np.uint8, np.uint16):
                    raise ValueError(
                        f"master image dtype must be uint8 or uint16, got {source.dtype}"
                    )
                if source.ndim == 2:
                    master_image = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
                elif source.ndim == 3 and source.shape[2] == 3:
                    master_image = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
                elif source.ndim == 3 and source.shape[2] == 4:
                    master_image = source
                else:
                    raise ValueError(f"unsupported master image shape: {source.shape}")
                if master_image.dtype == np.uint8:
                    master_image = master_image.astype(np.uint16) * 257
                master_image = np.ascontiguousarray(master_image, dtype=np.uint16)

                # Authoritative threshold processing is always uint8 grayscale derived
                # directly from the lossless master with the fixed full-range mapping.
                gray16 = cv2.cvtColor(master_image, cv2.COLOR_BGRA2GRAY)
                self.gray_image = (
                    (gray16.astype(np.uint32) + 128) // 257
                ).astype(np.uint8)

                # The canvas receives an explicit uint8 display raster. The master itself
                # remains untouched and retained losslessly for later transforms/export.
                display_image = (
                    (master_image.astype(np.uint32) + 128) // 257
                ).astype(np.uint8)

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

                # Retain the exact master in the shared self-describing array format.
                self.master_image_payload = compress_array(master_image)

                # First visible processing stage: source color + authoritative grayscale.
                if hasattr(self, "color_canvas"):
                    self.render_canvas_content(self.color_canvas, display_image)
                if hasattr(self, "threshold_canvas"):
                    self.render_canvas_content(self.threshold_canvas, self.gray_image)

            # Image preparation and Auto T own separate, sequential processing-ui
            # contexts. There is no nested processing_ui() state or bypass flag.
            if settings.threshold is None:
                self.auto_select_threshold()
            else:
                self.threshold.set(settings.threshold)
                self.refresh_preview(changed_setting="image load")
        finally:
            self.update_navigation_state()


    def previous_image(self):
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_image(self):
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    @contextmanager
    def processing_ui(self):
        """Keep interactive controls disabled during synchronous processing."""
        control_types = (tk.Button, tk.Scale, tk.Checkbutton, tk.Radiobutton)
        prior_states = []
        pending = [self.root]
        while pending:
            parent = pending.pop()
            for child in parent.winfo_children():
                pending.append(child)
                if isinstance(child, control_types):
                    prior_states.append((child, child.cget("state")))
                    child.config(state=tk.DISABLED)

        # Only flush Tk's pending geometry/repaint work; processing stays synchronous.
        self.root.update_idletasks()
        try:
            yield
        finally:
            # Consume mouse/keyboard events queued during synchronous processing while
            # every interactive control is still disabled. This prevents a click or
            # drag made during processing from being replayed after controls are restored.
            self.root.update()
            for widget, state in prior_states:
                if widget.winfo_exists():
                    widget.config(state=state)

    def update_navigation_state(self):
        count = len(self.image_paths)
        has_current = 0 <= self.current_index < count
        readable = (
            has_current
            and self.gray_image is not None
            and self.master_image_payload is not None
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
        self.threshold_canvas.bind("<Configure>", self._handle_canvas_resize)
        self.color_canvas.bind("<Configure>", self._handle_canvas_resize)

    # ------------------------------------------------------------------
    # Application actions and threshold preview
    # ------------------------------------------------------------------
    def save_centered_images(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the threshold-finder stage."
        )

    def auto_select_threshold(self):
        """Run or reuse image-only Auto T and stop after the Stage-B refined display."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        if (
            isinstance(state.get("auto_threshold_result"), AutoThresholdResult)
            and state["auto_threshold_result"].failure_reason is not None
        ):
            self.status.set(
                "Automatic threshold previously failed "
                f"({state['auto_threshold_result'].failure_reason})."
            )
            return

        with self.processing_ui():
            if not isinstance(state.get("auto_threshold_result"), AutoThresholdResult):
                state["auto_threshold_result"] = AutoThresholdResult()
            elif (
                state["auto_threshold_result"].full_res_refined_threshold is None
                or state["auto_threshold_result"].full_res_refined_component_mask is None
            ):
                # Incomplete, non-failed runs are not authoritative for a new attempt.
                state["auto_threshold_result"] = AutoThresholdResult()

            if state["auto_threshold_result"].full_res_refined_threshold is None:
                try:
                    find_separation_threshold(
                        self.gray_image,
                        state["auto_threshold_result"],
                    )
                    if (
                        state["auto_threshold_result"].full_res_separation_component_mask
                        is not None
                        and hasattr(self, "threshold_canvas")
                    ):
                        self.render_canvas_content(
                            self.threshold_canvas,
                            decompress_array(
                                state[
                                    "auto_threshold_result"
                                ].full_res_separation_component_mask
                            ),
                        )

                    selected_threshold = refine_threshold(
                        self.gray_image,
                        state["auto_threshold_result"],
                    )
                except ThresholdResolutionError as exc:
                    self.status.set(f"Automatic threshold could not be resolved ({exc}).")
                    return
            else:
                selected_threshold = (
                    state["auto_threshold_result"].full_res_refined_threshold
                )

            # The setting commit invalidates stale downstream/display state. Publish the
            # newly valid Stage-B display only after that state transition completes.
            self.commit_setting_change("threshold", selected_threshold)

            if (
                state["auto_threshold_result"].full_res_refined_component_mask is not None
                and hasattr(self, "threshold_canvas")
            ):
                self.render_canvas_content(
                    self.threshold_canvas,
                    decompress_array(
                        state[
                            "auto_threshold_result"
                        ].full_res_refined_component_mask
                    ),
                )

            self.status.set(
                "Automatic grayscale threshold selected: "
                f"T={selected_threshold} "
                f"(work-res separation T="
                f"{state['auto_threshold_result'].work_res_separation_threshold}, "
                f"histogram start="
                f"{state['auto_threshold_result'].histogram_start_threshold}); "
                "Stage-B refined component displayed."
            )


    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the threshold-finder stage."
        )




    def commit_setting_change(self, setting_name, value):
        """Synchronize and persist one completed setting change, invalidating stale display."""
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

        if hasattr(self, "threshold_canvas"):
            self.render_canvas_content(self.threshold_canvas, None)


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

    def refresh_preview(
        self,
        changed_setting: str | None = None,
        full_resolution: bool = False,
    ):
        """Run the current explicit preview-processing cascade from raw threshold onward."""
        if self.gray_image is None or self.current_path is None:
            action = "Apply Full Resolution" if full_resolution else "Refresh Preview"
            self.status.set(f"{action}: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        if settings.threshold is None:
            self.status.set("Threshold is not initialized for the current image.")
            return

        with self.processing_ui():
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


    def _handle_canvas_resize(self, event):
        """Refit only the resized canvas from its retained unscaled display raster."""
        canvas = event.widget
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
            if unscaled_render_raster.dtype == bool and unscaled_render_raster.ndim == 2:
                unscaled_render_raster = cv2.cvtColor(
                    np.where(unscaled_render_raster, 255, 0).astype(np.uint8),
                    cv2.COLOR_GRAY2BGRA,
                )
            elif (
                unscaled_render_raster.dtype == np.uint8
                and unscaled_render_raster.ndim == 2
            ):
                unscaled_render_raster = cv2.cvtColor(
                    unscaled_render_raster,
                    cv2.COLOR_GRAY2BGRA,
                )
            elif not (
                unscaled_render_raster.dtype == np.uint8
                and unscaled_render_raster.ndim == 3
                and unscaled_render_raster.shape[2] == 4
            ):
                raise ValueError(
                    "canvas content must be a 2D bool mask, 2D uint8 grayscale, "
                    "or uint8 BGRA image"
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

        # Flush pending repaint work before synchronous processing continues.
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
