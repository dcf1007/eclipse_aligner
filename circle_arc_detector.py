"""Eclipse alignment GUI with grayscale automatic threshold selection.

This module combines the application's user-interface foundation with the tested
image-only automatic threshold stages. The GUI owns image navigation, per-image
processing settings, control interaction, explicit processing-stage orchestration,
and cached automatic-threshold/SolarData outcomes. Color-to-grayscale conversion is
an input-stage responsibility and is performed before the threshold algorithm is called.

All processing controls are per-image. ``DetectorApp.image_state`` is keyed by the
absolute image path. Non-threshold settings are sparse overrides relative to the
application defaults. ``settings.threshold`` is different: ``None`` means T has never
been initialized, and once initialized its exact integer value is always stored. A new
image uses a stored T when present, otherwise the Auto-T winner, and falls back to the
application default T only when Auto T fails.

Slider labels update continuously, but a setting is applied only after mouse release
or the final keyboard key release. Checkboxes and radio buttons apply immediately.
Every completed setting change passes through
``apply_changed_setting(setting_name, value)``. A genuinely changed T first restores
authoritative grayscale, then resolves the selected-T SolarData and displays its solar
component when successful. Non-T changes do not repaint an already-identical plain
solar mask, but they replace stale downstream overlays with that authoritative mask.

Automatic threshold selection is orchestrated by ``find_auto_threshold()``, which
owns the current image's single ``AutoThresholdResult`` and runs the two tested
algorithmic stages as needed. Stage A, ``find_separation_threshold()``, fills the
work-resolution and full-resolution separation fields; Stage B,
``refine_threshold()``, fills the refined threshold, component, and contour fields.
Neither stage creates or replaces the result object. The Auto Select button commits
the winning T through the same setting-change path used by manual GUI interaction.

Selected-T resolution consumes the authoritative Auto-T identity instead of
independently re-identifying the Sun. ``resolve_threshold()`` owns SolarData reuse,
invalidation, construction, and failure publication. ``solar_data is None`` means
resolution has never been attempted for the selected T; a SolarData with
``failure_reason`` records an expected failed attempt; a complete SolarData contains
the authoritative geometry. Impossible partial or inconsistent states raise explicit
errors. Preview/full-resolution downstream actions may construct genuinely missing
SolarData, but never retry an already-recorded failed outcome.

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
CANVAS_RESIZE_SETTLE_MS = 100


# ---------------------------------------------------------------------------
# Generic image and kernel utilities
# ---------------------------------------------------------------------------
def bool_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert one two-dimensional bool mask to uint8 values 0/255."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != bool:
        raise ValueError("mask must be a two-dimensional bool array")
    mask_u8 = mask.astype(np.uint8)
    mask_u8 *= 255
    return mask_u8


def generate_kernel(
    shape: tuple[int, int],
    round_kernel: bool = False,
) -> np.ndarray:
    """Return a positive-odd rectangle or centered discrete L2 ellipse."""
    height, width = shape
    if height <= 0 or width <= 0 or height % 2 == 0 or width % 2 == 0:
        raise ValueError("kernel height and width must be positive odd integers")

    if not round_kernel:
        return np.ones((height, width), dtype=np.uint8)

    # A one-pixel axis is the exact degenerate ellipse: a straight filled line.
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=np.uint8)

    y_radius = height // 2
    x_radius = width // 2
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
        cleaned = (
            bool_mask_to_uint8(source)
            if source.dtype == bool
            else cv2.compare(source, 0, cv2.CMP_NE)
        )

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
    shape: tuple[int, int],
    mask: bool = False,
) -> np.ndarray:
    """Resize to an explicit NumPy ``(height, width)`` shape."""
    original_dtype = img.dtype
    original_height, original_width = img.shape[:2]
    height, width = shape

    if height <= 0 or width <= 0:
        raise ValueError("resize dimensions must be positive")
    if (height, width) == (original_height, original_width):
        return img.copy()

    if mask:
        # OpenCV cannot resize bool directly; preserve mask membership with exact nearest.
        resize_source = bool_mask_to_uint8(img) if img.dtype == bool else img
        interpolation = cv2.INTER_NEAREST_EXACT
    elif height < original_height or width < original_width:
        resize_source = img
        interpolation = cv2.INTER_AREA
    else:
        resize_source = img
        interpolation = cv2.INTER_LANCZOS4

    resized = cv2.resize(
        resize_source,
        (width, height),  # OpenCV alone uses (width, height).
        interpolation=interpolation,
    )
    return resized.astype(original_dtype, copy=False)


def compress_image(image: np.ndarray) -> bytes:
    """Compress one supported 1-, 8-, or 16-bit internal image or mask."""
    image = np.asarray(image)
    if image.ndim == 2:
        height, width = image.shape
        channels = 0
    elif image.ndim == 3:
        height, width, channels = image.shape
    else:
        raise ValueError("image must be two- or three-dimensional")

    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if channels not in (0, 1, 2, 3, 4):
        raise ValueError("image channel count must be 0, 1, 2, 3, or 4")

    samples_per_pixel = 1 if channels == 0 else channels

    if image.dtype == bool:
        if channels not in (0, 1):
            raise ValueError(
                "1-bit images must have no explicit channel axis or exactly one channel"
            )
        bit_depth = 1
        encoded_pixels = np.packbits(image.reshape(-1))
    elif image.dtype == np.uint8:
        bit_depth = 8
        encoded_pixels = np.ascontiguousarray(image)
    elif image.dtype == np.uint16:
        bit_depth = 16
        # Persist uint16 samples in an explicit byte order rather than native endian.
        encoded_pixels = np.ascontiguousarray(
            image.astype(np.dtype("<u2"), copy=False)
        )
    else:
        raise ValueError("image dtype must be bool, uint8, or uint16")

    expected_samples = height * width * samples_per_pixel
    if image.size != expected_samples:
        raise ValueError("image shape does not match its declared channel structure")

    header = struct.pack("<IIBB", height, width, channels, bit_depth)

    # Give zlib a byte view of the existing contiguous pixel storage instead of
    # image.tobytes(), which would allocate another complete copy of the image.
    encoded_pixel_bytes = memoryview(encoded_pixels).cast("B")

    # Compress the header and pixels as one continuous zlib stream without first
    # constructing header + encoded_pixels, which would make another full-size
    # uncompressed copy.
    compressor = zlib.compressobj(level=1)
    return b"".join(
        (
            compressor.compress(header),
            compressor.compress(encoded_pixel_bytes),
            compressor.flush(),
        )
    )


def decompress_image(payload: bytes) -> np.ndarray:
    """Restore one image or mask produced by :func:`compress_image`."""
    raw = zlib.decompress(payload)
    header_size = struct.calcsize("<IIBB")
    if len(raw) < header_size:
        raise ValueError("compressed image payload is truncated")

    height, width, channels, bit_depth = struct.unpack(
        "<IIBB", raw[:header_size]
    )
    if height <= 0 or width <= 0:
        raise ValueError("compressed image dimensions must be positive")
    if channels not in (0, 1, 2, 3, 4):
        raise ValueError(
            f"compressed image has unsupported channel count: {channels}"
        )

    encoded_pixel_byte_count = len(raw) - header_size
    samples_per_pixel = 1 if channels == 0 else channels
    sample_count = height * width * samples_per_pixel
    shape = (height, width) if channels == 0 else (height, width, channels)

    if bit_depth == 1:
        if channels not in (0, 1):
            raise ValueError(
                "compressed 1-bit image must have no explicit channel axis "
                "or exactly one channel"
            )
        expected_bytes = (sample_count + 7) // 8
        if encoded_pixel_byte_count != expected_bytes:
            raise ValueError(
                f"compressed 1-bit image has {encoded_pixel_byte_count} bytes; "
                f"expected {expected_bytes}"
            )

        # Read the packed pixels directly from the decompressed payload after its
        # header. Using an offset avoids copying the complete pixel payload.
        packed = np.frombuffer(
            raw,
            dtype=np.uint8,
            count=expected_bytes,
            offset=header_size,
        )
        return np.unpackbits(packed, count=sample_count).reshape(shape) != 0

    if bit_depth == 8:
        bytes_per_sample = 1
        dtype = np.uint8
    elif bit_depth == 16:
        bytes_per_sample = 2
        dtype = np.dtype("<u2")
    else:
        raise ValueError(f"unsupported compressed image bit depth: {bit_depth}")

    expected_bytes = sample_count * bytes_per_sample
    if encoded_pixel_byte_count != expected_bytes:
        raise ValueError(
            f"compressed image pixel payload has {encoded_pixel_byte_count} bytes; "
            f"expected {expected_bytes}"
        )

    # Interpret the pixel portion of raw directly instead of creating
    # raw[header_size:], which would duplicate the complete decompressed image.
    restored = np.frombuffer(
        raw,
        dtype=dtype,
        count=sample_count,
        offset=header_size,
    ).reshape(shape)
    return restored.astype(np.uint16, copy=False) if bit_depth == 16 else restored


def compress_contour(contour: np.ndarray) -> bytes:
    """Compress one ordered ``(N, 2)`` int32 XY contour."""
    contour = np.asarray(contour)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must be an (N, 2) XY array")
    if len(contour) == 0:
        raise ValueError("contour must contain at least one point")
    if contour.dtype != np.int32:
        raise ValueError("contour dtype must be int32")

    encoded_points = np.ascontiguousarray(
        contour.astype(np.dtype("<i4"), copy=False)
    ).tobytes()
    return zlib.compress(encoded_points, level=1)


def decompress_contour(payload: bytes) -> np.ndarray:
    """Restore one ordered ``(N, 2)`` int32 XY contour."""
    raw = zlib.decompress(payload)
    bytes_per_point = 2 * np.dtype("<i4").itemsize
    if len(raw) == 0:
        raise ValueError("compressed contour is empty")
    if len(raw) % bytes_per_point != 0:
        raise ValueError(
            "compressed contour payload does not contain complete int32 XY points"
        )

    contour = np.frombuffer(raw, dtype=np.dtype("<i4")).reshape(-1, 2)
    return contour.astype(np.int32, copy=False)


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


def calculate_work_res_shape(full_res_shape: tuple[int, int]) -> tuple[int, int]:
    """Return the aspect-preserving work raster shape in NumPy ``(height, width)`` order."""
    full_res_height, full_res_width = full_res_shape
    if full_res_height <= 0 or full_res_width <= 0:
        raise ValueError("full-resolution shape dimensions must be positive")

    full_res_max_dim = max(full_res_shape)
    if full_res_max_dim <= WORK_RES_MAX_DIM:
        return full_res_shape

    work_res_scale = WORK_RES_MAX_DIM / full_res_max_dim
    return (
        round(full_res_height * work_res_scale),
        round(full_res_width * work_res_scale),
    )


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
    full_res_refined_component_contour: bytes | None = None

    failure_reason: str | None = None

    @property
    def separation_threshold_complete(self) -> bool:
        """Return whether every result owned by ``find_separation_threshold`` exists."""
        return all(
            value is not None
            for value in (
                self.histogram_start_threshold,
                self.work_res_seed_point,
                self.work_res_separation_threshold,
                self.work_res_separation_component_mask,
                self.full_res_seed_point,
                self.full_res_separation_threshold,
                self.full_res_separation_component_mask,
                self.full_res_separation_guard_mask,
            )
        )

    @property
    def threshold_refinement_complete(self) -> bool:
        """Return whether every result owned by ``refine_threshold`` exists."""
        return all(
            value is not None
            for value in (
                self.full_res_refined_threshold,
                self.full_res_refined_component_mask,
                self.full_res_refined_component_contour,
            )
        )


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
    return rightmost_peak


def brightest_supported_component_point(
    gray: np.ndarray,
    component: np.ndarray,
    support_kernel: np.ndarray,
) -> tuple[int, int] | None:
    """Return the brightest support-eligible component pixel; depth breaks ties.

    The caller owns support geometry and the meaning of an unavailable point. Empty
    or unsupported components return ``None``; malformed caller inputs remain errors.
    """
    component = np.asarray(component)
    source = (
        bool_mask_to_uint8(component)
        if component.dtype == bool
        else cv2.compare(component, 0, cv2.CMP_NE)
    )
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
    binary = np.asarray(binary)
    binary_u8 = (
        bool_mask_to_uint8(binary)
        if binary.dtype == bool
        else cv2.compare(binary, 0, cv2.CMP_NE)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_u8,
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

    flood = (
        bool_mask_to_uint8(binary_mask)
        if binary_mask.dtype == bool
        else cv2.compare(binary_mask, 0, cv2.CMP_NE)
    )
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
                raise ValueError(
                    "tracked work-resolution seed component disappeared while lowering T"
                )
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
        raise ValueError("cannot dilate empty solar component")
    if margin < 0:
        raise ValueError("dilation margin must be non-negative")

    # distanceTransform measures each non-component pixel's L2 distance to the
    # nearest zero pixel, so encode the component itself as zero and threshold the
    # resulting full-frame distance field at the requested dilation margin.
    outside = bool_mask_to_uint8(component)
    cv2.bitwise_not(outside, dst=outside)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    return distance <= margin


def find_guard_boundary(guard_mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel inner boundary of a non-empty boolean guard mask."""
    if guard_mask.ndim != 2 or not np.any(guard_mask):
        raise ValueError("guard must be a non-empty two-dimensional mask")

    guard_u8 = (
        bool_mask_to_uint8(guard_mask)
        if guard_mask.dtype == bool
        else cv2.compare(guard_mask, 0, cv2.CMP_NE)
    )
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
        raise ValueError("full-resolution Auto-T guard is empty")

    seed_x, seed_y = full_res_seed_point
    full_res_height, full_res_width = full_res_gray.shape
    if not (0 <= seed_x < full_res_width and 0 <= seed_y < full_res_height):
        raise ValueError("full-resolution tracking seed lies outside the image")
    if not full_res_guard_mask[seed_y, seed_x]:
        raise ValueError("full-resolution tracking seed lies outside the Auto-T guard")

    full_res_guard_boundary = find_guard_boundary(full_res_guard_mask)
    full_res_guard_u8 = bool_mask_to_uint8(full_res_guard_mask)
    full_res_guard_boundary_indices = np.flatnonzero(full_res_guard_boundary)
    del full_res_guard_boundary

    # Evaluate the starting T after the fixed D7 cleanup used by coarse separation.
    binary = morphological_cleanup(full_res_gray, SEPARATION_KERNEL, start_T)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed does not survive D7 cleanup at start T={start_T}"
        )
    cv2.bitwise_and(binary, full_res_guard_u8, dst=binary)
    component = extract_component(binary, full_res_seed_point)
    del binary
    if component is None:
        raise ValueError(
            "full-resolution tracking seed disappeared after clipping to its containing guard"
        )

    if not np.any(component.ravel()[full_res_guard_boundary_indices]):
        best_T = start_T
        best_component = component
        for threshold in range(start_T - 1, -1, -1):
            binary = morphological_cleanup(
                full_res_gray,
                SEPARATION_KERNEL,
                threshold,
            )
            cv2.bitwise_and(binary, full_res_guard_u8, dst=binary)
            component = extract_component(binary, full_res_seed_point)
            del binary
            if component is None:
                raise ValueError(
                    "tracked full-resolution seed component disappeared while lowering T"
                )
            if np.any(component.ravel()[full_res_guard_boundary_indices]):
                break
            best_T = threshold
            best_component = component
        return best_T, best_component

    for threshold in range(start_T + 1, 256):
        binary = morphological_cleanup(full_res_gray, SEPARATION_KERNEL, threshold)
        if binary[seed_y, seed_x] == 0:
            break
        cv2.bitwise_and(binary, full_res_guard_u8, dst=binary)
        component = extract_component(binary, full_res_seed_point)
        del binary
        if component is None:
            raise ValueError(
                "full-resolution tracking seed disappeared after surviving D7 cleanup"
            )
        if not np.any(component.ravel()[full_res_guard_boundary_indices]):
            return threshold, component

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became separated after D7 cleanup"
    )



def derive_full_res_seed_and_guard(
    full_res_gray: np.ndarray,
    work_res_component: np.ndarray,
    work_res_seed_kernel: np.ndarray,
) -> tuple[tuple[int, int], np.ndarray]:
    """Transfer mature work geometry and return its supported full-res seed and fixed guard."""
    full_res_search_mask = resize_img(
        work_res_component,
        full_res_gray.shape,
        mask=True,
    )

    work_res_support_size = max(work_res_seed_kernel.shape)
    mapped_kernel_size = (
        work_res_support_size
        * max(full_res_gray.shape)
        / max(work_res_component.shape)
    )
    if mapped_kernel_size <= 0:
        raise ValueError("mapped full-resolution support size must be positive")
    full_res_kernel_size = 2 * math.ceil(mapped_kernel_size / 2) - 1
    full_res_seed_kernel = generate_kernel(
        (full_res_kernel_size, full_res_kernel_size),
        round_kernel=False,
    )

    full_res_seed_point = brightest_supported_component_point(
        full_res_gray,
        full_res_search_mask,
        full_res_seed_kernel,
    )
    if full_res_seed_point is None:
        raise ThresholdResolutionError(
            f"Transferred solar component has no {full_res_kernel_size}x"
            f"{full_res_kernel_size}-supported full-resolution seed"
        )

    image_scale = math.sqrt(full_res_gray.shape[0] * full_res_gray.shape[1])
    full_res_guard_mask = dilate_component_mask(
        full_res_search_mask,
        AUTO_T_GUARD_DILATION_FRACTION * image_scale,
    )
    return full_res_seed_point, full_res_guard_mask


def find_auto_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int | None:
    """Run or reuse the complete Auto-T operation and return its winning T."""
    if full_res_gray.ndim != 2 or full_res_gray.dtype != np.uint8:
        raise ValueError("automatic thresholding requires authoritative 2D uint8 grayscale")
    if not isinstance(image_state, dict):
        raise ValueError("automatic thresholding requires the current image state")

    auto_threshold_result = image_state.get("auto_threshold_result")
    if auto_threshold_result is None:
        auto_threshold_result = AutoThresholdResult()
        image_state["auto_threshold_result"] = auto_threshold_result
    elif not isinstance(auto_threshold_result, AutoThresholdResult):
        raise ValueError("stored Auto-T state must be an AutoThresholdResult or None")

    if auto_threshold_result.failure_reason is not None:
        return None

    if not auto_threshold_result.separation_threshold_complete:
        find_separation_threshold(full_res_gray, image_state)
        auto_threshold_result = image_state.get("auto_threshold_result")
        if not isinstance(auto_threshold_result, AutoThresholdResult):
            raise ValueError("Stage A returned without preserving AutoThresholdResult")

    if auto_threshold_result.failure_reason is not None:
        return None
    if not auto_threshold_result.separation_threshold_complete:
        raise ValueError(
            "separation threshold remained incomplete without a recorded failure"
        )

    if auto_threshold_result.threshold_refinement_complete:
        return auto_threshold_result.full_res_refined_threshold

    refine_threshold(full_res_gray, image_state)
    auto_threshold_result = image_state.get("auto_threshold_result")
    if not isinstance(auto_threshold_result, AutoThresholdResult):
        raise ValueError("Stage B returned without preserving AutoThresholdResult")

    if auto_threshold_result.failure_reason is not None:
        return None
    if not auto_threshold_result.threshold_refinement_complete:
        raise ValueError(
            "threshold refinement remained incomplete without a recorded failure"
        )
    return auto_threshold_result.full_res_refined_threshold


def find_separation_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> None:
    """Run Auto-T Stage A on the result established by :func:`find_auto_threshold`."""
    auto_threshold_result = image_state["auto_threshold_result"]

    # An incomplete non-failed separation is a recoverable interrupted attempt.
    # Restart Auto-T from Stage A without allowing stale downstream values to mix
    # with the new progressive result.
    auto_threshold_result.histogram_start_threshold = None
    auto_threshold_result.work_res_seed_point = None
    auto_threshold_result.work_res_separation_threshold = None
    auto_threshold_result.work_res_separation_component_mask = None
    auto_threshold_result.full_res_seed_point = None
    auto_threshold_result.full_res_separation_threshold = None
    auto_threshold_result.full_res_separation_component_mask = None
    auto_threshold_result.full_res_separation_guard_mask = None
    auto_threshold_result.full_res_refined_threshold = None
    auto_threshold_result.full_res_refined_component_mask = None
    auto_threshold_result.full_res_refined_component_contour = None

    work_res_shape = calculate_work_res_shape(full_res_gray.shape)
    work_res_gray = resize_img(full_res_gray, work_res_shape)
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
        auto_threshold_result.work_res_separation_component_mask = compress_image(
            work_res_component
        )

        resolution_step = "full-resolution seed selection"
        (
            auto_threshold_result.full_res_seed_point,
            full_res_guard_mask,
        ) = derive_full_res_seed_and_guard(
            full_res_gray,
            work_res_component,
            work_res_seed_kernel,
        )

        resolution_step = "full-resolution guard construction"
        auto_threshold_result.full_res_separation_guard_mask = compress_image(
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
        auto_threshold_result.full_res_separation_component_mask = compress_image(
            full_res_separation_component
        )
    except ThresholdResolutionError as exc:
        auto_threshold_result.failure_reason = f"{resolution_step}: {exc}"
        return


def find_external_contour(component: np.ndarray) -> np.ndarray:
    """Return the ordered largest external contour as an ``(N, 2)`` int32 XY array."""
    if component.ndim != 2 or not np.any(component):
        raise ValueError("solar component is empty or not two-dimensional")
    component_u8 = (
        bool_mask_to_uint8(component)
        if component.dtype == bool
        else cv2.compare(component, 0, cv2.CMP_NE)
    )
    contours, _ = cv2.findContours(
        component_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("solar component has no external contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if contour.size == 0:
        raise ValueError("solar component external contour is empty")
    return np.ascontiguousarray(contour, dtype=np.int32)



def measure_filled_area(contour: np.ndarray) -> int:
    """Return raster-equivalent area enclosed by the external lattice contour."""
    points = contour.astype(np.int64)
    following = np.roll(points, -1, axis=0)
    distance = np.abs(following - points)
    boundary_points = int(np.gcd(distance[:, 0], distance[:, 1]).sum())
    polygon_area = float(cv2.contourArea(contour))
    filled_area = int(round(polygon_area + 0.5 * boundary_points + 1.0))
    if filled_area <= 0:
        raise ValueError("filled external contour area is empty")
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
        raise ValueError("external contour perimeter is empty")
    hole_area = filled_area - component_area
    minimum_hole_perimeter = 2.0 * math.sqrt(math.pi * hole_area)
    return external_perimeter / (external_perimeter + minimum_hole_perimeter)


def sample_grayscale_profiles(
    full_res_gray_float: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one averaged outward grayscale profile per raster-scale polygon side."""
    if full_res_gray_float.ndim != 2 or full_res_gray_float.dtype != np.float32:
        raise ValueError("edge-profile grayscale must be a two-dimensional float32 array")
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
    polygon = cv2.approxPolyDP(contour, raster_tolerance, True).reshape(-1, 2)
    points = polygon.astype(np.float64)
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
    height, width = full_res_gray_float.shape
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
    full_res_gray_float: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, float]:
    """Return photometric edge distance and image-local profile reliability."""
    profiles, segment_lengths = sample_grayscale_profiles(full_res_gray_float, contour)
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
    if not math.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("sampled grayscale profile length must be finite and positive")
    edge_reliability = float(
        np.dot(segment_lengths, profile_reliability) / total_length
    )
    return edge_distance, edge_reliability


def extract_separated_seed_component(
    threshold_mask: np.ndarray,
    seed_point: tuple[int, int],
    guard_u8: np.ndarray,
    guard_boundary_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the P3/P5/P7-cleaned seeded component and contour when separated."""
    cleaned = threshold_mask
    for kernel in SOLAR_CLEANUP_KERNELS:
        cleaned = morphological_cleanup(cleaned, kernel)
    cv2.bitwise_and(cleaned, guard_u8, dst=cleaned)
    component = extract_component(cleaned, seed_point)
    if component is None or np.any(component.ravel()[guard_boundary_indices]):
        return None
    return component, find_external_contour(component)


def refine_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int | None:
    """Run Auto-T Stage B on the completed Stage-A result."""
    auto_threshold_result = image_state["auto_threshold_result"]

    auto_threshold_result.full_res_refined_threshold = None
    auto_threshold_result.full_res_refined_component_mask = None
    auto_threshold_result.full_res_refined_component_contour = None

    base_threshold = auto_threshold_result.full_res_separation_threshold
    full_res_seed_point = auto_threshold_result.full_res_seed_point
    full_res_guard_mask = decompress_image(
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
        full_res_guard_u8 = bool_mask_to_uint8(full_res_guard_mask)
        full_res_guard_boundary_indices = np.flatnonzero(full_res_guard_boundary)
        del full_res_guard_boundary
        del full_res_guard_mask
        full_res_gray_float = full_res_gray.astype(np.float32)
        measurements: list[ThresholdMeasurement] = []
        compressed_masks: dict[int, bytes] = {}
        candidate_contours: dict[int, np.ndarray] = {}
        raw_reference_area: int | None = None
        raw_reference_roughness: float | None = None

        max_threshold = min(255, base_threshold + MAX_T_REFINEMENT_STEPS)
        for threshold in range(base_threshold, max_threshold + 1):
            threshold_mask = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)

            candidate = extract_separated_seed_component(
                threshold_mask,
                full_res_seed_point,
                full_res_guard_u8,
                full_res_guard_boundary_indices,
            )

            if candidate is not None:
                cleaned_component, contour = candidate
                filled_area = measure_filled_area(contour)
                roughness = measure_roughness(contour, filled_area)
                component_area = int(np.count_nonzero(cleaned_component))
                hole_quality = measure_hole_quality(
                    contour,
                    component_area,
                    filled_area,
                )
                edge_distance, edge_reliability = measure_edge_alignment(
                    full_res_gray_float,
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
                compressed_masks[threshold] = compress_image(cleaned_component)
                candidate_contours[threshold] = contour
                del cleaned_component

            # The candidate tuple can otherwise keep a full-resolution cleaned mask
            # alive while the raw reference for this threshold is measured.
            del candidate

            # Raw geometry only anchors the largest/roughest end of the score scale.
            # Measure the first separated raw component once, then stop evaluating raw.
            if raw_reference_area is None:
                raw_mask = threshold_mask.copy()
                cv2.bitwise_and(raw_mask, full_res_guard_u8, dst=raw_mask)
                raw_component = extract_component(raw_mask, full_res_seed_point)
                del raw_mask
                if (
                    raw_component is not None
                    and not np.any(
                        raw_component.ravel()[full_res_guard_boundary_indices]
                    )
                ):
                    raw_contour = find_external_contour(raw_component)
                    raw_reference_area = measure_filled_area(raw_contour)
                    raw_reference_roughness = measure_roughness(
                        raw_contour,
                        raw_reference_area,
                    )
                    del raw_contour
                del raw_component

            # This threshold mask has served both cleaned-candidate and raw-reference
            # measurement and should not overlap the next full-resolution threshold.
            del threshold_mask

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
            raise ValueError("invalid within-image score scale")

        edge_reliability = float(
            np.median([measurement.edge_reliability for measurement in measurements])
        )
        if not math.isfinite(edge_reliability):
            raise ValueError("median edge reliability must be finite")

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
            raise ValueError("threshold refinement produced no score winner")

        winning_contour = candidate_contours[best_threshold]

        auto_threshold_result.full_res_refined_threshold = best_threshold
        auto_threshold_result.full_res_refined_component_mask = compressed_masks[
            best_threshold
        ]
        auto_threshold_result.full_res_refined_component_contour = compress_contour(
            winning_contour
        )
        return best_threshold
    except ThresholdResolutionError as exc:
        auto_threshold_result.failure_reason = f"fine refinement: {exc}"
        return None


# ---------------------------------------------------------------------------
# Final-T full-resolution solar resolution and persistence
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolarData:
    """Outcome of solar-geometry resolution at exactly one selected threshold T."""

    threshold: int
    seed_point: tuple[int, int] | None = None
    component_mask: bytes | None = None
    guard_mask: bytes | None = None
    component_contour: bytes | None = None
    failure_reason: str | None = None

    @property
    def complete(self) -> bool:
        """Return whether this object contains the complete authoritative geometry."""
        return (
            self.failure_reason is None
            and self.seed_point is not None
            and self.component_mask is not None
            and self.guard_mask is not None
            and self.component_contour is not None
        )


def resolve_threshold(
    full_res_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Resolve selected T and atomically publish complete or failed SolarData."""
    if full_res_gray.ndim != 2 or full_res_gray.dtype != np.uint8:
        raise ValueError("threshold resolution requires authoritative 2D uint8 grayscale")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be 0..255")
    if not isinstance(image_state, dict):
        raise ValueError("threshold resolution requires the current image state")

    existing = image_state.get("solar_data")
    if existing is not None and not isinstance(existing, SolarData):
        raise ValueError("stored solar data must be SolarData or None")
    if isinstance(existing, SolarData):
        if existing.threshold != threshold:
            image_state["solar_data"] = None
        elif existing.failure_reason is not None:
            if any(
                value is not None
                for value in (
                    existing.seed_point,
                    existing.component_mask,
                    existing.guard_mask,
                    existing.component_contour,
                )
            ):
                raise ValueError("failed same-T SolarData must not contain geometry")
            raise ThresholdResolutionError(existing.failure_reason)
        else:
            if not existing.complete:
                raise ValueError("stored same-T SolarData is incomplete without a failure")
            try:
                component = decompress_image(existing.component_mask)
                guard_mask = decompress_image(existing.guard_mask)
                contour = decompress_contour(existing.component_contour)
            except (ValueError, zlib.error) as exc:
                raise ValueError("stored same-T SolarData payload is corrupt") from exc

            seed_x, seed_y = existing.seed_point
            height, width = full_res_gray.shape
            if (
                component.dtype != bool
                or component.shape != full_res_gray.shape
                or not np.any(component)
                or guard_mask.dtype != bool
                or guard_mask.shape != full_res_gray.shape
                or not np.any(guard_mask)
                or not (0 <= seed_x < width and 0 <= seed_y < height)
                or not component[seed_y, seed_x]
                or not guard_mask[seed_y, seed_x]
                or np.any(component & ~guard_mask)
                or full_res_gray[seed_y, seed_x] <= threshold
                or contour.dtype != np.int32
                or contour.ndim != 2
                or contour.shape[1] != 2
                or len(contour) == 0
            ):
                raise ValueError("stored same-T SolarData is inconsistent")
            return component

    try:
        find_auto_threshold(full_res_gray, image_state)
        auto_threshold_result = image_state.get("auto_threshold_result")
        if not isinstance(auto_threshold_result, AutoThresholdResult):
            raise ValueError("Auto-T returned without preserving AutoThresholdResult")

        if (
            auto_threshold_result.failure_reason is None
            and auto_threshold_result.threshold_refinement_complete
            and threshold == auto_threshold_result.full_res_refined_threshold
        ):
            component = decompress_image(
                auto_threshold_result.full_res_refined_component_mask
            )
            solar_data = SolarData(
                threshold=threshold,
                seed_point=auto_threshold_result.full_res_seed_point,
                component_mask=auto_threshold_result.full_res_refined_component_mask,
                guard_mask=auto_threshold_result.full_res_separation_guard_mask,
                component_contour=auto_threshold_result.full_res_refined_component_contour,
            )
            image_state["solar_data"] = solar_data
            return component

        full_res_seed_point = auto_threshold_result.full_res_seed_point
        guard_payload = auto_threshold_result.full_res_separation_guard_mask
        if (full_res_seed_point is None) != (guard_payload is None):
            raise ValueError("stored Auto-T full-resolution identity is inconsistent")

        if full_res_seed_point is not None:
            full_res_guard_mask = decompress_image(guard_payload)
            if (
                full_res_guard_mask.dtype != bool
                or full_res_guard_mask.shape != full_res_gray.shape
                or not np.any(full_res_guard_mask)
            ):
                raise ValueError(
                    "stored Auto-T guard must be a non-empty mask matching grayscale"
                )
            seed_x, seed_y = full_res_seed_point
            height, width = full_res_gray.shape
            if not (0 <= seed_x < width and 0 <= seed_y < height):
                raise ValueError(
                    "stored Auto-T full-resolution seed lies outside grayscale"
                )
            if not full_res_guard_mask[seed_y, seed_x]:
                raise ValueError(
                    "stored Auto-T full-resolution seed lies outside its guard"
                )
        else:
            work_res_seed_kernel = generate_kernel((5, 5), round_kernel=False)
            work_component_payload = (
                auto_threshold_result.work_res_separation_component_mask
            )
            if work_component_payload is not None:
                work_res_component = decompress_image(work_component_payload)
                if (
                    work_res_component.dtype != bool
                    or work_res_component.ndim != 2
                    or not np.any(work_res_component)
                ):
                    raise ValueError("stored Auto-T work component is invalid")
            else:
                work_res_shape = calculate_work_res_shape(full_res_gray.shape)
                work_res_gray = resize_img(full_res_gray, work_res_shape)
                work_res_component = largest_enclosed_bright_component(
                    work_res_gray > threshold
                )
                if work_res_component is None:
                    raise ThresholdResolutionError(
                        f"No enclosed work-resolution solar proposal exists at selected T={threshold}"
                    )
                work_res_seed = brightest_supported_component_point(
                    work_res_gray,
                    work_res_component,
                    work_res_seed_kernel,
                )
                if work_res_seed is None:
                    raise ThresholdResolutionError(
                        f"No 5x5-supported enclosed work-resolution solar proposal exists "
                        f"at selected T={threshold}"
                    )

            (
                full_res_seed_point,
                full_res_guard_mask,
            ) = derive_full_res_seed_and_guard(
                full_res_gray,
                work_res_component,
                work_res_seed_kernel,
            )

        full_res_guard_boundary = find_guard_boundary(full_res_guard_mask)
        full_res_guard_u8 = bool_mask_to_uint8(full_res_guard_mask)
        full_res_guard_boundary_indices = np.flatnonzero(full_res_guard_boundary)
        del full_res_guard_boundary
        threshold_mask = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
        candidate = extract_separated_seed_component(
            threshold_mask,
            full_res_seed_point,
            full_res_guard_u8,
            full_res_guard_boundary_indices,
        )
        del threshold_mask
        if candidate is None:
            raise ThresholdResolutionError(
                f"No separated cleaned solar component exists at selected T={threshold}"
            )
        component, contour = candidate
        del candidate

        solar_data = SolarData(
            threshold=threshold,
            seed_point=full_res_seed_point,
            component_mask=compress_image(component),
            guard_mask=compress_image(full_res_guard_mask),
            component_contour=compress_contour(contour),
        )
        image_state["solar_data"] = solar_data
        return component
    except ThresholdResolutionError as exc:
        image_state["solar_data"] = SolarData(
            threshold=threshold,
            failure_reason=str(exc),
        )
        raise


class DetectorApp:
    """Own GUI state, per-image settings, cached auto thresholds, and previews.

    Named control callbacks and application actions are public methods. Underscore-
    prefixed methods are internal GUI mechanics. Threshold acquisition and final-T
    resolution both write their agreed derived objects into the current per-image state;
    this class coordinates those operations with the processing settings shown by the GUI.
    """

    def __init__(self, root: tk.Tk, image_paths: list[str]):
        self.root = root
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path: str | None = None
        self.master_image_payload: bytes | None = None
        self.gray_image = None

        # Per-image state keeps sparse settings, the cached automatic threshold
        # result, and the attempted SolarData outcome for the selected threshold.
        self.image_state: dict[str, dict[str, object]] = {}

        # Keyboard auto-repeat can emit intermediate release/press pairs on some
        # Tk platforms. Keep one deferred commit job so only the final key-up
        # applies a slider-driven setting change.
        self.slider_keyboard_commit_job = None
        self.slider_keyboard_widget = None
        self.slider_keyboard_start_value = None

        self.threshold = tk.IntVar(value=8)
        self.min_radius = tk.IntVar(value=1000)
        self.max_radius = tk.IntVar(value=1500)
        self.max_error = tk.DoubleVar(value=8.0)
        self.min_coverage = tk.IntVar(value=8)
        self.morphology = tk.BooleanVar(value=False)
        self.outer_limb_assistance = tk.BooleanVar(value=False)
        self.use_horizon = tk.BooleanVar(value=True)

        # Mutually exclusive by construction: both Radiobuttons share this one
        # StringVar. Light is the requested default.
        self.center_target = tk.StringVar(value="light")
        self.center_preview_text = tk.StringVar()

        # Ordinary controls use these values as sparse baselines. Threshold is
        # different: once initialized, its exact current integer is always stored.
        self.default_settings = ImageSettings(
            threshold=self.threshold.get(),
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

        self.status = tk.StringVar(
            value="Threshold finder integrated. Load images to inspect automatic T selection."
        )
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector — threshold finder")
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)

        self._build_navigation_bar()
        self._build_settings_panel()
        self._build_preview_panes()
        self._update_center_preview_label()

        # Tk's toplevel bindtag receives mouse events from every child widget.
        # Use it to clear slider keyboard focus as soon as the user clicks
        # anywhere outside the currently focused slider.
        root.bind(
            "<ButtonPress-1>",
            self._release_slider_focus_if_clicked_elsewhere,
            add="+",
        )

        if self.image_paths:
            self.load_image_at(0)
        else:
            self._update_navigation_state()

    # ------------------------------------------------------------------
    # Image list / navigation (GUI support only)
    # ------------------------------------------------------------------
    def load_images_button_clicked(self):
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
        self._update_navigation_state()
        self.root.update_idletasks()

        try:
            with self.blocked_gui():
                # Load source pixels without changing their channel count or integer depth.
                unchanged_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                if unchanged_image is None:
                    self.master_image_payload = None
                    self.gray_image = None
                    if hasattr(self, "threshold_canvas"):
                        self.render_canvas_content(self.threshold_canvas, None)
                    if hasattr(self, "color_canvas"):
                        self.render_canvas_content(self.color_canvas, None)
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
                    master_image = master_image.astype(np.uint16)
                    master_image *= 257
                master_image = np.ascontiguousarray(master_image, dtype=np.uint16)

                # The unchanged source has now served its only purpose: constructing the
                # normalized lossless master. Drop those references before allocating any
                # additional full-resolution representations.
                del source
                del unchanged_image

                # Authoritative threshold processing is always uint8 grayscale derived
                # directly from the lossless master with the exact full-range mapping.
                gray16 = cv2.cvtColor(master_image, cv2.COLOR_BGRA2GRAY)
                self.gray_image = cv2.convertScaleAbs(
                    gray16,
                    alpha=1.0 / 257.0,
                )
                del gray16

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

                # Retain the exact master in the shared self-describing array format.
                self.master_image_payload = compress_image(master_image)
                del master_image

                # First visible processing stage: lossless source color + authoritative
                # grayscale. The color canvas retains the already-compressed master.
                if hasattr(self, "color_canvas"):
                    self.render_canvas_content(
                        self.color_canvas,
                        self.master_image_payload,
                    )
                if hasattr(self, "threshold_canvas"):
                    self.render_canvas_content(
                        self.threshold_canvas,
                        self.gray_image,
                    )

            # Image preparation and Auto T own separate, sequential blocked-GUI
            # contexts. Auto T establishes its result without selecting a GUI T yet.
            with self.blocked_gui():
                automatic_threshold = find_auto_threshold(self.gray_image, state)
                auto_threshold_result = state.get("auto_threshold_result")
                if not isinstance(auto_threshold_result, AutoThresholdResult):
                    raise ValueError(
                        "Auto-T returned without establishing AutoThresholdResult"
                    )

            # Restore every per-image setting through the same application path used
            # by completed GUI interaction. A stored threshold wins; otherwise Auto T
            # initializes T, with the application default used only if Auto T failed.
            for setting_name in self.setting_variables:
                if setting_name == "threshold":
                    value = settings.threshold
                    if value is None:
                        value = (
                            automatic_threshold
                            if automatic_threshold is not None
                            else self.default_settings.threshold
                        )
                else:
                    stored_value = getattr(settings, setting_name)
                    value = (
                        getattr(self.default_settings, setting_name)
                        if stored_value is None
                        else stored_value
                    )
                self.apply_changed_setting(setting_name, value)

            self._update_center_preview_label()

            # Load, Previous, and Next all execute the complete workflow. SolarData
            # is normally current because restoration ran through apply_changed_setting;
            # the preview action can construct it if it was genuinely never attempted.
            self.preview_button_clicked()
        finally:
            self._update_navigation_state()


    def previous_button_clicked(self):
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_button_clicked(self):
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    @contextmanager
    def blocked_gui(self):
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

    def _update_navigation_state(self):
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

        load_images_button = tk.Button(
            frame,
            text="Load images...",
            width=14,
            command=self.load_images_button_clicked,
        )
        load_images_button.grid(row=0, column=0, padx=(0, 8))

        save_centered_button = tk.Button(
            frame,
            text="Save centered images",
            width=20,
            command=self.save_centered_button_clicked,
        )
        save_centered_button.grid(row=0, column=1, padx=(0, 10))

        self.previous_button = tk.Button(
            frame,
            text="◀ Previous",
            width=12,
            command=self.previous_button_clicked,
        )
        self.previous_button.grid(row=0, column=2, padx=(0, 5))

        self.next_button = tk.Button(
            frame,
            text="Next ▶",
            width=12,
            command=self.next_button_clicked,
        )
        self.next_button.grid(row=0, column=3, padx=(0, 10))

        image_info_label = tk.Label(
            frame,
            textvariable=self.image_info,
            anchor="w",
        )
        image_info_label.grid(row=0, column=4, sticky="ew")

    def _build_settings_panel(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        # Keep the existing GUI policy that the radius sliders extend to at least 1600 px.
        radius_limit = max(
            1600,
            round(max(self.max_radius.get(), self.min_radius.get()) * 1.5),
        )

        slider_specs = [
            (
                "threshold",
                "Brightness threshold (dark <= T, light > T)",
                self.threshold,
                0,
                255,
                1,
            ),
            (
                "min_radius",
                "Minimum FINAL fitted semi-axis radius (px)",
                self.min_radius,
                1,
                radius_limit,
                1,
            ),
            (
                "max_radius",
                "Maximum FINAL fitted semi-axis radius (px)",
                self.max_radius,
                1,
                radius_limit,
                1,
            ),
            (
                "max_error",
                "Maximum average normalized ellipse error (%)",
                self.max_error,
                0.5,
                50,
                0.1,
            ),
            (
                "min_coverage",
                "Minimum TOTAL supported ellipse arc (%)",
                self.min_coverage,
                0,
                100,
                1,
            ),
        ]
        for row, spec in enumerate(slider_specs):
            self._add_slider(frame, row, *spec)

        threshold_auto_button = tk.Button(
            frame,
            text="Auto select",
            width=12,
            command=self.threshold_auto_button_clicked,
        )
        threshold_auto_button.grid(
            row=0,
            column=3,
            sticky="ns",
            padx=(10, 0),
            pady=2,
        )

        radius_auto_button = tk.Button(
            frame,
            text="Auto select",
            width=12,
            command=self.radius_auto_button_clicked,
        )
        radius_auto_button.grid(
            row=1,
            column=3,
            rowspan=2,
            sticky="nsew",
            padx=(10, 0),
            pady=2,
        )

        options = tk.Frame(frame)
        options.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 3))

        morphology_checkbox = tk.Checkbutton(
            options,
            text="Morphology cleanup for candidate search",
            variable=self.morphology,
            command=self.morphology_checkbox_changed,
        )
        morphology_checkbox.grid(row=0, column=0, sticky="w", padx=(0, 20))

        outer_limb_assistance_checkbox = tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=self.outer_limb_assistance_checkbox_changed,
        )
        outer_limb_assistance_checkbox.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 20),
        )

        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=self.horizon_checkbox_changed,
            state=tk.DISABLED,
        )
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")

        center_frame = tk.Frame(frame)
        center_frame.grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 5))

        center_target_label = tk.Label(
            center_frame,
            text="Center full-color image on:",
        )
        center_target_label.grid(row=0, column=0, sticky="w", padx=(0, 8))

        light_center_radiobutton = tk.Radiobutton(
            center_frame,
            text="Light ellipse",
            variable=self.center_target,
            value="light",
            command=self.center_target_radiobutton_changed,
        )
        light_center_radiobutton.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(0, 14),
        )

        dark_center_radiobutton = tk.Radiobutton(
            center_frame,
            text="Dark ellipse",
            variable=self.center_target,
            value="dark",
            command=self.center_target_radiobutton_changed,
        )
        dark_center_radiobutton.grid(row=0, column=2, sticky="w")

        button_frame = tk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))

        self.preview_button = tk.Button(
            button_frame,
            text="Refresh Preview",
            width=16,
            command=self.preview_button_clicked,
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 8))

        self.full_button = tk.Button(
            button_frame,
            text="Apply Full Resolution",
            width=20,
            command=self.full_button_clicked,
        )
        self.full_button.grid(row=0, column=1)

        status_label = tk.Label(
            frame,
            textvariable=self.status,
            anchor="w",
            justify="left",
            wraplength=1150,
        )
        status_label.grid(
            row=8,
            column=0,
            columnspan=4,
            sticky="ew",
            pady=(8, 0),
        )

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
        slider_label = tk.Label(parent, width=18, anchor="e")
        slider_label.grid(row=row, column=2, pady=2)

        def update_value_label(*_args):
            value = variable.get()
            if setting_name == "threshold":
                text = str(value)
            elif setting_name in ("min_radius", "max_radius"):
                text = f"{value} px"
            elif setting_name == "max_error":
                text = f"{value:.1f}%"
            elif setting_name == "min_coverage":
                text = f"{value}% (~{value * 3.6:.0f}°)"
            else:
                raise ValueError(f"unknown slider setting: {setting_name}")
            slider_label.config(text=text)

        variable.trace_add("write", update_value_label)
        update_value_label()

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
            self.apply_changed_setting(event.widget._setting_name, event.widget.get())

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
            self.apply_changed_setting(widget._setting_name, widget.get())

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
            textvariable=self.center_preview_text,
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        # The canvas is only a display surface. It renders the supplied retained
        # grayscale, BGR/BGRA, compressed color, or empty content directly.
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
    def save_centered_button_clicked(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the threshold-finder stage."
        )

    def threshold_auto_button_clicked(self):
        """Run/reuse complete Auto T, apply its winning threshold, and stop at SolarData."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        with self.blocked_gui():
            selected_threshold = find_auto_threshold(self.gray_image, state)
            auto_threshold_result = state.get("auto_threshold_result")
            if not isinstance(auto_threshold_result, AutoThresholdResult):
                raise ValueError(
                    "Auto-T returned without establishing AutoThresholdResult"
                )

        if selected_threshold is None:
            reason = auto_threshold_result.failure_reason
            self.status.set(
                "Automatic threshold could not be resolved"
                + (f" ({reason})." if reason is not None else ".")
            )
            return

        self.apply_changed_setting("threshold", selected_threshold)

    def radius_auto_button_clicked(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the threshold-finder stage."
        )

    def apply_changed_setting(self, setting_name, value):
        """Persist one completed GUI setting change and synchronize through SolarData."""
        variable = self.setting_variables[setting_name]
        variable.set(value)
        value = variable.get()

        if self.current_path is None or self.current_path not in self.image_state:
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]

        threshold_changed = False
        if setting_name == "threshold":
            # Threshold is never sparse once initialized. None means never initialized.
            threshold_changed = settings.threshold != value
            settings.threshold = value
        else:
            baseline = getattr(self.default_settings, setting_name)
            setattr(settings, setting_name, None if value == baseline else value)

        if self.gray_image is None:
            return

        with self.blocked_gui():
            if settings.threshold is None:
                self.status.set("Threshold is not initialized for the current image.")
                return

            threshold = settings.threshold

            # A genuinely new T explicitly resets the threshold pane to authoritative
            # grayscale before the new selected-T solar resolution is attempted.
            if threshold_changed and hasattr(self, "threshold_canvas"):
                self.render_canvas_content(self.threshold_canvas, self.gray_image)

            try:
                refined_component = resolve_threshold(
                    self.gray_image,
                    threshold,
                    state,
                )
            except ThresholdResolutionError as exc:
                self.status.set(
                    f"Grayscale remains displayed; SolarData could not be established "
                    f"at T={threshold} ({exc})."
                )
                return

            if hasattr(self, "threshold_canvas"):
                # The processing pane retains the plain solar mask as 2D uint8.
                # render_canvas_content() owns byte-equivalence and repaint decisions,
                # including replacement of stale downstream BGR overlays.
                self.render_canvas_content(
                    self.threshold_canvas,
                    bool_mask_to_uint8(refined_component),
                )

            self.status.set(
                f"{setting_name} applied; SolarData synchronized at T={threshold}."
            )

    def _selected_center_target_name(self):
        """Return the user-facing name of the selected centering target."""
        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"

    def morphology_checkbox_changed(self):
        """Apply the morphology checkbox immediately."""
        self.apply_changed_setting("morphology", self.morphology.get())

    def outer_limb_assistance_checkbox_changed(self):
        """Apply the outer-limb-assistance checkbox immediately."""
        self.apply_changed_setting(
            "outer_limb_assistance",
            self.outer_limb_assistance.get(),
        )

    def horizon_checkbox_changed(self):
        """Apply the horizon checkbox immediately."""
        self.apply_changed_setting("use_horizon", self.use_horizon.get())

    def center_target_radiobutton_changed(self):
        self._update_center_preview_label()
        self.apply_changed_setting("center_target", self.center_target.get())
        self.status.set(
            f"Centering target set to {self._selected_center_target_name()}. "
            "Actual centering will be implemented with ellipse detection."
        )

    def _update_center_preview_label(self):
        target = self._selected_center_target_name()
        self.center_preview_text.set(f"Full-color image — center on {target}")

    def preview_button_clicked(self):
        """Run heavy preview processing from authoritative current SolarData."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        if settings.threshold is None:
            self.status.set("Refresh Preview requires an initialized threshold.")
            return

        with self.blocked_gui():
            solar_data = state.get("solar_data")
            if solar_data is None:
                try:
                    resolve_threshold(
                        self.gray_image,
                        settings.threshold,
                        state,
                    )
                except ThresholdResolutionError as exc:
                    self.status.set(
                        f"Refresh Preview stopped: SolarData could not be established "
                        f"at T={settings.threshold} ({exc})."
                    )
                    return
                solar_data = state.get("solar_data")

            if not isinstance(solar_data, SolarData):
                raise ValueError("Refresh Preview requires SolarData or None")
            if solar_data.threshold != settings.threshold:
                raise ValueError(
                    "Refresh Preview found SolarData for a different threshold"
                )
            if solar_data.failure_reason is not None:
                if any(
                    value is not None
                    for value in (
                        solar_data.seed_point,
                        solar_data.component_mask,
                        solar_data.guard_mask,
                        solar_data.component_contour,
                    )
                ):
                    raise ValueError("failed SolarData must not contain geometry")
                self.status.set(
                    f"Refresh Preview stopped: SolarData failed at "
                    f"T={solar_data.threshold} ({solar_data.failure_reason})."
                )
                return
            if not solar_data.complete:
                raise ValueError(
                    "Refresh Preview found incomplete SolarData without a failure"
                )

            # TODO: horizon finding consumes solar_data.
            # TODO: ellipse finding consumes the horizon/SolarData products.
            # TODO: center the full-color image from the validated geometry.
            self.status.set(
                f"SolarData ready at T={solar_data.threshold}; downstream preview "
                "processing is not implemented in the threshold-finder branch."
            )

    def full_button_clicked(self):
        """Run heavy full-resolution processing from authoritative current SolarData."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        if settings.threshold is None:
            self.status.set("Apply Full Resolution requires an initialized threshold.")
            return

        with self.blocked_gui():
            solar_data = state.get("solar_data")
            if solar_data is None:
                try:
                    resolve_threshold(
                        self.gray_image,
                        settings.threshold,
                        state,
                    )
                except ThresholdResolutionError as exc:
                    self.status.set(
                        f"Apply Full Resolution stopped: SolarData could not be established "
                        f"at T={settings.threshold} ({exc})."
                    )
                    return
                solar_data = state.get("solar_data")

            if not isinstance(solar_data, SolarData):
                raise ValueError("Apply Full Resolution requires SolarData or None")
            if solar_data.threshold != settings.threshold:
                raise ValueError(
                    "Apply Full Resolution found SolarData for a different threshold"
                )
            if solar_data.failure_reason is not None:
                if any(
                    value is not None
                    for value in (
                        solar_data.seed_point,
                        solar_data.component_mask,
                        solar_data.guard_mask,
                        solar_data.component_contour,
                    )
                ):
                    raise ValueError("failed SolarData must not contain geometry")
                self.status.set(
                    f"Apply Full Resolution stopped: SolarData failed at "
                    f"T={solar_data.threshold} ({solar_data.failure_reason})."
                )
                return
            if not solar_data.complete:
                raise ValueError(
                    "Apply Full Resolution found incomplete SolarData without a failure"
                )

            # TODO: full-resolution horizon finding consumes solar_data.
            # TODO: full-resolution ellipse finding consumes those products.
            # TODO: center/export the full-color image from the validated geometry.
            self.status.set(
                f"SolarData ready at T={solar_data.threshold}; downstream full-resolution "
                "processing is not implemented in the threshold-finder branch."
            )


    def _handle_canvas_resize(self, event):
        """Refit once after a canvas resize interaction has settled."""
        canvas = event.widget
        pending_job = getattr(canvas, "_resize_render_job", None)
        if pending_job is not None:
            self.root.after_cancel(pending_job)
        canvas._resize_render_job = self.root.after(
            CANVAS_RESIZE_SETTLE_MS,
            self._finish_canvas_resize,
            canvas,
        )

    def _finish_canvas_resize(self, canvas):
        """Render retained canvas content once after resize events stop arriving."""
        canvas._resize_render_job = None
        rendered_content = getattr(canvas, "_rendered_content", None)
        if rendered_content is not None:
            self.render_canvas_content(canvas, rendered_content)


    def render_canvas_content(self, canvas, content):
        """Retain supplied content and render only when pixels or viewport changed."""
        if content is not None and not isinstance(content, (bytes, np.ndarray)):
            raise ValueError(
                "canvas content must be None, compressed image bytes, or a NumPy array"
            )

        previous_content = getattr(canvas, "_rendered_content", None)
        decoded_content = None
        decoded_previous_content = None

        if content is previous_content:
            equivalent = True
        elif content is None or previous_content is None:
            equivalent = content is None and previous_content is None
        elif isinstance(content, bytes):
            if isinstance(previous_content, bytes):
                equivalent = content == previous_content
            elif isinstance(previous_content, np.ndarray):
                decoded_content = decompress_image(content)
                equivalent = (
                    decoded_content.dtype == previous_content.dtype
                    and decoded_content.shape == previous_content.shape
                    and np.array_equal(decoded_content, previous_content)
                )
            else:
                raise ValueError(
                    "retained canvas content must be None, compressed image bytes, "
                    "or a NumPy array"
                )
        elif isinstance(previous_content, bytes):
            decoded_previous_content = decompress_image(previous_content)
            equivalent = (
                content.dtype == decoded_previous_content.dtype
                and content.shape == decoded_previous_content.shape
                and np.array_equal(content, decoded_previous_content)
            )
        elif isinstance(previous_content, np.ndarray):
            equivalent = (
                content.dtype == previous_content.dtype
                and content.shape == previous_content.shape
                and np.array_equal(content, previous_content)
            )
        else:
            raise ValueError(
                "retained canvas content must be None, compressed image bytes, "
                "or a NumPy array"
            )

        # Retain exactly the representation supplied by the caller. Immutable bytes
        # stay compressed; ndarray content is retained by reference rather than copied.
        canvas._rendered_content = content

        # The canvas now owns the replacement content. Drop obsolete local references
        # before allocating the fitted raster and encoded PNG for the new render.
        del decoded_previous_content
        del previous_content

        # NumPy and resize_img both use (height, width), so keep that ordering here.
        canvas_shape = np.asarray(
            (
                max(2, canvas.winfo_height() - 2),
                max(2, canvas.winfo_width() - 2),
            )
        )
        canvas_size = tuple(int(value) for value in canvas_shape)

        if (
            equivalent
            and getattr(canvas, "_rendered_canvas_size", None) == canvas_size
        ):
            return

        if content is None:
            canvas.delete("all")
            canvas._tk_photo_image = None
            canvas._rendered_canvas_size = canvas_size
            self.root.update_idletasks()
            return

        # Only now has rendering been selected. Reuse a decompression already required
        # for cross-representation equivalence; otherwise decode compressed content once.
        if decoded_content is not None:
            render_raster = decoded_content
        elif isinstance(content, bytes):
            render_raster = decompress_image(content)
        else:
            render_raster = content

        raster_shape = np.asarray(render_raster.shape[:2])
        scale = np.min(canvas_shape / raster_shape)
        fitted_shape = tuple(
            int(value) for value in np.rint(raster_shape * scale)
        )

        scaled_raster = resize_img(render_raster, fitted_shape)
        ok, encoded_png = cv2.imencode(".png", scaled_raster)
        if not ok:
            raise ValueError("could not encode canvas content")

        tk_photo = tk.PhotoImage(
            data=encoded_png.tobytes(),
            format="png",
        )
        canvas.delete("all")
        canvas.create_image(
            0,
            0,
            image=tk_photo,
            anchor="nw",
        )
        # Tk does not retain the Python PhotoImage object. Keep it alive while displayed.
        canvas._tk_photo_image = tk_photo
        canvas._rendered_canvas_size = canvas_size

        # Flush pending repaint work before synchronous processing continues.
        self.root.update_idletasks()


def main():
    parser = argparse.ArgumentParser(
        description="threshold-finder stage for the eclipse detector rebuild."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Ordered input files, or one folder containing .jpg/.tif/.tiff images",
    )
    args = parser.parse_args()

    image_paths = list(args.images)
    directories = [path for path in image_paths if os.path.isdir(path)]
    if directories:
        if len(image_paths) != 1:
            parser.error("pass either one folder or a list of files, not both")
        folder = image_paths[0]
        folder_names = sorted(os.listdir(folder), key=str.casefold)
        image_paths = [
            os.path.join(folder, name)
            for name in folder_names
            if os.path.isfile(os.path.join(folder, name))
            and os.path.splitext(name)[1].lower() in (".jpg", ".tif", ".tiff")
        ]

    root = tk.Tk()
    DetectorApp(root, image_paths)
    root.mainloop()


if __name__ == "__main__":
    main()
