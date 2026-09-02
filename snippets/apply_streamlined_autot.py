#!/usr/bin/env python3
"""Apply the approved streamlined Auto-T architecture to circle_arc_detector.py.

This retained patch is intentionally structural: it replaces only the threshold,
resizing, kernel-generation, and SolarData-dilation sections whose ownership changed
in the approved threshold-finder refactor. Unrelated GUI/application code is left
untouched.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys


MODULE_DOCSTRING = '''"""Eclipse alignment GUI with grayscale automatic threshold selection.

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
"""'''

CONSTANTS = '''WORK_MAX_DIM = 1200
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
REFINEMENT_ITERATIONS = 1'''

AUTO_RESULT = '''@dataclass(frozen=True)
class AutoThresholdResult:
    threshold: int | None
    histogram_start_threshold: int
    work_res_threshold: int | None
    full_res_seed_point: tuple[int, int] | None
    resolved: bool
    reason: str = ""'''

AUTO_FUNCTIONS = r'''def resize_img(
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
'''

GENERATE_KERNEL = r'''def generate_kernel(size: int, round_kernel: bool = False) -> np.ndarray:
    """Return a centered positive odd square or discrete Euclidean-disk kernel."""
    size = int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    if not round_kernel:
        return np.ones((size, size), dtype=np.uint8)
    radius = size // 2
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return ((xx * xx + yy * yy) <= radius * radius).astype(np.uint8)
'''

CLEANUP_CANDIDATES = r'''def cleanup_morphology_candidates(component: np.ndarray) -> dict[str, np.ndarray]:
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
'''


RESOLVE = r'''def resolve_threshold(
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
'''


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} replacement, found {count}")
    return updated


def patch_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    text = replace_once(text, r'\A""".*?"""', MODULE_DOCSTRING, "module docstring")
    text = replace_once(
        text,
        r'WORK_MAX_DIM = 1200\n.*?REFINEMENT_ITERATIONS = 1',
        CONSTANTS,
        "threshold constants",
    )
    text = replace_once(
        text,
        r'@dataclass\(frozen=True\)\nclass CoarseThresholdResult:.*?\n\ndef to_gray',
        AUTO_RESULT + "\n\n\ndef to_gray",
        "Auto-T result classes",
    )
    text = replace_once(
        text,
        r'def resize_gray_max_dim\(.*?(?=\n# ---------------------------------------------------------------------------\n# (?:Cleanup morphology candidates|Final-T full-resolution solar resolution and persistence))',
        AUTO_FUNCTIONS + "\n",
        "Auto-T function block",
    )
    if "def euclidean_disk_kernel" in text:
        text = replace_once(
            text,
            r'def euclidean_disk_kernel\(.*?\n\n\ndef open_close_component',
            GENERATE_KERNEL + "\n\ndef open_close_component",
            "kernel generator",
        )
    else:
        marker = "# ---------------------------------------------------------------------------\n# Final-T full-resolution solar resolution and persistence"
        if marker not in text:
            raise RuntimeError("could not place kernel generator")
        text = text.replace(marker, GENERATE_KERNEL + "\n\n" + marker, 1)

    if "def cleanup_morphology_candidates" in text:
        text = replace_once(
            text,
            r'def cleanup_morphology_candidates\(.*?\n\n\ndef seed_connected_cleanup_candidates',
            CLEANUP_CANDIDATES + "\n\ndef seed_connected_cleanup_candidates",
            "cleanup candidate generation",
        )
    text = replace_once(
        text,
        r'def _full_mask_from_observation_region\(.*?\n\n\ndef _ordered_external_component_contour',
        "def _ordered_external_component_contour",
        "obsolete full-mask observation wrapper",
    )
    text = replace_once(
        text,
        r'def resolve_threshold\(.*?\n\nclass DetectorApp:',
        RESOLVE + "\n\nclass DetectorApp:",
        "final threshold resolver",
    )
    text = replace_once(
        text,
        r'\s*interpolation = cv2\.INTER_AREA if scale < 1 else cv2\.INTER_LINEAR\n'
        r'\s*scaled_raster = cv2\.resize\(\n'
        r'\s*unscaled_render_raster, fitted_size, interpolation=interpolation\n'
        r'\s*\)',
        "\n        scaled_raster = resize_img(unscaled_render_raster, fitted_size)",
        "GUI canvas resize",
    )

    # resize_img() is the single raster-scaling primitive after this refactor.
    if text.count("cv2.resize(") != 1:
        raise RuntimeError("direct cv2.resize call remains outside resize_img")

    # No old architecture names should survive production after the structural replacements.
    forbidden = (
        "CoarseThresholdResult",
        "ObservationRegion",
        "resize_gray_max_dim",
        "find_rightmost_histogram_peak",
        "coarse_threshold_search",
        "equivalent_full_resolution_seed_kernel",
        "_map_interval",
        "establish_full_resolution_seed",
        "find_full_resolution_enclosed_seed_component",
        "_build_observation_region",
        "_evaluate_observation_region",
        "_full_mask_from_observation_region",
        "euclidean_disk_kernel",
    )
    leftovers = [name for name in forbidden if name in text]
    if leftovers:
        raise RuntimeError(f"obsolete production names remain: {leftovers}")

    path.write_text(text, encoding="utf-8")


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")
    patch_source(target)


if __name__ == "__main__":
    main()
