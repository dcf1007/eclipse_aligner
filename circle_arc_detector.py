"""GUI-first eclipse detector shell.

This milestone contains the rebuilt interface plus the grayscale-only automatic
threshold finder. Ellipse fitting, horizon handling, radius auto-selection, image
centering, and export are not implemented yet. The interface is based on the latest GUI from
``refactor/cleanup-performance`` and adds a mutually exclusive centering target:
light ellipse (default) or dark ellipse. A clicked slider retains keyboard focus
for arrow-key adjustment until the mouse is clicked anywhere outside that slider.
GUI-only Auto select buttons are provided for threshold and radius, and a
Save centered images placeholder is available beside the image-loading control.
The preview rasters contain no placeholder text. Empty preview regions are stored
as BGRA image data with alpha=0 rather than simulated with a matching Tk background.
Any setting change replaces the threshold preview with a fully transparent BGRA
frame until the next explicit Refresh Preview or Apply Full Resolution action.
"""


import argparse
import base64
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np



IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)


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
# Grayscale-only automatic threshold finder (tested implementation)
# ---------------------------------------------------------------------------
"""Optimized grayscale-only eclipse automatic threshold finder candidate.

The algorithm intentionally uses only grayscale histogram modes and 8-connected
component topology. Threshold semantics are fixed:

    dark  = gray <= T
    light = gray > T

No HSV/color interpretation, Otsu thresholding, morphology, ellipse fitting,
bright-pixel dominance, competitor gain, or horizon logic is used here.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

WORK_MAX_DIM = 1200
PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)
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


def local_peak_signal(histogram: np.ndarray) -> np.ndarray:
    """Return a 3-bin [1,2,1]/4 signal for local peak/valley detection."""
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.shape != (256,):
        raise ValueError(f"Expected 256-bin histogram, got shape {hist.shape}")
    return np.convolve(hist, PEAK_KERNEL, mode="same")


def histogram_with_peak_signal(gray: np.ndarray):
    """Return exact histogram plus the local 3-bin peak/valley signal."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    return hist, local_peak_signal(hist)


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
    """Rightmost locally smoothed mode and the valley defining its left edge."""
    _hist, signal = histogram_with_peak_signal(gray)
    peak = max(_local_peaks(signal))
    return int(peak), int(_preceding_valley(signal, peak))


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


class DetectorApp:
    """GUI shell for the eclipse detector rebuild.

    Only interface behavior is implemented here: image loading/navigation,
    controls, preview panes, and the centering-target selector. Detector buttons
    deliberately report that backend functionality has not yet been implemented.
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

        # Threshold is stored per image so a manual override survives navigation.
        # Automatic selection runs on first load and only reruns when Auto select is
        # explicitly clicked for that image.
        self.image_thresholds: dict[str, int] = {}
        self.image_auto_results = {}

        self.threshold_photo = None
        self.color_photo = None
        self.resize_job = None

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

        self.status = tk.StringVar(value="GUI-only milestone. Load images to inspect the interface.")
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector — GUI milestone")
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.build_navigation()
        self.build_controls()
        self.build_previews()
        self.update_center_preview_label()

        # Tk's toplevel bindtag receives mouse events from every child widget.
        # Use it to clear slider keyboard focus as soon as the user clicks
        # anywhere outside the currently focused slider.
        root.bind("<ButtonPress-1>", self.release_scale_focus_if_outside, add="+")
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
            self.threshold_photo = None
            self.color_photo = None
            self.redraw()
            self.status.set(f"Could not load image: {path}")
        else:
            # Convert the original image to authoritative grayscale ONCE. The auto
            # threshold module derives its 1200px INTER_AREA working raster from
            # this grayscale image; no HSV/color threshold path exists.
            self.color_image = opaque_bgra(image)
            self.gray_image = to_gray(image)

            restored = path in self.image_thresholds
            if restored:
                selected_threshold = int(self.image_thresholds[path])
            else:
                result = auto_threshold_from_gray(self.gray_image)
                selected_threshold = int(result.threshold)
                self.image_thresholds[path] = selected_threshold
                self.image_auto_results[path] = result

            # Setting the Tk variable may clear an older preview through its trace;
            # image loading is one of the explicit operations that immediately
            # regenerates the black/white threshold raster afterward.
            self.threshold.set(selected_threshold)
            self.render_threshold_preview()
            self.redraw()

            if restored:
                self.status.set(
                    f"Image loaded. Restored stored threshold T={selected_threshold}."
                )
            elif result.resolved:
                self.status.set(
                    "Image loaded. Automatic grayscale threshold "
                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                    f"histogram start={result.histogram_left_edge})."
                )
            else:
                self.status.set(
                    "Image loaded. Automatic component tracking was unresolved; "
                    f"using rightmost-histogram left edge T={selected_threshold}."
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
    def build_navigation(self):
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

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        radius_limit = max(1600, round(max(self.args.max_radius, self.args.min_radius) * 1.5))

        rows = [
            ("Brightness threshold (dark <= T, light > T)", self.threshold,
             0, 255, 1, lambda v: str(int(float(v)))),
            ("Minimum FINAL fitted semi-axis radius (px)", self.min_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum FINAL fitted semi-axis radius (px)", self.max_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum average normalized ellipse error (%)", self.max_error,
             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),
            ("Minimum TOTAL supported ellipse arc (%)", self.min_coverage,
             0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)

        # Selection buttons are GUI placeholders at this milestone. Threshold has
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
            command=self.pending,
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=self.pending,
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=self.pending,
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
            command=self.center_target_changed,
        ).grid(row=0, column=1, sticky="w", padx=(0, 14))
        tk.Radiobutton(
            center_frame,
            text="Dark ellipse",
            variable=self.center_target,
            value="dark",
            command=self.center_target_changed,
        ).grid(row=0, column=2, sticky="w")

        tk.Label(frame, text="Useful threshold candidates:").grid(
            row=7, column=0, sticky="nw"
        )
        self.palette_frame = tk.Frame(frame)
        self.palette_frame.grid(row=7, column=1, columnspan=3, sticky="w", pady=(0, 8))
        tk.Label(
            self.palette_frame,
            text="Not implemented yet",
            fg="#666666",
        ).grid(row=0, column=0, sticky="w")

        button_frame = tk.Frame(frame)
        button_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(2, 0))
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
        ).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=42, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        scale = tk.Scale(
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
        scale.grid(row=row, column=1, sticky="ew", pady=2)

        # Tk Scale supports precise arrow-key adjustment while it owns keyboard
        # focus. Explicitly focus the clicked scale and leave focus there after the
        # mouse interaction, instead of relying on platform-specific focus policy.
        scale.bind("<ButtonPress-1>", self.focus_scale, add="+")
        scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")
        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            value_label.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update_value)
        update_value()

    @staticmethod
    def focus_scale(event):
        """Keep a clicked slider focused so arrow keys continue to adjust it."""
        event.widget.focus_set()

    def release_scale_focus_if_outside(self, event):
        """Release slider focus immediately when the mouse clicks elsewhere.

        Clicking the focused slider itself keeps focus. Clicking a different slider
        transfers focus through that slider's own ButtonPress binding, so this
        handler also leaves it alone. Any non-slider click removes the keyboard
        focus ring from the previously focused slider.
        """
        focused = self.root.focus_get()
        if isinstance(focused, tk.Scale) and event.widget is not focused:
            self.root.focus_set()

    def build_previews(self):
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
        self.threshold_canvas.bind("<Configure>", self.schedule_redraw)
        self.color_canvas.bind("<Configure>", self.schedule_redraw)

    # ------------------------------------------------------------------
    # GUI-only actions
    # ------------------------------------------------------------------
    def save_centered_images(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the GUI milestone."
        )

    def auto_select_threshold(self):
        """Rerun image-only automatic T selection without regenerating the preview."""
        if self.gray_image is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        result = auto_threshold_from_gray(self.gray_image)
        selected_threshold = int(result.threshold)
        if self.current_path is not None:
            self.image_thresholds[self.current_path] = selected_threshold
            self.image_auto_results[self.current_path] = result

        # The threshold variable trace deliberately clears any stale preview. Per
        # user requirement, Auto select itself DOES NOT regenerate that preview;
        # Refresh Preview / Apply Full Resolution remain explicit preview actions.
        self.threshold.set(selected_threshold)
        if result.resolved:
            self.status.set(
                "Automatic grayscale threshold selected: "
                f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                f"histogram start={result.histogram_left_edge}). "
                "Preview not regenerated."
            )
        else:
            self.status.set(
                "Automatic component tracking unresolved; "
                f"using rightmost-histogram left edge T={selected_threshold}. "
                "Preview not regenerated."
            )

    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the GUI milestone."
        )

    def render_threshold_preview(self):
        """Render authoritative black/white mask for the currently selected T."""
        if self.gray_image is None:
            self.threshold_preview = transparent_bgra()
            self.threshold_photo = None
            return

        # Exact threshold semantics: dark = gray <= T, light = gray > T.
        light_mask = cv2.compare(
            self.gray_image,
            int(self.threshold.get()),
            cv2.CMP_GT,
        )
        preview = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGRA)
        preview[:, :, 3] = 255
        self.threshold_preview = preview
        self.threshold_photo = None

    def clear_threshold_preview(self):
        """Replace stale threshold output with a fully transparent BGRA frame."""
        if self.color_image is not None:
            height, width = self.color_image.shape[:2]
            self.threshold_preview = transparent_bgra(width, height)
        else:
            self.threshold_preview = transparent_bgra()
        self.threshold_photo = None
        if hasattr(self, "threshold_canvas"):
            self.threshold_photo = self.show_image(
                self.threshold_canvas, self.threshold_preview
            )

    def pending(self, *_args):
        if self.current_path is not None and self.gray_image is not None:
            self.image_thresholds[self.current_path] = int(self.threshold.get())
        self.clear_threshold_preview()
        self.status.set(
            "Settings changed. Threshold preview cleared; Refresh Preview or Apply Full Resolution to recompute."
        )

    def center_target_changed(self):
        self.clear_threshold_preview()
        self.update_center_preview_label()
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.status.set(
            f"Centering target set to {target}. Actual centering will be implemented with ellipse detection."
        )

    def update_center_preview_label(self):
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.center_preview_label.set(f"Full-color image — center on {target}")

    def refresh_preview(self):
        if self.gray_image is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return
        self.render_threshold_preview()
        self.redraw()
        self.status.set(
            f"Threshold preview regenerated at T={int(self.threshold.get())}."
        )

    def apply_full_resolution(self):
        if self.gray_image is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return
        self.render_threshold_preview()
        self.redraw()
        self.status.set(
            "Full-resolution threshold preview applied at "
            f"T={int(self.threshold.get())}. Ellipse detector backend not implemented yet."
        )

    def schedule_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        self.resize_job = None
        if hasattr(self, "threshold_canvas"):
            self.threshold_photo = self.show_image(
                self.threshold_canvas, self.threshold_preview
            )
        if hasattr(self, "color_canvas"):
            if self.color_image is None:
                self.color_photo = self.show_image(
                    self.color_canvas, transparent_bgra()
                )
            else:
                self.color_photo = self.show_image(self.color_canvas, self.color_image)

    @staticmethod
    def show_image(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(image, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
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
        return photo

    def close(self):
        self.root.destroy()


def build_parser():
    parser = argparse.ArgumentParser(description="GUI milestone for the eclipse detector rebuild.")
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
