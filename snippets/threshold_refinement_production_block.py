"""Reusable production block for the agreed automatic-threshold search/refinement.

The block assumes the host module already provides ``generate_kernel``,
``compress_full_mask``, ``ThresholdResolutionError``,
``largest_enclosed_bright_component`` and ``brightest_supported_component_point``.
It intentionally contains no SolarData/resolve_threshold changes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

MAX_T_REFINEMENT_STEPS = 10

# Photometric edge profiles are sampled along estimated outward contour normals.
# The 25-pixel radius supplies enough interior/exterior context to see the full
# brightness transition and, by the accepted current design, also defines the
# offset at which edge-alignment quality falls to zero.
EDGE_PROFILE_RADIUS_PX = 25

# After the strongest outward intensity fall, the wanted edge is the point where
# the negative slope has recovered to 25% of that peak and remains recovered for
# four consecutive profile positions. This locates the outer end of the transition
# rather than its strongest-gradient midpoint.
EDGE_SLOPE_RECOVERY_FRACTION = 0.25
EDGE_SLOPE_RECOVERY_PERSISTENCE_PX = 4

# Adjacent CHAIN_APPROX_NONE contour points yield highly redundant normal profiles.
# Sample at least eight contour indices apart, and increase that spacing only for
# exceptionally long contours so approximately no more than 2000 profiles are
# evaluated. These limits control runtime while retaining broad contour coverage.
EDGE_PROFILE_MIN_SAMPLE_STRIDE = 8
EDGE_PROFILE_MAX_SAMPLE_COUNT = 2000

# Estimate each local tangent from contour points +/-12 indices around the sample.
# The wider chord suppresses raster stair-step noise before the tangent is rotated
# into an outward normal; it remains short enough to preserve local curvature.
EDGE_NORMAL_TANGENT_HALF_SPAN = 12

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
    solidity: float
    internal_dark_fraction: float
    edge_alignment: float
    edge_credible_fraction: float
    edge_transition_monotonicity: float







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



def find_work_res_solar_component(
    work_res_gray: np.ndarray,
    start_T: int,
    work_res_seed_kernel: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Establish one supported work-resolution seed, then track that identity downward."""
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
    """Progressively clean, guard, then extract the component containing ``seed_point``."""
    if binary_mask.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")
    if guard_mask.ndim != 2 or guard_mask.shape != binary_mask.shape or not np.any(guard_mask):
        raise ValueError("guard must be a non-empty mask matching binary mask")
    if not np.any(binary_mask):
        return None

    cleaned = np.where(binary_mask != 0, 255, 0).astype(np.uint8)
    for kernel in SOLAR_CLEANUP_KERNELS:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    cleaned[~guard_mask] = 0
    return extract_component(cleaned, seed_point)



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



def find_separation_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> int:
    """Return the lowest T whose D7-cleaned component is separated by the fixed guard."""
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
    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        SEPARATION_KERNEL,
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        SEPARATION_KERNEL,
        iterations=1,
    )
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
        for threshold in range(start_T - 1, -1, -1):
            binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                SEPARATION_KERNEL,
                iterations=1,
            )
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                SEPARATION_KERNEL,
                iterations=1,
            )
            binary[~full_res_guard_mask] = 0
            component = extract_component(binary, full_res_seed_point)
            if component is None or np.any(component & full_res_guard_boundary):
                break
            best_T = threshold
        return best_T

    for threshold in range(start_T + 1, 256):
        binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            SEPARATION_KERNEL,
            iterations=1,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            SEPARATION_KERNEL,
            iterations=1,
        )
        if binary[seed_y, seed_x] == 0:
            break
        binary[~full_res_guard_mask] = 0
        component = extract_component(binary, full_res_seed_point)
        if component is not None and not np.any(component & full_res_guard_boundary):
            return threshold

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became separated after D7 cleanup"
    )



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
    tangent_half_span: int = EDGE_NORMAL_TANGENT_HALF_SPAN,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled contour points and consistently outward unit normals."""
    points = contour[:, 0, :].astype(np.float32)
    count = len(points)
    if count < 2 * tangent_half_span + 3:
        tangent_half_span = max(1, count // 8)

    indices = np.arange(0, count, max(1, sample_stride), dtype=np.int32)
    sampled = points[indices]
    previous = points[(indices - tangent_half_span) % count]
    following = points[(indices + tangent_half_span) % count]
    tangent = following - previous
    lengths = np.linalg.norm(tangent, axis=1)
    usable = lengths > 1e-6
    sampled = sampled[usable]
    tangent = tangent[usable] / lengths[usable, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])).astype(np.float32)

    # A tangent determines two opposite normals. Probe 1..5 pixels on both sides
    # of the component boundary and call the side with more component support inward.
    # This short multi-pixel vote is more stable than deciding from one raster sample.
    positive_inside = np.zeros(len(sampled), np.int16)
    negative_inside = np.zeros(len(sampled), np.int16)
    for distance in (1.0, 2.0, 3.0, 4.0, 5.0):
        positive_inside += _nearest_mask(component, sampled + normal * distance).astype(np.int16)
        negative_inside += _nearest_mask(component, sampled - normal * distance).astype(np.int16)

    positive_is_inward = positive_inside > negative_inside
    tied = positive_inside == negative_inside
    if np.any(tied):
        # Resolve otherwise ambiguous votes with a direct two-pixel inside/outside
        # check. Samples still ambiguous here cannot supply a trustworthy normal.
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
    full_res_gray: np.ndarray,
    component: np.ndarray,
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample lightly smoothed full-resolution grayscale profiles across the contour."""
    sample_stride = max(
        EDGE_PROFILE_MIN_SAMPLE_STRIDE,
        math.ceil(max(1, len(contour)) / EDGE_PROFILE_MAX_SAMPLE_COUNT),
    )
    points, outward = _contour_normals(component, contour, sample_stride)
    if len(points) == 0:
        raise ThresholdResolutionError("no oriented contour samples for photometric edge")

    distances = np.arange(
        -EDGE_PROFILE_RADIUS_PX,
        EDGE_PROFILE_RADIUS_PX + 1,
        dtype=np.float32,
    )
    xy = points[:, None, :] + outward[:, None, :] * distances[None, :, None]
    full_res_height, full_res_width = full_res_gray.shape
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    complete = (
        (x.min(axis=1) >= 0)
        & (x.max(axis=1) <= full_res_width - 1)
        & (y.min(axis=1) >= 0)
        & (y.max(axis=1) <= full_res_height - 1)
    )
    xy = xy[complete]
    if len(xy) == 0:
        raise ThresholdResolutionError("all photometric edge profiles leave the image")

    full_res_gray_float = full_res_gray.astype(np.float32, copy=False)
    map_x = xy[:, :, 0].astype(np.float32)
    map_y = xy[:, :, 1].astype(np.float32)
    profile_chunks = []
    for start in range(0, len(map_x), 15000):
        stop = min(len(map_x), start + 15000)
        profile_chunks.append(
            cv2.remap(
                full_res_gray_float,
                map_x[start:stop],
                map_y[start:stop],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
    profiles = np.vstack(profile_chunks)

    # Light 1-D smoothing prevents interpolation noise or one-pixel fluctuations
    # from determining the strongest derivative. The accepted 7-pixel, sigma=1.25
    # filter is intentionally kept local because it may be revisited with the edge
    # confidence implementation rather than treated as a public tuning parameter.
    profiles = cv2.GaussianBlur(profiles, (7, 1), sigmaX=1.25, sigmaY=0)
    return profiles, distances




def measure_edge_alignment(
    full_res_gray: np.ndarray,
    component: np.ndarray,
    contour: np.ndarray,
) -> tuple[float, float, float]:
    """Measure edge alignment plus credibility and transition-monotonicity confidence."""
    profiles, distances = _sample_edge_profiles(full_res_gray, component, contour)
    profile_count = len(profiles)
    drop = (profiles[:, :-2] - profiles[:, 2:]) * 0.5
    derivative_positions = distances[1:-1]
    peak_indices = np.argmax(drop, axis=1)
    rows = np.arange(profile_count)
    peak_drop = drop[rows, peak_indices]
    peak_position = distances[peak_indices + 1]

    # Use the median of the six samples at each profile end as robust interior and
    # exterior plateau estimates. Reject profiles that have neither a minimal total
    # inside-to-outside drop nor a detectable local falling slope; these deliberately
    # permissive accepted floors keep flat/noise-only profiles out of edge confidence.
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
        recovery_limit = (
            EDGE_SLOPE_RECOVERY_FRACTION * float(peak_drop[index])
        )
        for derivative_index in range(
            peak_derivative_index + 1,
            len(derivative_positions) - EDGE_SLOPE_RECOVERY_PERSISTENCE_PX + 1,
        ):
            segment = drop[
                index,
                derivative_index : derivative_index + EDGE_SLOPE_RECOVERY_PERSISTENCE_PX,
            ]
            if (
                len(segment) == EDGE_SLOPE_RECOVERY_PERSISTENCE_PX
                and np.all(segment <= recovery_limit)
            ):
                endpoints[index] = derivative_positions[derivative_index]
                break

        # Estimate the transition interval from local transition and plateau lines.
        # The accepted 7-sample fits, small slope tolerances, +/-5-pixel intersection
        # allowance, +/-6-pixel fallback, and -0.03 monotonicity tolerance are retained
        # because they reproduce the approved Auto-T choices. They are implementation
        # parameters, not immutable theory, and may change or disappear after review.
        #
        # TODO: Revisit and benchmark this edge-confidence calculation after the
        # threshold-finder branch is updated and regression-stable. The current
        # transition/plateau fitting and monotonicity logic works as expected but may
        # be more complex than necessary. Compare against a simpler confidence measure
        # before making any behavioral change.
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

    # Credibility asks whether profiles contain a meaningful falling transition.
    # Alignment itself uses only credible profiles where the 25%/4-pixel recovery
    # endpoint was actually found. Keep these concepts separate for now; their
    # relationship belongs in the later confidence-simplification benchmark above.
    valid_endpoints = endpoints[np.isfinite(endpoints) & credible]
    valid_monotonicity = monotonicity[np.isfinite(monotonicity) & credible]
    median_abs_offset = (
        float(np.median(np.abs(valid_endpoints)))
        if len(valid_endpoints)
        else float(EDGE_PROFILE_RADIUS_PX)
    )
    transition_monotonicity = (
        float(np.median(valid_monotonicity))
        if len(valid_monotonicity)
        else 0.0
    )
    alignment = 1.0 - min(
        median_abs_offset / EDGE_PROFILE_RADIUS_PX,
        1.0,
    )
    return (
        float(alignment),
        float(credible_fraction),
        float(transition_monotonicity),
    )




def refine_threshold(
    full_res_gray: np.ndarray,
    base_threshold: int,
    full_res_seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> tuple[int, bytes]:
    """Fine-tune one separated threshold and return T plus its compressed cleaned mask."""
    if full_res_gray.ndim != 2 or full_res_gray.dtype != np.uint8:
        raise ValueError("threshold refinement requires authoritative uint8 grayscale")
    if (
        full_res_guard_mask.ndim != 2
        or full_res_guard_mask.shape != full_res_gray.shape
        or not np.any(full_res_guard_mask)
    ):
        raise ValueError("guard must be a non-empty mask matching grayscale")
    if not 0 <= base_threshold <= 255:
        raise ValueError("base threshold must be 0..255")

    seed_x, seed_y = full_res_seed_point
    full_res_height, full_res_width = full_res_gray.shape
    if not (0 <= seed_x < full_res_width and 0 <= seed_y < full_res_height):
        raise ValueError("full-resolution seed lies outside grayscale raster")
    if not full_res_guard_mask[seed_y, seed_x]:
        raise ValueError("full-resolution seed must lie inside the fixed guard")

    full_res_guard_boundary = find_guard_boundary(full_res_guard_mask)
    measurements: list[ThresholdMeasurement] = []
    compressed_masks: dict[int, bytes] = {}
    raw_reference_area: int | None = None
    raw_reference_roughness: float | None = None

    max_threshold = min(255, base_threshold + MAX_T_REFINEMENT_STEPS)
    for threshold in range(base_threshold, max_threshold + 1):
        threshold_mask = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)

        cleaned_component = clean_solar_component(
            threshold_mask,
            full_res_seed_point,
            full_res_guard_mask,
        )
        if (
            cleaned_component is not None
            and not np.any(cleaned_component & full_res_guard_boundary)
        ):
            contour = find_external_contour(cleaned_component)
            filled_area = measure_filled_area(contour)
            roughness = measure_roughness(contour, filled_area)
            solidity = measure_solidity(contour, filled_area)
            component_area = int(np.count_nonzero(cleaned_component))
            internal_dark_fraction = measure_internal_dark_fraction(
                component_area,
                filled_area,
            )
            edge_alignment, credible_fraction, transition_monotonicity = (
                measure_edge_alignment(full_res_gray, cleaned_component, contour)
            )
            measurements.append(
                ThresholdMeasurement(
                    threshold=threshold,
                    filled_area=filled_area,
                    roughness=roughness,
                    solidity=solidity,
                    internal_dark_fraction=internal_dark_fraction,
                    edge_alignment=edge_alignment,
                    edge_credible_fraction=credible_fraction,
                    edge_transition_monotonicity=transition_monotonicity,
                )
            )
            compressed_masks[threshold] = compress_full_mask(cleaned_component)

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

    reliability_samples = [
        measurement.edge_credible_fraction
        * measurement.edge_transition_monotonicity
        for measurement in measurements
    ]
    edge_reliability = float(np.median(reliability_samples))
    edge_reliability = (
        float(np.clip(edge_reliability, 0.0, 1.0))
        if math.isfinite(edge_reliability)
        else 0.0
    )
    edge_weight = edge_reliability**2

    best_threshold: int | None = None
    best_score = -math.inf
    for measurement in measurements:
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
        # Measurements are in ascending T order. Strict '>' therefore keeps the
        # lower T only when two floating-point scores are exactly equal.
        if score > best_score:
            best_score = score
            best_threshold = measurement.threshold

    if best_threshold is None:
        raise ThresholdResolutionError("threshold refinement produced no score winner")
    return best_threshold, compressed_masks[best_threshold]

