"""Apply the final agreed Auto-T contract corrections to circle_arc_detector.py.

This patch starts from the audited chat-14 corrected source. It changes only the
Auto-T section through AutoThresholdResult generation; SolarData and all downstream
selected-T processing are intentionally untouched.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _replace_node(source: str, node: ast.AST, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    while start > 0 and lines[start - 1].lstrip().startswith("@"):
        start -= 1
    return "".join(lines[:start]) + replacement.rstrip() + "\n\n" + "".join(lines[end:])


def _replace_named(source: str, name: str, replacement: str, kind: type[ast.AST]) -> str:
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, kind) and getattr(node, "name", None) == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one top-level {kind.__name__} named {name}, got {len(matches)}")
    return _replace_node(source, matches[0], replacement)


def apply(source: str) -> str:
    source = _replace_named(
        source,
        "AutoThresholdResult",
        '''@dataclass(frozen=True)
class AutoThresholdResult:
    threshold: int | None
    histogram_start_threshold: int
    work_res_threshold: int | None
    full_res_seed_point: tuple[int, int] | None
    cleaned_component_mask: bytes | None
    # ``resolved`` is currently redundant with whether Auto-T produced a complete
    # final result. Keep it explicit for readability for now; it may disappear later.
    resolved: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        """Reject contradictory resolved/unresolved result states."""
        if self.resolved:
            if self.threshold is None:
                raise ValueError("resolved Auto-T result requires a threshold")
            if self.work_res_threshold is None:
                raise ValueError("resolved Auto-T result requires a work-resolution threshold")
            if self.full_res_seed_point is None:
                raise ValueError("resolved Auto-T result requires a full-resolution seed point")
            if self.cleaned_component_mask is None:
                raise ValueError("resolved Auto-T result requires a cleaned component mask")
        else:
            if self.threshold is not None:
                raise ValueError("unresolved Auto-T result cannot contain a final threshold")
            if self.cleaned_component_mask is not None:
                raise ValueError("unresolved Auto-T result cannot contain a cleaned component mask")''',
        ast.ClassDef,
    )

    old_constants = '''EDGE_RADIUS = 25
EDGE_RECOVERY_FRACTION = 0.25
EDGE_RECOVERY_PERSISTENCE = 4
EDGE_MIN_SAMPLE_STRIDE = 8
EDGE_TARGET_PROFILE_COUNT = 2000
EDGE_TANGENT_SPAN = 12'''
    new_constants = '''# Photometric edge profiles are sampled along estimated outward contour normals.
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
EDGE_NORMAL_TANGENT_HALF_SPAN = 12'''
    if old_constants not in source:
        raise RuntimeError("expected original edge constant block not found")
    source = source.replace(old_constants, new_constants, 1)

    source = _replace_named(
        source,
        "_contour_normals",
        '''def _contour_normals(
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
    return sampled, outward''',
        ast.FunctionDef,
    )

    source = _replace_named(
        source,
        "_sample_edge_profiles",
        '''def _sample_edge_profiles(
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
    return profiles, distances''',
        ast.FunctionDef,
    )

    source = _replace_named(
        source,
        "measure_edge_alignment",
        '''def measure_edge_alignment(
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
    )''',
        ast.FunctionDef,
    )

    source = _replace_named(
        source,
        "find_auto_threshold",
        '''def find_auto_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int:
    """Determine automatic T, retain its seed, and cache the winning cleaned mask."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if full_res_gray.dtype != np.uint8:
        raise ValueError("automatic thresholding requires authoritative uint8 grayscale")

    # Auto-T limits the longest work-resolution dimension to 1200 pixels while
    # passing the complete target size explicitly to resize_img().
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

    # Build the work-resolution grayscale raster used only for component identity tracking.
    work_res_gray = resize_img(full_res_gray, work_res_size)

    # Start the downward work-resolution search at the histogram valley before the bright mode.
    histogram_start_T = find_histogram_start_threshold(work_res_gray)

    # Generate the fixed square work-resolution seed-support kernel here so its
    # footprint remains explicit and can be scaled equivalently at full resolution.
    work_res_seed_kernel = generate_kernel((5, 5), round_kernel=False)

    work_res_T: int | None = None
    full_res_seed_point: tuple[int, int] | None = None
    resolution_step = "work-resolution component search"

    try:
        # Establish one supported seed, then track only that connected identity downward.
        work_res_T, work_res_component = find_work_res_solar_component(
            work_res_gray,
            histogram_start_T,
            work_res_seed_kernel,
        )

        # Transfer only the mature work component geometry onto the exact source raster.
        full_res_search_mask = resize_img(
            work_res_component,
            (full_res_width, full_res_height),
            mask=True,
        )

        # Preserve the actual work seed-support footprint at the realized source scale.
        work_res_support_size = max(work_res_seed_kernel.shape)
        mapped_kernel_size = (
            work_res_support_size
            * max(full_res_gray.shape)
            / max(work_res_gray.shape)
        )
        full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)
        # Generate the equivalent full-resolution square support kernel once before
        # selecting the fixed authoritative full-resolution seed point.
        full_res_seed_kernel = generate_kernel(
            (full_res_kernel_size, full_res_kernel_size),
            round_kernel=False,
        )

        # Select the brightest deeply supported full-resolution seed anywhere in the
        # transferred mature component. This point remains authoritative through refinement.
        resolution_step = "full-resolution seed selection"
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

        # Expand the transferred component by the fixed 10% L2 margin to create the
        # immutable full-resolution guard shared by coarse separation and refinement.
        resolution_step = "full-resolution guard construction"
        image_scale = math.sqrt(full_res_width * full_res_height)
        full_res_guard_mask = dilate_component_mask(
            full_res_search_mask,
            AUTO_T_GUARD_DILATION_FRACTION * image_scale,
        )

        # Find the exact coarse separation boundary using fixed D7 cleanup, the
        # authoritative seed point, and the immutable full-resolution guard.
        resolution_step = "coarse separation"
        separation_T = find_separation_threshold(
            full_res_gray,
            work_res_T,
            full_res_seed_point,
            full_res_guard_mask,
        )

        # Fine refinement may only raise the proven separation T. It reuses the same
        # authoritative seed and fixed guard and returns its winning full-resolution mask.
        resolution_step = "fine refinement"
        final_T, cleaned_component_mask = refine_threshold(
            full_res_gray,
            separation_T,
            full_res_seed_point,
            full_res_guard_mask,
        )

    except ThresholdResolutionError as exc:
        # Expected inability to resolve at any Auto-T stage is persisted explicitly.
        # Keep any intermediate work T / source seed already established for diagnostics,
        # but a failed Auto-T never exposes a final threshold or cleaned component mask.
        result = AutoThresholdResult(
            threshold=None,
            histogram_start_threshold=histogram_start_T,
            work_res_threshold=work_res_T,
            full_res_seed_point=full_res_seed_point,
            cleaned_component_mask=None,
            reason=f"{resolution_step}: {exc}",
        )
        image_state["auto_threshold_result"] = result
        raise

    result = AutoThresholdResult(
        threshold=final_T,
        histogram_start_threshold=histogram_start_T,
        work_res_threshold=work_res_T,
        full_res_seed_point=full_res_seed_point,
        cleaned_component_mask=cleaned_component_mask,
        resolved=True,
    )
    image_state["auto_threshold_result"] = result
    return final_T''',
        ast.FunctionDef,
    )

    ast.parse(source)
    return source


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(apply(args.source.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
