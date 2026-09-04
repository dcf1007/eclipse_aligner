"""Reusable production block for coarse component extraction and threshold refinement.

This snippet is copied from the validated ``circle_arc_detector.py`` implementation.
It intentionally relies on production ``generate_kernel``, ``compress_full_mask``,
``ThresholdResolutionError``, ``largest_enclosed_bright_component`` and
``brightest_supported_component_point`` rather than duplicating them.
"""

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
