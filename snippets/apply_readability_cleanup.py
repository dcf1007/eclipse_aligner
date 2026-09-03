#!/usr/bin/env python3
"""Apply the approved readability cleanup to the threshold-finder implementation.

The transformation keeps the algorithmic stages intact while removing silent
coercions, repair clamps, redundant invariant checks, GUI lambdas, and the cropped
implementation detail from component-mask dilation. It also synchronizes the
retained streamlined Auto-T patch with the current production conventions.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re

ROOT = Path('.')
SOURCE_PATH = ROOT / 'circle_arc_detector.py'
RETAINED_PATCH_PATH = ROOT / 'snippets' / 'apply_streamlined_autot.py'


def replace_module_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = next(
        (item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name),
        None,
    )
    if node is None:
        raise RuntimeError(f'module function {name!r} not found')
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    replacement_text = replacement.rstrip() + ('\n' if replacement else '')
    lines[start:end] = [replacement_text]
    return ''.join(lines)


def replace_class_method(source: str, class_name: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    class_node = next(
        (item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name),
        None,
    )
    if class_node is None:
        raise RuntimeError(f'class {class_name!r} not found')
    node = next(
        (item for item in class_node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name),
        None,
    )
    if node is None:
        raise RuntimeError(f'{class_name}.{name} not found')
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    lines[start:end] = [replacement.rstrip() + '\n']
    return ''.join(lines)


GENERATE_KERNEL = r'''def generate_kernel(
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
'''

TRANSPARENT_BGRA = r'''def transparent_bgra(width: int = 1, height: int = 1) -> np.ndarray:
    """Return a BGRA frame whose pixels are fully transparent (alpha = 0)."""
    if width <= 0 or height <= 0:
        raise ValueError("transparent raster dimensions must be positive")
    return np.zeros((height, width, 4), dtype=np.uint8)
'''

OPAQUE_BGRA = r'''def opaque_bgra(bgr: np.ndarray) -> np.ndarray:
    """Convert a normal OpenCV BGR image to BGRA with fully opaque image pixels."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
'''

COMPONENT_DESCRIPTOR = r'''def _component_descriptor(component: np.ndarray, threshold: int) -> ThresholdTopology:
    component = np.asarray(component, dtype=bool)
    area = np.count_nonzero(component)
    if area <= 0:
        raise ValueError("empty component")

    component_u8 = np.where(component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("component has no external contour")

    contour = max(contours, key=cv2.contourArea)
    contour_n = len(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0.0 else 0.0
    roughness = perimeter / (2.0 * math.sqrt(math.pi * area))

    # Internal dark fraction is measured in raster pixels, not contourArea. Fill
    # the selected external contour, then count original component pixels (N1) and
    # missing/dark pixels (N0) inside that fill.
    filled = np.zeros(component.shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    filled_bool = filled != 0
    n1 = np.count_nonzero(component & filled_bool)
    n0 = np.count_nonzero(filled_bool & ~component)
    total = n0 + n1
    internal_dark_fraction = n0 / total if total else 0.0

    return ThresholdTopology(
        threshold=threshold,
        area=area,
        contour_n=contour_n,
        perimeter=perimeter,
        roughness=roughness,
        solidity=solidity,
        internal_dark_fraction=internal_dark_fraction,
    )
'''

TOPOLOGY_TRAJECTORY = r'''def topology_trajectory_from_separated_component(
    full_res_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> tuple[ThresholdTopology, ...]:
    """Measure exact seeded topology for T..T+max_delta in the base-component crop."""
    base_component = np.asarray(base_component, dtype=bool)
    if base_component.ndim != 2 or not np.any(base_component):
        raise ValueError("base component must be a non-empty two-dimensional mask")
    if not 0 <= base_threshold <= 255:
        raise ValueError("base threshold must be 0..255")
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative")

    ys, xs = np.nonzero(base_component)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    base_crop = base_component[y0:y1, x0:x1]
    gray_crop = full_res_gray[y0:y1, x0:x1]
    seed_x, seed_y = seed_point
    sx = seed_x - x0
    sy = seed_y - y0
    if not (0 <= sx < gray_crop.shape[1] and 0 <= sy < gray_crop.shape[0]):
        raise ValueError("seed lies outside base component crop")

    trajectory: list[ThresholdTopology] = []
    max_threshold = min(255, base_threshold + max_delta)
    for threshold in range(base_threshold, max_threshold + 1):
        light = base_crop & (gray_crop > threshold)
        if not light[sy, sx]:
            break

        flood = np.where(light, 255, 0).astype(np.uint8)
        cv2.floodFill(flood, None, (sx, sy), 128, flags=8)
        component = flood == 128

        # Measure the fixed seed's connected component at this threshold.
        trajectory.append(_component_descriptor(component, threshold))

    if not trajectory:
        raise ValueError("no valid topology samples")
    return tuple(trajectory)
'''

SELECT_TOPOLOGY_KNEE = r'''def select_topology_knee(
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
    if base.area <= 0 or base.contour_n <= 0:
        raise ValueError("base topology must have positive area and contour length")

    base_area = base.area
    base_contour = base.contour_n
    base_roughness = base.roughness
    base_solidity = base.solidity
    solidity_headroom = 1.0 - base_solidity

    net: list[float] = []
    for row in rows:
        # Benefit terms count improvements only; worsening is not mislabeled as cleanup.
        contour_cleanup = max(0.0, 1.0 - row.contour_n / base_contour)
        roughness_cleanup = (
            max(0.0, 1.0 - row.roughness / base_roughness)
            if base_roughness > 0.0
            else 0.0
        )
        solidity_gain = (
            max(0.0, (row.solidity - base_solidity) / solidity_headroom)
            if solidity_headroom > 0.0
            else 0.0
        )
        benefit = (contour_cleanup + roughness_cleanup + solidity_gain) / 3.0

        # Cost terms likewise count erosion/damage only when it actually worsens.
        area_loss = max(0.0, 1.0 - row.area / base_area)
        solidity_loss = (
            max(0.0, (base_solidity - row.solidity) / base_solidity)
            if base_solidity > 0.0
            else 0.0
        )
        cost = (area_loss + solidity_loss) / 2.0
        net.append(benefit - cost)

    best_so_far = np.maximum.accumulate(np.asarray(net))
    best_value = np.max(best_so_far)
    if not math.isfinite(best_value) or best_value <= 0.0:
        selected_index = 0
        knee = np.zeros(len(rows))
    else:
        quality_progress = best_so_far / best_value
        threshold_progress = np.linspace(0.0, 1.0, len(rows))
        knee = quality_progress - threshold_progress
        selected_index = int(np.argmax(knee))

    selected = rows[selected_index]
    return ThresholdTopologySelection(
        threshold=selected.threshold,
        base_threshold=base.threshold,
        delta=selected.threshold - base.threshold,
        trajectory=rows,
        net_quality=tuple(net),
        knee_curve=tuple(knee.tolist()),
    )
'''

OPTIMIZE_SEPARATED = r'''def optimize_separated_threshold(
    full_res_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> ThresholdTopologySelection:
    """Optimize a proven separated threshold without ever lowering it."""
    # Measure the fixed seed's topology from the proven separated T upward.
    trajectory = topology_trajectory_from_separated_component(
        full_res_gray,
        base_threshold,
        seed_point,
        base_component,
        max_delta=max_delta,
    )

    # If only the input T survives, the knee selector returns that input unchanged.
    return select_topology_knee(trajectory)
'''

RESIZE_IMG = r'''def resize_img(
    img: np.ndarray,
    size: tuple[int | None, int | None],
) -> np.ndarray:
    """Resize to ``(width, height)``, inferring one omitted dimension if requested."""
    original_dtype = img.dtype
    original_height, original_width = img.shape[:2]
    width, height = size

    if width is None and height is None:
        raise ValueError("resize size must provide width, height, or both")
    if width is None:
        if height <= 0:
            raise ValueError("resize height must be positive")
        width = round(original_width * height / original_height)
    elif height is None:
        if width <= 0:
            raise ValueError("resize width must be positive")
        height = round(original_height * width / original_width)

    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")

    is_binary = np.all((img == img.min()) | (img == img.max()))
    if is_binary:
        # OpenCV does not resize boolean arrays, but integer binary rasters must keep
        # their exact original values instead of being truncated through uint8.
        resize_source = img.astype(np.uint8) if img.dtype == bool else img
        interpolation = cv2.INTER_NEAREST_EXACT
    elif width < original_width:
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
'''

NEAREST_ODD = r'''def nearest_positive_odd(value: float) -> int:
    """Return the nearest positive odd integer; exact ties choose the lower odd."""
    if value <= 0:
        raise ValueError("value must be positive")
    return 2 * math.ceil(value / 2) - 1
'''

HISTOGRAM = r'''def find_histogram_start_threshold(work_res_gray: np.ndarray) -> int:
    """Return the left valley preceding the rightmost 3-bin-smoothed histogram mode."""
    histogram = np.bincount(work_res_gray.ravel(), minlength=256).astype(np.float64)
    signal = np.convolve(histogram, PEAK_KERNEL, mode="same")

    rightmost_peak = None
    for index in range(1, len(signal) - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            rightmost_peak = index
    if len(signal) >= 2 and signal[-1] > signal[-2]:
        rightmost_peak = len(signal) - 1
    if rightmost_peak is None:
        rightmost_peak = int(np.argmax(signal))

    for index in range(rightmost_peak - 1, 0, -1):
        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:
            return index
    return 0
'''

FIND_WORK = r'''def find_work_res_solar_component(
    work_res_gray: np.ndarray,
    start_T: int,
    work_res_seed_kernel: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Establish one supported work-res seed, then track only that identity downward."""
    seed: tuple[int, int] | None = None
    work_res_T: int | None = None
    work_res_component: np.ndarray | None = None

    for threshold in range(start_T, -1, -1):
        if seed is None:
            # Before identity is established, propose the largest enclosed bright component.
            component = largest_enclosed_bright_component(work_res_gray > threshold)
            if component is not None:
                # Accept the first candidate with the required full seed-support footprint.
                seed = brightest_supported_component_point(
                    work_res_gray,
                    component,
                    work_res_seed_kernel,
                )
                if seed is not None:
                    work_res_T = threshold
                    work_res_component = component
        else:
            binary = work_res_gray > threshold
            flooded = np.where(binary, 255, 0).astype(np.uint8)
            cv2.floodFill(flooded, None, seed, 128, flags=8)
            component = flooded == 128

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
        raise ThresholdResolutionError(
            "No 5x5-supported enclosed bright component exists through T=0"
        )

    return work_res_T, work_res_component
'''

DILATE = r'''def dilate_component_mask(component_mask: np.ndarray, margin: float) -> np.ndarray:
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
'''

FIND_LOWEST_FULL = r'''def find_lowest_full_res_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Find the lowest enclosed full-res T by searching only the monotonic direction."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if not 0 <= start_T <= 255:
        raise ValueError("start threshold must be 0..255")

    guard = np.asarray(full_res_guard_mask, dtype=bool)
    if guard.shape != full_res_gray.shape:
        raise ValueError("full-resolution guard and grayscale image must have identical shapes")
    if not np.any(guard):
        raise ThresholdResolutionError("Full-resolution Auto-T guard is empty")

    seed_x, seed_y = full_res_seed
    height, width = full_res_gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the image")
    if not guard[seed_y, seed_x]:
        raise ThresholdResolutionError("Full-resolution tracking seed lies outside the Auto-T guard")

    guard_u8 = np.where(guard, 255, 0).astype(np.uint8)

    # Build the fixed one-pixel guard boundary once before either directional loop.
    boundary_kernel = generate_kernel((3, 3), round_kernel=False)
    eroded_guard = cv2.erode(
        guard_u8,
        boundary_kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
    full_res_guard_boundary = guard & ~eroded_guard

    # Evaluate the starting work-resolution T on the authoritative source raster.
    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)
    cv2.bitwise_and(binary, guard_u8, dst=binary)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed is not light at start T={start_T}"
        )
    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    component = binary == 128
    enclosed = not np.any(component & full_res_guard_boundary)

    if enclosed:
        best_T = start_T
        best_component = component
        for threshold in range(start_T - 1, -1, -1):
            binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
            cv2.bitwise_and(binary, guard_u8, dst=binary)
            cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
            component = binary == 128
            if np.any(component & full_res_guard_boundary):
                break
            best_T = threshold
            best_component = component
        return best_T, best_component

    for threshold in range(start_T + 1, 256):
        binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
        cv2.bitwise_and(binary, guard_u8, dst=binary)
        if binary[seed_y, seed_x] == 0:
            break
        cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
        component = binary == 128
        if not np.any(component & full_res_guard_boundary):
            return threshold, component

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became enclosed by the 10% guard"
    )
'''

FIND_AUTO = r'''def find_auto_threshold(
    full_res_gray: np.ndarray,
    image_state: dict[str, object],
) -> int:
    """Determine Auto T and cache only search state; ``resolve_threshold`` owns SolarData."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")

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

    # Generate the fixed square work-resolution support kernel once and reuse it through the search.
    work_res_seed_kernel = generate_kernel((5, 5), round_kernel=False)

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
        )

        # Preserve the 5-pixel work support footprint at the realized source scale.
        mapped_kernel_size = 5 * max(full_res_gray.shape) / max(work_res_gray.shape)
        full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)

        # Generate the equivalent source support kernel once before selecting the fixed source seed.
        full_res_seed_kernel = generate_kernel(
            (full_res_kernel_size, full_res_kernel_size),
            round_kernel=False,
        )

        # Select the brightest deeply supported source seed anywhere in the transferred mature component.
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

        # Expand the transferred component by the fixed 10% L2 margin to make the source guard.
        image_scale = math.sqrt(full_res_width * full_res_height)
        full_res_guard_mask = dilate_component_mask(
            full_res_search_mask,
            AUTO_T_GUARD_DILATION_FRACTION * image_scale,
        )

        # Find the exact source separation boundary by searching only the required monotonic direction.
        full_res_T, full_res_component = find_lowest_full_res_threshold(
            full_res_gray,
            work_res_T,
            full_res_seed,
            full_res_guard_mask,
        )

        # Stage B may raise T to a topology knee but never lowers the proven separated threshold.
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
    return topology_selection.threshold
'''

CLEANUP_MORPHOLOGY = r'''def cleanup_morphology_candidates(component: np.ndarray) -> dict[str, np.ndarray]:
    """Build raw, direct 3/5/7, and progressive 3->5 / 3->5->7 candidates."""
    raw = np.asarray(component, dtype=bool)
    if raw.ndim != 2 or not np.any(raw):
        raise ValueError("component mask must be a non-empty two-dimensional mask")

    # Generate every reused Euclidean cleanup kernel once before applying any path.
    k3 = generate_kernel((3, 3), round_kernel=True)
    k5 = generate_kernel((5, 5), round_kernel=True)
    k7 = generate_kernel((7, 7), round_kernel=True)

    # Apply direct cleanup to raw, then reuse prior results for the progressive paths.
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

SEED_CONNECTED = r'''def seed_connected_cleanup_candidates(
    component: np.ndarray,
    seed_point: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Return each cleanup candidate's 8-component containing the authoritative seed."""
    raw = np.asarray(component, dtype=bool)
    if raw.ndim != 2 or not np.any(raw):
        raise ThresholdResolutionError("Raw cleanup component is empty")

    seed_x, seed_y = seed_point
    height, width = raw.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("Cleanup seed lies outside the component raster")
    if not raw[seed_y, seed_x]:
        raise ThresholdResolutionError("Cleanup seed lies outside the raw component")

    # Build the fixed raw/direct/progressive morphology candidate family once.
    candidates = cleanup_morphology_candidates(raw)
    valid: dict[str, np.ndarray] = {}
    for name in CLEANUP_CANDIDATE_ORDER:
        candidate = candidates[name]
        if not candidate[seed_y, seed_x]:
            continue
        flood = np.where(candidate, 255, 0).astype(np.uint8)
        cv2.floodFill(flood, None, (seed_x, seed_y), 128, flags=8)
        valid[name] = flood == 128

    return valid
'''

EVALUATE_CLEANUP = r'''def evaluate_cleanup_candidates(
    component: np.ndarray,
    seed_point: tuple[int, int],
    threshold: int,
) -> tuple[CleanupCandidateEvaluation, ...]:
    """Measure every valid cleanup candidate against the same raw component."""
    # Restrict every morphology candidate to the authoritative seed-connected identity.
    candidates = seed_connected_cleanup_candidates(component, seed_point)

    # Measure the unmodified component once; every candidate is scored against this baseline.
    raw_topology = _component_descriptor(candidates["raw"], threshold)
    base_area = raw_topology.area
    base_contour = raw_topology.contour_n
    base_roughness = raw_topology.roughness
    base_solidity = raw_topology.solidity
    solidity_headroom = 1.0 - base_solidity

    rows: list[CleanupCandidateEvaluation] = []
    for name in CLEANUP_CANDIDATE_ORDER:
        candidate = candidates.get(name)
        if candidate is None:
            continue

        # Reuse the already measured raw descriptor; measure only actual cleanup variants.
        topology = raw_topology if name == "raw" else _component_descriptor(candidate, threshold)

        # Benefit terms count only actual cleanup improvements.
        contour_cleanup = max(0.0, 1.0 - topology.contour_n / base_contour)
        roughness_cleanup = (
            max(0.0, 1.0 - topology.roughness / base_roughness)
            if base_roughness > 0.0
            else 0.0
        )
        solidity_gain = (
            max(0.0, (topology.solidity - base_solidity) / solidity_headroom)
            if solidity_headroom > 0.0
            else 0.0
        )
        internal_dark_cleanup = max(
            0.0,
            raw_topology.internal_dark_fraction - topology.internal_dark_fraction,
        )

        # Cost terms count only actual loss relative to the raw component.
        area_loss = max(0.0, 1.0 - topology.area / base_area)
        solidity_loss = (
            max(0.0, (base_solidity - topology.solidity) / base_solidity)
            if base_solidity > 0.0
            else 0.0
        )
        rows.append(
            CleanupCandidateEvaluation(
                name=name,
                mask=candidate,
                topology=topology,
                metrics=CleanupMetrics(
                    contour_cleanup=contour_cleanup,
                    roughness_cleanup=roughness_cleanup,
                    solidity_gain=solidity_gain,
                    internal_dark_cleanup=internal_dark_cleanup,
                    area_loss=area_loss,
                    solidity_loss=solidity_loss,
                ),
            )
        )
    return tuple(rows)
'''

REFINE = r'''def refine_solar_component_mask(component: np.ndarray) -> np.ndarray:
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
'''

DECOMPRESS = r'''def decompress_full_mask(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
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
'''

RESOLVE = r'''def resolve_threshold(
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
'''

BUILD_SETTINGS = r'''    def _build_settings_panel(self):
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
'''

ADD_SLIDER = r'''    def _add_slider(
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
'''

FORMAT_SLIDER = r'''    def _format_slider_value(self, setting_name, value) -> str:
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
'''

CHECKBOX_HANDLERS = r'''    def _handle_morphology_change(self):
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
'''

APPLY_FULL = r'''    def apply_full_resolution(self, _event=None):
        # Reuse the same authoritative preview/resolution path with full-resolution status text.
        self.refresh_preview(full_resolution=True)
'''

RENDER_CANVAS = r'''    def render_canvas_content(self, canvas, content):
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
'''

CLOSE = r'''    def close(self, _event=None):
        self.root.destroy()
'''


def apply_production_cleanup(source: str) -> str:
    # Move/generate the generic kernel function before constants that instantiate kernels.
    source = replace_module_function(source, 'generate_kernel', '')
    marker = '# ---------------------------------------------------------------------------\n# Per-image processing settings\n'
    if marker not in source:
        raise RuntimeError('settings marker not found')
    source = source.replace(marker, GENERATE_KERNEL + '\n\n' + marker, 1)

    replacements = {
        'transparent_bgra': TRANSPARENT_BGRA,
        'opaque_bgra': OPAQUE_BGRA,
        '_component_descriptor': COMPONENT_DESCRIPTOR,
        'topology_trajectory_from_separated_component': TOPOLOGY_TRAJECTORY,
        'select_topology_knee': SELECT_TOPOLOGY_KNEE,
        'optimize_separated_threshold': OPTIMIZE_SEPARATED,
        'resize_img': RESIZE_IMG,
        'find_histogram_start_threshold': HISTOGRAM,
        'find_work_res_solar_component': FIND_WORK,
        'dilate_component_mask': DILATE,
        'find_lowest_full_res_threshold': FIND_LOWEST_FULL,
        'find_auto_threshold': FIND_AUTO,
        'cleanup_morphology_candidates': CLEANUP_MORPHOLOGY,
        'seed_connected_cleanup_candidates': SEED_CONNECTED,
        'evaluate_cleanup_candidates': EVALUATE_CLEANUP,
        'refine_solar_component_mask': REFINE,
        'decompress_full_mask': DECOMPRESS,
        'resolve_threshold': RESOLVE,
    }
    for name, replacement in replacements.items():
        source = replace_module_function(source, name, replacement)

    # nearest_positive_odd belongs with resize/search primitives and has two real callers.
    resize_end = RESIZE_IMG.rstrip() + '\n'
    if source.count(resize_end) != 1:
        raise RuntimeError('updated resize_img block not found exactly once')
    source = source.replace(resize_end, resize_end + '\n\n' + NEAREST_ODD.rstrip() + '\n', 1)

    old_constants = '''WORK_MAX_DIM = 1200\nPEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)\nROI_DILATION_FRACTION = 0.065\nGUARD_DILATION_FRACTION = 0.195\nAUTO_T_GUARD_DILATION_FRACTION = 0.10\nTOPOLOGY_OPTIMIZATION_STEPS = 5\nTRACKING_SEED_KERNEL_SIZE = 5\nSOLAR_COMPONENT_KERNEL_SIZE = 7\nSOLAR_COMPONENT_KERNEL = cv2.getStructuringElement(\n    cv2.MORPH_ELLIPSE,\n    (SOLAR_COMPONENT_KERNEL_SIZE, SOLAR_COMPONENT_KERNEL_SIZE),\n)\nREFINEMENT_ITERATIONS = 1'''
    new_constants = '''WORK_RES_MAX_DIM = 1200\nPEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)\nROI_DILATION_FRACTION = 0.065\nGUARD_DILATION_FRACTION = 0.195\nAUTO_T_GUARD_DILATION_FRACTION = 0.10\nTOPOLOGY_OPTIMIZATION_STEPS = 5\n\n# Build the fixed final-T Euclidean cleanup kernel once and reuse it.\nSOLAR_COMPONENT_KERNEL = generate_kernel((7, 7), round_kernel=True)'''
    if source.count(old_constants) != 1:
        raise RuntimeError(f'constant block match count={source.count(old_constants)}')
    source = source.replace(old_constants, new_constants, 1)

    # The cleanup-size tuple was never used by production; keep only the candidate order.
    source = source.replace('CLEANUP_KERNEL_SIZES = (3, 5, 7)\n', '', 1)

    # Document the currently redundant explicit resolution state.
    old_result = '''    full_res_seed_point: tuple[int, int] | None\n    resolved: bool\n    reason: str = ""'''
    new_result = '''    full_res_seed_point: tuple[int, int] | None\n    # Currently equivalent to ``threshold is not None``. Kept explicit for now in\n    # case resolution-state semantics become independent later.\n    resolved: bool\n    reason: str = ""'''
    if source.count(old_result) != 1:
        raise RuntimeError('AutoThresholdResult field block not found')
    source = source.replace(old_result, new_result, 1)

    # GUI method replacements.
    source = replace_class_method(source, 'DetectorApp', '_build_settings_panel', BUILD_SETTINGS)
    source = replace_class_method(source, 'DetectorApp', '_add_slider', ADD_SLIDER)
    source = replace_class_method(source, 'DetectorApp', 'apply_full_resolution', APPLY_FULL)
    source = replace_class_method(source, 'DetectorApp', 'render_canvas_content', RENDER_CANVAS)
    source = replace_class_method(source, 'DetectorApp', 'close', CLOSE)

    # Insert the slider formatter after _add_slider and checkbox callbacks before center-target callback.
    source = replace_class_method(
        source,
        'DetectorApp',
        '_add_slider',
        ADD_SLIDER.rstrip() + '\n\n' + FORMAT_SLIDER.rstrip(),
    )
    center_marker = '    def _handle_center_target_change(self):\n'
    if source.count(center_marker) != 1:
        raise RuntimeError('center target handler marker not found')
    source = source.replace(center_marker, CHECKBOX_HANDLERS.rstrip() + '\n\n' + center_marker, 1)

    # Remove callback lambdas by allowing the existing actions to receive Tk events directly.
    source = source.replace(
        '        root.bind("<Return>", lambda _event: self.apply_full_resolution())\n'
        '        root.bind("<Escape>", lambda _event: self.close())',
        '        root.bind("<Return>", self.apply_full_resolution)\n'
        '        root.bind("<Escape>", self.close)',
        1,
    )

    # Typed Tk values need no round-trip coercion when stored or reused.
    source = source.replace('            min_radius=int(self.min_radius.get()),', '            min_radius=self.min_radius.get(),')
    source = source.replace('            max_radius=int(self.max_radius.get()),', '            max_radius=self.max_radius.get(),')
    source = source.replace('            max_error=float(self.max_error.get()),', '            max_error=self.max_error.get(),')
    source = source.replace('            min_coverage=int(self.min_coverage.get()),', '            min_coverage=self.min_coverage.get(),')
    source = source.replace('            morphology=bool(self.morphology.get()),', '            morphology=self.morphology.get(),')
    source = source.replace('            outer_limb_assistance=bool(self.outer_limb_assistance.get()),', '            outer_limb_assistance=self.outer_limb_assistance.get(),')
    source = source.replace('            use_horizon=bool(self.use_horizon.get()),', '            use_horizon=self.use_horizon.get(),')
    source = source.replace('            self.threshold.set(int(settings.threshold))', '            self.threshold.set(settings.threshold)')
    source = source.replace('            selected_threshold = int(result.threshold)', '            selected_threshold = result.threshold')
    source = source.replace('            threshold = int(value)', '            threshold = value')
    source = source.replace('        threshold = int(settings.threshold)', '        threshold = settings.threshold')
    source = source.replace('        raw_component = self.gray_image > threshold', '        threshold_mask = self.gray_image > threshold')
    source = source.replace('self.render_canvas_content(self.threshold_canvas, raw_component)', 'self.render_canvas_content(self.threshold_canvas, threshold_mask)')

    # Normalize resolution-qualified naming throughout production.
    source = source.replace('WORK_MAX_DIM', 'WORK_RES_MAX_DIM')

    # Current production should contain no lambda expressions after the GUI cleanup.
    if 'lambda ' in source:
        raise RuntimeError('lambda expression remains in production source')

    return source


def synchronize_retained_patch(text: str) -> str:
    """Make the historical streamlined patch delegate to the current cleanup pass.

    The large embedded strings intentionally reconstruct the pre-cleanup streamlined
    architecture from its historical base. Keeping a second fully duplicated copy of
    every later readability change would create two authoritative implementations.
    Instead, the retained script applies that historical structural step and then
    delegates the resulting source to ``apply_readability_cleanup.py``.
    """
    if "import runpy\n" not in text:
        text = text.replace("import re\n", "import re\nimport runpy\n", 1)

    old_main = (
        'def main() -> None:\n'
        '    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")\n'
        '    patch_source(target)'
    )
    new_main = (
        'def main() -> None:\n'
        '    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")\n\n'
        '    # Reconstruct the approved streamlined Auto-T architecture from its historical base.\n'
        '    patch_source(target)\n\n'
        '    # Apply the current readability/invariant cleanup once rather than duplicating\n'
        '    # that implementation inside this historical transformation script.\n'
        '    cleanup_path = Path(__file__).with_name("apply_readability_cleanup.py")\n'
        '    cleanup = runpy.run_path(str(cleanup_path))\n'
        '    source = target.read_text(encoding="utf-8")\n'
        '    target.write_text(\n'
        '        cleanup["apply_production_cleanup"](source),\n'
        '        encoding="utf-8",\n'
        '    )'
    )
    if text.count(old_main) != 1:
        raise RuntimeError("retained streamlined patch main block not found")
    text = text.replace(old_main, new_main, 1)

    marker = "GENERATE_KERNEL = r'''def generate_kernel"
    if marker in text and "Intermediate reconstruction only" not in text:
        text = text.replace(
            marker,
            "# Intermediate reconstruction only; main() delegates final cleanup afterward.\n" + marker,
            1,
        )
    return text

def main() -> None:
    source = SOURCE_PATH.read_text(encoding='utf-8')
    updated = apply_production_cleanup(source)
    ast.parse(updated)
    SOURCE_PATH.write_text(updated, encoding='utf-8')

    if RETAINED_PATCH_PATH.exists():
        retained = RETAINED_PATCH_PATH.read_text(encoding='utf-8')
        RETAINED_PATCH_PATH.write_text(synchronize_retained_patch(retained), encoding='utf-8')


if __name__ == '__main__':
    main()
