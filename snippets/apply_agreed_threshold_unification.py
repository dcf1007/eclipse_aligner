"""Apply the agreed threshold-flow unification to circle_arc_detector.py.

This is a reproducible local transformation for the threshold-finder branch. It
keeps Auto-T acquisition separate from final-T resolution, makes threshold state
explicit once initialized, uses one seed-selection rule, and makes final-T
resolution + SolarData persistence one atomic operation.
"""
from __future__ import annotations

from pathlib import Path
import re


def replace_function(text: str, name: str, replacement: str, indent: str = "") -> str:
    marker = f"{indent}def {name}("
    start = text.index(marker)
    # Next function/class at the same indentation level, or EOF.
    candidates = []
    for token in (f"\n{indent}def ", f"\n{indent}class "):
        pos = text.find(token, start + len(marker))
        if pos != -1:
            candidates.append(pos + 1)
    end = min(candidates) if candidates else len(text)
    return text[:start] + replacement.rstrip() + "\n\n\n" + text[end:]


def replace_method(text: str, name: str, replacement: str) -> str:
    return replace_function(text, name, replacement, indent="    ")


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[:start] + replacement.rstrip() + "\n\n\n" + text[end:]


def apply(source: str) -> str:
    text = source

    # Current threshold-finder architecture: explicit threshold state, no fixed
    # refinement-display timer, and one shared 7x7 elliptical component kernel.
    text = text.replace(
        'SLIDER_KEY_RELEASE_SETTLE_MS = 45\nPREVIEW_REDRAW_DELAY_MS = 60\n',
        'SLIDER_KEY_RELEASE_SETTLE_MS = 45\nCANVAS_REDRAW_DELAY_MS = 60\n',
    )
    text = text.replace(
        'class ImageSettings:\n    """Sparse per-image overrides; ``None`` means use that setting\'s baseline."""',
        'class ImageSettings:\n    """Per-image settings. Threshold ``None`` means never initialized; other ``None`` values use defaults."""',
    )

    constants_old = (
        'WORK_MAX_DIM = 1200\n'
        'PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)\n'
        'ROI_DILATION_FRACTION = 0.065\n'
        'GUARD_DILATION_FRACTION = 0.195\n\n\n'
        'class ThresholdResolutionError(RuntimeError):'
    )
    constants_new = '''WORK_MAX_DIM = 1200
PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)
ROI_DILATION_FRACTION = 0.065
GUARD_DILATION_FRACTION = 0.195
TOPOLOGY_OPTIMIZATION_STEPS = 5
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
    return ThresholdTopology(
        threshold=int(threshold),
        area=area,
        contour_n=contour_n,
        perimeter=perimeter,
        roughness=roughness,
        solidity=solidity,
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


class ThresholdResolutionError(RuntimeError):'''
    if constants_old not in text:
        raise RuntimeError("threshold constants anchor not found")
    text = text.replace(constants_old, constants_new, 1)

    # Obsolete deepest-only seed selection is not used anywhere in production and
    # would violate the single seed-selection rule if reused later.
    text = replace_function(text, "deepest_component_point", "")

    text = replace_function(text, "brightest_supported_component_point", '''def brightest_supported_component_point(
    gray: np.ndarray,
    component_u8: np.ndarray,
) -> tuple[int, int]:
    """Choose the brightest 7x7-supported pixel; depth breaks brightness ties.

    Eligibility requires surviving the same 7x7 elliptical erosion used by the
    subsequent solar-component refinement. There is deliberately no unsupported
    fallback: a component with no robust interior cannot supply an authoritative
    or tracking seed.
    """
    source = (component_u8 != 0).astype(np.uint8)
    if gray.shape != source.shape:
        raise ValueError("gray and component must have identical shapes")
    if not np.any(source):
        raise ThresholdResolutionError("Empty component")

    supported = cv2.erode(
        source,
        SOLAR_COMPONENT_KERNEL,
        iterations=1,
    ) != 0
    if not np.any(supported):
        raise ThresholdResolutionError(
            "Solar component has no 7x7-supported interior seed"
        )

    max_gray = int(gray[supported].max())
    brightest = supported & (gray == max_gray)

    # The 5 here is OpenCV's L2 distance-transform approximation mask size; it is
    # unrelated to the 7x7 support kernel above.
    distance = cv2.distanceTransform(source, cv2.DIST_L2, 5)
    scores = np.where(brightest, distance, -1.0)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return int(x), int(y)''')

    text = replace_function(text, "auto_threshold", '''def find_auto_threshold(
    gray: np.ndarray,
    image_state: dict[str, object],
) -> int:
    """Determine automatic T, store AutoThresholdResult, and return the selected T.

    Auto-T owns threshold acquisition only. Components and seeds used here are
    search/tracking data. The definitive current-T component, authoritative seed,
    refinement, and SolarData are established later by ``resolve_threshold``.
    """
    work_gray = resize_gray_max_dim(gray)
    peak, left_edge = find_rightmost_histogram_peak(work_gray)

    try:
        coarse = coarse_threshold_search(work_gray)
        full_seed = establish_full_resolution_seed(
            gray, work_gray.shape, coarse.seed_mask, coarse.seed_threshold
        )
        roi_seed_threshold, roi_seed_component = (
            find_full_resolution_enclosed_seed_component(
                gray, coarse.threshold, full_seed
            )
        )
        separated_threshold, used_guard = find_lowest_full_threshold(
            gray,
            full_seed,
            roi_seed_component,
        )

        separated_binary = cv2.compare(gray, int(separated_threshold), cv2.CMP_GT)
        seed_x, seed_y = map(int, full_seed)
        if separated_binary[seed_y, seed_x] == 0:
            raise ThresholdResolutionError(
                f"Auto-T tracking seed is not light at separated T={separated_threshold}"
            )
        cv2.floodFill(separated_binary, None, (seed_x, seed_y), 128, flags=8)
        separated_component = separated_binary == 128
        if _touches_image_border(separated_component):
            raise ThresholdResolutionError(
                f"Auto-T solar component touches the image border at T={separated_threshold}"
            )

        topology_selection = optimize_separated_threshold(
            gray,
            separated_threshold,
            full_seed,
            separated_component,
        )
        result = AutoThresholdResult(
            threshold=topology_selection.threshold,
            histogram_peak=coarse.histogram_peak,
            histogram_left_edge=coarse.histogram_left_edge,
            seed_threshold=coarse.seed_threshold,
            coarse_threshold=coarse.threshold,
            roi_seed_threshold=roi_seed_threshold,
            full_seed_point=full_seed,
            used_guard=used_guard,
            resolved=True,
        )
    except ThresholdResolutionError as exc:
        result = AutoThresholdResult(
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

    image_state["auto_threshold_result"] = result
    return int(result.threshold)''')

    # Replace the complete post-threshold section. Resolution and SolarData writing
    # are one atomic stage, as agreed; there is no separate build_solar_data call.
    post = '''# ---------------------------------------------------------------------------
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


def _full_mask_from_observation_region(
    full_gray: np.ndarray,
    component: np.ndarray,
    margin: float,
    seed_point: tuple[int, int],
) -> np.ndarray:
    """Promote the existing cropped Euclidean observation dilation to full size."""
    region = _build_observation_region(full_gray, component, margin, seed_point)
    full_mask = np.zeros(full_gray.shape, dtype=bool)
    x0, y0, x1, y1 = region.bbox
    full_mask[y0:y1, x0:x1] = region.allowed_u8 != 0
    return full_mask


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
    raw_u8 = np.where(raw_component, 255, 0).astype(np.uint8)
    seed_x, seed_y = brightest_supported_component_point(full_gray, raw_u8)
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
    roi_6_5 = _full_mask_from_observation_region(
        full_gray,
        refined_component,
        ROI_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
    )
    guard_19_5 = _full_mask_from_observation_region(
        full_gray,
        refined_component,
        GUARD_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
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
    return refined_component'''
    text = replace_block(
        text,
        'class SolarData:',
        'class DetectorApp:',
        post,
    )

    # No application-level duplicate threshold raster state. The canvas renderer
    # retains only its last supplied display content for resize redraws.
    text = text.replace('        self.threshold_preview = transparent_bgra()\n', '')
    text = text.replace('        self.threshold_photo = None\n', '')
    text = text.replace('        self.color_photo = None\n', '')
    text = text.replace('        self.threshold_refinement_job = None\n', '')

    # Comments describing the old Auto-T baseline semantics are no longer true.
    text = text.replace(
        '        # Every processing control is per-image. Ordinary controls use these values\n'
        '        # as their baseline; threshold instead uses the cached automatic T for the\n'
        '        # current image. Only deviations from those baselines are stored.\n',
        '        # Ordinary controls use these values as sparse baselines. Threshold is\n'
        '        # different: once initialized, its exact current integer is always stored.\n',
    )

    # Loading an image restores an explicit T, or initializes it through Auto T.
    text = replace_method(text, "load_images", '''    def load_images(self):
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
        self.load_image_at(0)''')

    text = replace_method(text, "load_image_at", '''    def load_image_at(self, index: int):
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
            threshold = find_auto_threshold(self.gray_image, state)
            self.commit_setting_change("threshold", threshold)
        else:
            self.threshold.set(int(settings.threshold))
            self.refresh_preview(changed_setting="image load")

        self.update_navigation_state()''')

    # All actual setting changes now pass the new value through one public method.
    text = text.replace('self._commit_setting_change(event.widget._setting_name)',
                        'self.commit_setting_change(event.widget._setting_name, event.widget.get())')
    text = text.replace('command=lambda: self._commit_setting_change("morphology")',
                        'command=lambda: self.commit_setting_change("morphology", self.morphology.get())')
    text = text.replace('command=lambda: self._commit_setting_change("outer_limb_assistance")',
                        'command=lambda: self.commit_setting_change("outer_limb_assistance", self.outer_limb_assistance.get())')
    text = text.replace('command=lambda: self._commit_setting_change("use_horizon")',
                        'command=lambda: self.commit_setting_change("use_horizon", self.use_horizon.get())')

    text = replace_method(text, "auto_select_threshold", '''    def auto_select_threshold(self):
        """Restore/recompute image-only Auto T, then commit it like any other T change."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        result = state.get("auto_threshold_result")
        if isinstance(result, AutoThresholdResult):
            selected_threshold = int(result.threshold)
        else:
            selected_threshold = find_auto_threshold(self.gray_image, state)
            result = state["auto_threshold_result"]

        self.commit_setting_change("threshold", selected_threshold)

        solar_data = state.get("solar_data")
        if isinstance(solar_data, SolarData) and solar_data.threshold == selected_threshold:
            if result.resolved:
                self.status.set(
                    "Automatic grayscale threshold selected: "
                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                    f"histogram start={result.histogram_left_edge}); refined component displayed."
                )
            else:
                self.status.set(
                    "Automatic component tracking unresolved; "
                    f"using rightmost-histogram left edge T={selected_threshold}; "
                    "SolarData established directly at that T."
                )''')

    # Remove old threshold-specific refresh helpers and use the agreed single path.
    for obsolete in ("_refresh_threshold_image", "_ensure_solar_data_for_current_threshold"):
        if f"    def {obsolete}(" in text:
            text = replace_method(text, obsolete, "")

    if "    def _refresh_display_images(" in text:
        text = replace_method(text, "_refresh_display_images", "")

    text = replace_method(text, "_commit_setting_change", '''    def commit_setting_change(self, setting_name, value):
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
            self.refresh_preview(changed_setting=setting_name)''')

    text = replace_method(text, "_handle_center_target_change", '''    def _handle_center_target_change(self):
        self._update_center_preview_label()
        self.commit_setting_change("center_target", self.center_target.get())
        self.status.set(
            f"Centering target set to {self._selected_center_target_name()}. "
            "Actual centering will be implemented with ellipse detection."
        )''')

    text = replace_method(text, "refresh_preview", '''    def refresh_preview(self, changed_setting: str | None = None, full_resolution: bool = False):
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
            )''')

    text = replace_method(text, "apply_full_resolution", '''    def apply_full_resolution(self):
        if self.gray_image is None or self.current_path is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return
        self.refresh_preview(full_resolution=True)''')

    if "    def _schedule_preview_redraw(" in text:
        text = replace_method(text, "_schedule_preview_redraw", '''    def _schedule_canvas_redraw(self, _event=None):
        if self.canvas_redraw_job is not None:
            self.root.after_cancel(self.canvas_redraw_job)
        self.canvas_redraw_job = self.root.after(
            CANVAS_REDRAW_DELAY_MS, self._redraw_cached_canvases
        )''')

    text = replace_method(text, "_redraw_previews", '''    def _redraw_cached_canvases(self):
        """Refit each canvas's retained unscaled raster after a canvas resize."""
        self.canvas_redraw_job = None
        for canvas_name in ("threshold_canvas", "color_canvas"):
            if not hasattr(self, canvas_name):
                continue
            canvas = getattr(self, canvas_name)
            unscaled_raster = getattr(canvas, "_unscaled_render_raster", None)
            if unscaled_raster is not None:
                self.render_canvas_content(canvas, unscaled_raster)''')

    renderer_source = '''    def render_canvas_content(self, canvas, content):
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

        # Canvas-owned display cache: normalized pixels before fitting to the
        # current canvas size, regardless of which path supplied the content.
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
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        scaled_raster = cv2.resize(
            unscaled_render_raster, fitted_size, interpolation=interpolation
        )
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
        canvas._tk_photo_image = tk_photo
        self.root.update_idletasks()'''

    if '    def _show_image_on_canvas(' in text:
        text = replace_method(text, "_show_image_on_canvas", renderer_source)
    else:
        text = replace_method(text, "display_on_canvas", renderer_source)

    # Remove any old renderer calls that survived targeted method replacement.
    text = text.replace('self._show_image_on_canvas(', 'self.render_canvas_content(')

    # Update old method-name references if present anywhere else.
    text = text.replace('self._commit_setting_change(', 'self.commit_setting_change(')

    # Update module description only where it directly contradicts the agreed flow.
    text = text.replace('``settings`` stores sparse ``ImageSettings`` overrides,',
                        '``settings`` stores per-image ``ImageSettings`` values/overrides,')
    text = text.replace(
        'the threshold uses the cached automatic threshold as its image-specific baseline.\n'
        'Returning a control to its baseline removes that key from ``settings``.',
        'the threshold is always stored explicitly once initialized. Ordinary controls may\n'
        'still remove an override when returned to their application baseline.',
    )

    text = text.replace("self.preview_redraw_job", "self.canvas_redraw_job")
    text = text.replace("self._schedule_preview_redraw", "self._schedule_canvas_redraw")
    text = text.replace("self._redraw_previews", "self._redraw_cached_canvases")
    text = text.replace("self.display_on_canvas(", "self.render_canvas_content(")

    return text


def main() -> None:
    source_path = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
    source = source_path.read_text(encoding="utf-8")
    updated = apply(source)
    source_path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
