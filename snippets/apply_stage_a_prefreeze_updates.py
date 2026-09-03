#!/usr/bin/env python3
"""Apply the final pre-freeze Stage-A/input contract updates.

This transformation starts from the finalized Stage-A layout. It makes mask
resampling explicit, changes Stage A to return only its minimum defensible T after
one 7x7 Euclidean OPEN/CLOSE separation cleanup, hands T/seed/guard to Stage B,
and normalizes loaded source images to a lossless uint16 BGRA master while deriving
one fixed-range uint8 grayscale raster for processing.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys


def _replace_top_level_function(source: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1 : node.end_lineno] = [replacement.rstrip() + "\n"]
    return "".join(lines)


def _replace_method(source: str, class_name: str, name: str, replacement: str) -> str:
    tree = ast.parse(source)
    cls = next(
        item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    node = next(
        item
        for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    lines = source.splitlines(keepends=True)
    indented = "\n".join("    " + line if line else line for line in replacement.rstrip().split("\n")) + "\n"
    lines[node.lineno - 1 : node.end_lineno] = [indented]
    return "".join(lines)


def apply_prefreeze_updates(source: str) -> str:
    resize_function = '''def resize_img(
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
'''
    source = _replace_top_level_function(source, "resize_img", resize_function)

    marker = "\ndef nearest_positive_odd(value: float) -> int:\n"
    master_helpers = '''\ndef normalize_master_bgra16(image: np.ndarray) -> np.ndarray:
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
'''
    if marker not in source:
        raise RuntimeError("nearest_positive_odd marker changed")
    source = source.replace(marker, master_helpers + marker, 1)

    threshold_function = '''def find_lowest_full_res_threshold(
    full_res_gray: np.ndarray,
    start_T: int,
    full_res_seed: tuple[int, int],
    full_res_guard_mask: np.ndarray,
) -> int:
    """Return the lowest T whose D7-cleaned seeded component remains separated."""
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

    # Build the fixed guard boundary and strongest Stage-A cleanup kernel once.
    boundary_kernel = generate_kernel((3, 3), round_kernel=False)
    separation_cleanup_kernel = generate_kernel((7, 7), round_kernel=True)
    eroded_guard = cv2.erode(
        guard_u8,
        boundary_kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) != 0
    full_res_guard_boundary = guard & ~eroded_guard

    # Evaluate the starting T after the same maximum cleanup allowed by Stage A.
    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed is not light at start T={start_T}"
        )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        separation_cleanup_kernel,
        iterations=1,
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        separation_cleanup_kernel,
        iterations=1,
    )
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Full-resolution tracking seed does not survive D7 cleanup at start T={start_T}"
        )
    cv2.bitwise_and(binary, guard_u8, dst=binary)
    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    component = binary == 128
    enclosed = not np.any(component & full_res_guard_boundary)

    if enclosed:
        best_T = start_T
        for threshold in range(start_T - 1, -1, -1):
            binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                separation_cleanup_kernel,
                iterations=1,
            )
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                separation_cleanup_kernel,
                iterations=1,
            )
            cv2.bitwise_and(binary, guard_u8, dst=binary)
            cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
            component = binary == 128
            if np.any(component & full_res_guard_boundary):
                break
            best_T = threshold
        return best_T

    for threshold in range(start_T + 1, 256):
        binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            separation_cleanup_kernel,
            iterations=1,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            separation_cleanup_kernel,
            iterations=1,
        )
        if binary[seed_y, seed_x] == 0:
            break
        cv2.bitwise_and(binary, guard_u8, dst=binary)
        cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
        component = binary == 128
        if not np.any(component & full_res_guard_boundary):
            return threshold

    raise ThresholdResolutionError(
        "Tracked full-resolution solar component never became separated after D7 cleanup"
    )
'''
    source = _replace_top_level_function(source, "find_lowest_full_res_threshold", threshold_function)

    trajectory_function = '''def topology_trajectory_from_separation_threshold(
    full_res_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> tuple[ThresholdTopology, ...]:
    """Measure the current D7-cleaned Stage-B baseline from T through T+max_delta."""
    if full_res_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")
    if not 0 <= base_threshold <= 255:
        raise ValueError("base threshold must be 0..255")
    if max_delta < 0:
        raise ValueError("max_delta must be non-negative")

    guard = np.asarray(full_res_guard_mask, dtype=bool)
    if guard.shape != full_res_gray.shape or not np.any(guard):
        raise ValueError("Stage-B guard must be a non-empty mask matching grayscale")
    seed_x, seed_y = seed_point
    height, width = full_res_gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height) or not guard[seed_y, seed_x]:
        raise ValueError("Stage-B seed must lie inside the guard")

    guard_u8 = np.where(guard, 255, 0).astype(np.uint8)
    cleanup_kernel = generate_kernel((7, 7), round_kernel=True)
    trajectory: list[ThresholdTopology] = []
    max_threshold = min(255, base_threshold + max_delta)
    for threshold in range(base_threshold, max_threshold + 1):
        binary = cv2.compare(full_res_gray, threshold, cv2.CMP_GT)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
        if binary[seed_y, seed_x] == 0:
            break
        cv2.bitwise_and(binary, guard_u8, dst=binary)
        cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
        component = binary == 128

        # Measure a fresh Stage-B component at each T; Stage A provides no reference mask.
        trajectory.append(_component_descriptor(component, threshold))

    if not trajectory:
        raise ValueError("no valid topology samples")
    return tuple(trajectory)
'''
    source = _replace_top_level_function(
        source,
        "topology_trajectory_from_separated_component",
        trajectory_function,
    )

    optimizer_function = '''def optimize_separated_threshold(
    full_res_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    full_res_guard_mask: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> ThresholdTopologySelection:
    """Run the current Stage-B baseline from Stage A's T/seed/guard contract."""
    # Rebuild each topology sample from grayscale, T, seed, and guard; no Stage-A
    # component is authoritative or passed across the stage boundary.
    trajectory = topology_trajectory_from_separation_threshold(
        full_res_gray,
        base_threshold,
        seed_point,
        full_res_guard_mask,
        max_delta=max_delta,
    )

    # If only the input T survives, the knee selector returns that input unchanged.
    return select_topology_knee(trajectory)
'''
    source = _replace_top_level_function(source, "optimize_separated_threshold", optimizer_function)

    source = source.replace(
        """        full_res_search_mask = resize_img(\n            work_res_component,\n            (full_res_width, full_res_height),\n        )\n""",
        """        full_res_search_mask = resize_img(\n            work_res_component,\n            (full_res_width, full_res_height),\n            mask=True,\n        )\n""",
        1,
    )
    source = source.replace(
        """        full_res_T, full_res_component = find_lowest_full_res_threshold(\n            full_res_gray,\n            work_res_T,\n            full_res_seed,\n            full_res_guard_mask,\n        )\n""",
        """        full_res_T = find_lowest_full_res_threshold(\n            full_res_gray,\n            work_res_T,\n            full_res_seed,\n            full_res_guard_mask,\n        )\n""",
        1,
    )
    source = source.replace(
        """    # Stage A is complete: these are the proven source-separation T, component, and seed.\n    # Stage B may raise T to an optimization knee but must never lower this Stage-A result.\n    topology_selection = optimize_separated_threshold(\n        full_res_gray,\n        full_res_T,\n        full_res_seed,\n        full_res_component,\n    )\n""",
        """    # Stage A is complete: only the minimum separation T, fixed seed, and fixed guard cross the boundary.\n    # Stage B rebuilds and evaluates its own components and may raise T, but never lower Stage A's result.\n    topology_selection = optimize_separated_threshold(\n        full_res_gray,\n        full_res_T,\n        full_res_seed,\n        full_res_guard_mask,\n    )\n""",
        1,
    )

    init_method = '''def __init__(self, root: tk.Tk, image_paths: list[str], args: argparse.Namespace):
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
'''
    source = _replace_method(source, "DetectorApp", "__init__", init_method)

    load_images_method = '''def load_images(self):
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
'''
    source = _replace_method(source, "DetectorApp", "load_images", load_images_method)

    load_method = '''def load_image_at(self, index: int):
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
'''
    source = _replace_method(source, "DetectorApp", "load_image_at", load_method)

    nav_method = '''def update_navigation_state(self):
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
'''
    source = _replace_method(source, "DetectorApp", "update_navigation_state", nav_method)

    source = source.replace(
        "Stage A establishes one authoritative full-resolution seed-connected component\n"
        "# and the lowest threshold at which that component remains separated inside one\n"
        "# fixed 10% L2 guard. Stage A does not perform topology/morphology optimization.",
        "Stage A establishes one fixed full-resolution seed/guard pair and the lowest T at\n"
        "# which a D7-cleaned seed-connected component remains separated. The temporary\n"
        "# proof component is discarded; Stage B rebuilds every component it evaluates.",
        1,
    )

    return source


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")
    target.write_text(apply_prefreeze_updates(target.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
