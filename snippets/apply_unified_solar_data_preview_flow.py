from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"


def replace_block(text: str, start: str, end: str, replacement: str) -> str:
    """Replace from start through just before end; the end marker is preserved."""
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one occurrence, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


source = SOURCE.read_text()

source = replace_once(
    source,
    """A completed
setting change calls ``_commit_setting_change(setting_name)``; that function updates
only the changed sparse override and then invokes ``_refresh_threshold_image()``.
That lightweight path performs only the current-T B/W conversion and threshold-pane
update. It does not run the broader Refresh Preview processing path.

``refresh_preview()`` is the explicit processing boundary invoked by the user. A
readable image load only draws the image panes after automatic thresholding; if
automatic thresholding resolves, SolarData is built as a separate downstream stage
after ``AutoThresholdResult`` has already been returned. If automatic thresholding
is unresolved, image load stops after the fallback B/W preview and waits for an
explicit Refresh Preview. Later ellipse, arc, and horizon preview processing belongs
behind that same explicit boundary. Apply Full Resolution remains a
separate explicit action.""",
    """A completed
setting change calls ``_commit_setting_change(setting_name)``. Non-threshold controls
only persist their sparse override. A completed threshold change first displays the
pure ``gray > T`` raster, then immediately establishes full-resolution SolarData at
that exact T and replaces the threshold canvas with the finalized refined component.

``refresh_preview()`` remains the explicit boundary for later ellipse, arc, and
horizon processing, but threshold/SolarData establishment is already complete before
those later stages begin. Image load follows the same two-stage threshold display:
pure threshold first, finalized refined component second. Apply Full Resolution remains
a separate explicit action.""",
)

# The old public split between seeded flooding, fixed-T identity establishment, and
# SolarData construction is replaced by one coherent post-T stage below.
source = replace_block(
    source,
    "def solar_component_from_seed_at_threshold(\n",
    "def _full_mask_from_observation_region(\n",
    "",
)

new_build = '''def build_solar_data_at_threshold(
    full_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Establish, refine, and persist SolarData at exactly one already-selected T.

    If this is the resolved automatic threshold, its full-resolution seed is an
    invariant and is validated rather than recalculated. At any other T, establish
    the solar identity at that exact threshold and calculate a new full-resolution
    seed before extracting the component. The raw seeded component is internal only;
    the returned mask and every persisted SolarData geometry derive from the same
    7x7 open-then-close refined component.
    """
    threshold = int(threshold)
    if full_gray.ndim != 2:
        raise ValueError("grayscale image must be two-dimensional")

    existing = image_state.get("solar_data")
    if isinstance(existing, SolarData) and existing.threshold == threshold:
        return decompress_full_mask(existing.component_mask, full_gray.shape)

    auto_result = image_state.get("auto_threshold_result")
    if not isinstance(auto_result, AutoThresholdResult):
        raise ValueError("image state has no AutoThresholdResult")

    height, width = full_gray.shape

    # T is already known. Establish its seed first. The exact resolved Auto T owns
    # an authoritative seed calculated during auto thresholding; every other T
    # establishes its identity and calculates a new seed at that exact threshold.
    if auto_result.resolved and threshold == int(auto_result.threshold):
        if auto_result.full_seed_point is None:
            raise ThresholdResolutionError(
                "Resolved Auto T has no full-resolution solar seed"
            )
        seed_x, seed_y = map(int, auto_result.full_seed_point)
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError(
                "Stored Auto-T solar seed lies outside the image"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                "Stored Auto-T solar seed is not light at its Auto T"
            )
    else:
        work_gray = resize_gray_max_dim(full_gray)
        work_component = largest_enclosed_bright_component(work_gray > threshold)
        if work_component is None:
            raise ThresholdResolutionError(
                f"No enclosed solar component exists at selected T={threshold}"
            )
        work_mask = np.where(work_component, 255, 0).astype(np.uint8)
        seed_x, seed_y = establish_full_resolution_seed(
            full_gray,
            work_gray.shape,
            work_mask,
            threshold,
        )
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            raise ThresholdResolutionError(
                "Calculated solar seed lies outside the image"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                f"Calculated solar seed is not light at threshold T={threshold}"
            )

    binary = cv2.compare(full_gray, threshold, cv2.CMP_GT)
    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    raw_component = binary == 128
    if _touches_image_border(raw_component):
        raise ThresholdResolutionError(
            f"Seeded solar component touches the image border at T={threshold}"
        )

    refined_component = refine_solar_component_mask(raw_component)
    if not refined_component[seed_y, seed_x]:
        raise ThresholdResolutionError(
            "Refined solar component no longer contains the solar seed"
        )

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

    # No partially constructed state is published if any step above fails.
    image_state["solar_data"] = solar_data
    return refined_component


'''
source = replace_block(
    source,
    "def build_solar_data(\n",
    "class DetectorApp:\n",
    new_build,
)

source = replace_once(
    source,
    """        self.threshold_photo = None
        self.color_photo = None
        self.preview_redraw_job = None
""",
    """        self.preview_redraw_job = None
""",
)

new_load = '''    def load_image_at(self, index: int):
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
            self._redraw_previews()
            self.status.set(f"Could not load image: {path}")
        else:
            # Convert the original image to authoritative grayscale ONCE. The auto
            # threshold module derives its 1200px INTER_AREA working raster from
            # this grayscale image; no HSV/color threshold path exists.
            self.color_image = opaque_bgra(image)
            self.gray_image = to_gray(image)

            restored = path in self.image_state
            if restored:
                state = self.image_state[path]
                result = state["auto_threshold_result"]
                state.setdefault("solar_data", None)
            else:
                result = auto_threshold(self.gray_image)
                state = {
                    "settings": ImageSettings(),
                    "auto_threshold_result": result,
                    "solar_data": None,
                }
                self.image_state[path] = state

            settings = state["settings"]
            for setting_name, variable in self.setting_variables.items():
                if setting_name == "threshold":
                    baseline = int(result.threshold)
                else:
                    baseline = getattr(self.default_settings, setting_name)
                override = getattr(settings, setting_name)
                variable.set(baseline if override is None else override)

            self._update_center_preview_label()
            selected_threshold = int(self.threshold.get())

            # Stage 1: show the pure threshold result immediately.
            self.threshold_preview = self.gray_image > selected_threshold
            self._refresh_display_images()

            # Stage 2: establish the seed/component/SolarData at this exact T and
            # leave the final refined component as the persistent threshold canvas.
            try:
                refined_component = build_solar_data_at_threshold(
                    self.gray_image,
                    selected_threshold,
                    state,
                )
            except (ThresholdResolutionError, ValueError) as exc:
                self.status.set(
                    f"Threshold T={selected_threshold} is visible, but SolarData could not "
                    f"be established ({exc}). Adjust T if needed."
                )
            else:
                self.threshold_preview = refined_component
                if hasattr(self, "threshold_canvas"):
                    self.display_on_canvas(
                        self.threshold_canvas,
                        self.threshold_preview,
                    )

                if restored:
                    self.status.set(
                        f"Image loaded. Restored per-image settings at T={selected_threshold}; "
                        "final refined SolarData mask is current."
                    )
                elif not result.resolved:
                    self.status.set(
                        "Automatic component tracking was unresolved; using "
                        f"rightmost-histogram left edge T={selected_threshold}. SolarData was "
                        "established directly at that T and the refined component is displayed."
                    )
                else:
                    self.status.set(
                        "Image loaded. Automatic grayscale threshold "
                        f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                        f"histogram start={result.histogram_left_edge}); final refined SolarData "
                        "mask displayed."
                    )

        self.update_navigation_state()

'''
source = replace_block(
    source,
    "    def load_image_at(self, index: int):\n",
    "    def previous_image(self):\n",
    new_load,
)

new_threshold_actions = '''    def auto_select_threshold(self):
        """Restore the cached image-only automatic T and process that threshold."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        result = state["auto_threshold_result"]
        if result is None:
            result = auto_threshold(self.gray_image)
            state["auto_threshold_result"] = result

        selected_threshold = int(result.threshold)
        self.threshold.set(selected_threshold)
        self._commit_setting_change("threshold")

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
                )

    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the threshold-finder stage."
        )

    def _refresh_display_images(self):
        """Redraw the currently stored preview rasters without processing them."""
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.display_on_canvas(self.color_canvas, image)

    def _commit_setting_change(self, setting_name):
        """Persist one completed setting change and process T immediately when it changes."""
        if self.current_path is None or self.current_path not in self.image_state:
            return

        state = self.image_state[self.current_path]
        settings = state["settings"]
        value = self.setting_variables[setting_name].get()

        if setting_name == "threshold":
            baseline = int(state["auto_threshold_result"].threshold)
        else:
            baseline = getattr(self.default_settings, setting_name)

        setattr(settings, setting_name, None if value == baseline else value)

        # Other controls do not alter the threshold canvas. T changes are exactly
        # two visual stages: pure threshold first, finalized refined component last.
        if setting_name != "threshold" or self.gray_image is None:
            return

        threshold = int(self.threshold.get())
        self.threshold_preview = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        try:
            refined_component = build_solar_data_at_threshold(
                self.gray_image,
                threshold,
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure threshold displayed at T={threshold}, but SolarData could not be "
                f"established ({exc})."
            )
            return

        self.threshold_preview = refined_component
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

'''
source = replace_block(
    source,
    "    def auto_select_threshold(self):\n",
    "    def _selected_center_target_name(self):\n",
    new_threshold_actions,
)

new_refresh = '''    def refresh_preview(self):
        """Regenerate current-T SolarData before later preview processing stages."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        threshold = int(self.threshold.get())
        existing = state.get("solar_data")
        reused = isinstance(existing, SolarData) and existing.threshold == threshold

        self.threshold_preview = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        try:
            refined_component = build_solar_data_at_threshold(
                self.gray_image,
                threshold,
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure threshold displayed at T={threshold}, but SolarData could not be "
                f"established ({exc}). Adjust the threshold and try again."
            )
            return

        self.threshold_preview = refined_component
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        if reused:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; existing SolarData reused."
            )
        else:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; SolarData rebuilt for this T."
            )

    def apply_full_resolution(self):
        if self.gray_image is None or self.current_path is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        threshold = int(self.threshold.get())
        self.threshold_preview = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        try:
            refined_component = build_solar_data_at_threshold(
                self.gray_image,
                threshold,
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure full-resolution threshold displayed at T={threshold}, but SolarData "
                f"could not be established ({exc})."
            )
            return

        self.threshold_preview = refined_component
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)
        self.status.set(
            "Full-resolution refined solar-component preview applied at "
            f"T={threshold}. Ellipse detector backend not implemented yet."
        )

'''
source = replace_block(
    source,
    "    def refresh_preview(self):\n",
    "    def _schedule_preview_redraw(self, _event=None):\n",
    new_refresh,
)

new_render = '''    def _redraw_previews(self):
        self.preview_redraw_job = None
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.display_on_canvas(self.color_canvas, image)

    def display_on_canvas(self, canvas, canvas_content):
        """Display supplied image/mask content on one canvas and flush pending repaint work."""
        if canvas_content is None:
            display = transparent_bgra()
        else:
            display = np.asarray(canvas_content)
            if display.dtype == bool:
                display = np.where(display, 255, 0).astype(np.uint8)
            elif display.dtype != np.uint8:
                display = np.clip(display, 0, 255).astype(np.uint8)

            if display.ndim == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGRA)
                display[:, :, 3] = 255
            elif display.ndim != 3 or display.shape[2] not in (3, 4):
                raise ValueError("canvas content must be a 2D mask or 3/4-channel image")

        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = display.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(display, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            raise ValueError("could not encode canvas content")

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
        canvas._display_photo = photo

        # Let staged processing become visible without opening arbitrary event
        # callbacks while image-processing state is only partly updated.
        self.root.update_idletasks()

'''
source = replace_block(
    source,
    "    def _redraw_previews(self):\n",
    "    def close(self):\n",
    new_render,
)

for stale_api in (
    "def solar_component_from_seed_at_threshold(",
    "def establish_solar_component_at_threshold(",
    "def build_solar_data(",
    "def _refresh_threshold_image(",
    "def _ensure_solar_data_for_current_threshold(",
    "def _show_image_on_canvas(",
):
    if stale_api in source:
        raise RuntimeError(f"stale API remains after refactor: {stale_api}")

for required in (
    "def build_solar_data_at_threshold(",
    "def display_on_canvas(",
    'image_state["solar_data"] = solar_data',
    "self.root.update_idletasks()",
):
    if required not in source:
        raise RuntimeError(f"required refactor element missing: {required}")

SOURCE.write_text(source)
print("Applied unified SolarData/current-T preview flow")
