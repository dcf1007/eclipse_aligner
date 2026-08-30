"""Apply the validated threshold-preview refresh and manual-T reseed refactor.

This transformation is intentionally preserved in ``snippets`` so the production
change can be reproduced exactly.  It targets the threshold-finder source at the
unified SolarData-preview baseline.
"""

from pathlib import Path


SOURCE_PATH = Path("circle_arc_detector.py")


def replace_function(source: str, name: str, replacement: str, *, class_method: bool) -> str:
    """Replace one complete Python function by its next same-level definition."""
    prefix = "    def " if class_method else "def "
    marker = f"{prefix}{name}("
    start = source.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not find {marker!r}")

    search_from = start + len(marker)
    if class_method:
        candidates = [
            pos for pos in (
                source.find("\n    def ", search_from),
                source.find("\n\nclass ", search_from),
            ) if pos >= 0
        ]
    else:
        candidates = [
            pos for pos in (
                source.find("\ndef ", search_from),
                source.find("\n\nclass ", search_from),
            ) if pos >= 0
        ]
    if not candidates:
        raise RuntimeError(f"Could not locate end of {name}")
    end = min(candidates)
    return source[:start] + replacement.rstrip() + "\n" + source[end:]


source = SOURCE_PATH.read_text(encoding="utf-8")

constant_anchor = "PREVIEW_REDRAW_DELAY_MS = 60\n"
constant_replacement = (
    "PREVIEW_REDRAW_DELAY_MS = 60\n"
    "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS = 40\n"
)
if "THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS" not in source:
    if constant_anchor not in source:
        raise RuntimeError("Preview delay constant anchor not found")
    source = source.replace(constant_anchor, constant_replacement, 1)

init_anchor = "        self.preview_redraw_job = None\n"
init_replacement = (
    "        self.preview_redraw_job = None\n"
    "        self.threshold_refinement_job = None\n"
)
if "self.threshold_refinement_job = None" not in source:
    if init_anchor not in source:
        raise RuntimeError("DetectorApp preview job anchor not found")
    source = source.replace(init_anchor, init_replacement, 1)

build_solar_data = '''def build_solar_data_at_threshold(
    full_gray: np.ndarray,
    threshold: int,
    image_state: dict[str, object],
) -> np.ndarray:
    """Establish, refine, and persist SolarData at exactly one already-selected T.

    At the exact resolved automatic threshold, the Auto-T full-resolution seed is
    authoritative and must survive refinement. At every other T, a freshly
    calculated seed is used only to identify/flood the current-T raw component; the
    finalized refined component then receives its own robust interior seed. This
    prevents a valid manual/current-T component from failing merely because 7x7
    cleanup removed the particular pre-cleanup flood seed.
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
    exact_auto_threshold = bool(
        auto_result.resolved and threshold == int(auto_result.threshold)
    )

    # T is already known. Establish an identity/flood seed first. The exact
    # resolved Auto T owns its stored authoritative seed. Every other T establishes
    # solar identity and calculates a fresh seed at that exact threshold.
    if exact_auto_threshold:
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
    if not np.any(refined_component):
        raise ThresholdResolutionError(
            f"Solar component was removed by refinement at T={threshold}"
        )

    if exact_auto_threshold:
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Refined solar component no longer contains the Auto-T solar seed"
            )
    else:
        # The current-T flood seed established component identity before cleanup.
        # Once cleanup has finalized that same component, seed the authoritative
        # SolarData from the finalized geometry rather than requiring one raw-mask
        # pixel to survive morphology.
        refined_u8 = np.where(refined_component, 255, 0).astype(np.uint8)
        seed_x, seed_y = brightest_supported_component_point(full_gray, refined_u8)
        if not refined_component[seed_y, seed_x]:
            raise ThresholdResolutionError(
                "Calculated refined-component seed lies outside the solar component"
            )
        if int(full_gray[seed_y, seed_x]) <= threshold:
            raise ThresholdResolutionError(
                f"Calculated refined-component seed is not light at T={threshold}"
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
source = replace_function(
    source,
    "build_solar_data_at_threshold",
    build_solar_data,
    class_method=False,
)

commit_setting = '''    def _commit_setting_change(self, setting_name):
        """Persist one completed setting change, then refresh the current preview."""
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

        # A completed control interaction is the processing boundary. Slider value
        # labels may update continuously while dragging/auto-repeating, but the
        # expensive preview refresh happens once when that interaction is committed.
        if self.gray_image is not None:
            self.refresh_preview(changed_setting=setting_name)
'''
source = replace_function(
    source,
    "_commit_setting_change",
    commit_setting,
    class_method=True,
)

refresh_preview = '''    def refresh_preview(self, changed_setting: str | None = None, full_resolution: bool = False):
        """Show pure current-T threshold now, then refine it on the next GUI turn.

        Separating the two visual stages through ``after`` guarantees that Tk can
        paint the raw ``gray > T`` result before SolarData construction replaces it
        with the finalized refined component. Any pending stage-2 job is cancelled
        when a newer control commit arrives, so stale T values cannot win a race.
        """
        if self.gray_image is None or self.current_path is None:
            action = "Apply Full Resolution" if full_resolution else "Refresh Preview"
            self.status.set(f"{action}: no readable image is loaded.")
            return

        state = self.image_state[self.current_path]
        threshold = int(self.threshold.get())
        existing = state.get("solar_data")
        reused = isinstance(existing, SolarData) and existing.threshold == threshold

        # Never expose SolarData from a different T as current while the new T is
        # waiting for its stage-2 rebuild. This also makes downstream state ownership
        # explicit if current-T establishment later fails.
        if isinstance(existing, SolarData) and existing.threshold != threshold:
            state["solar_data"] = None

        self.threshold_preview = self.gray_image > threshold
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

        if self.threshold_refinement_job is not None:
            self.root.after_cancel(self.threshold_refinement_job)
            self.threshold_refinement_job = None

        if full_resolution:
            self.status.set(
                f"Pure full-resolution threshold displayed at T={threshold}; refining current-T component."
            )
        elif changed_setting is not None:
            self.status.set(
                f"{changed_setting} committed; pure threshold displayed at T={threshold}; "
                "refining current preview."
            )
        else:
            self.status.set(
                f"Pure threshold displayed at T={threshold}; refining current preview."
            )

        self.threshold_refinement_job = self.root.after(
            THRESHOLD_REFINEMENT_DISPLAY_DELAY_MS,
            self._finish_threshold_preview_refresh,
            self.current_path,
            threshold,
            reused,
            changed_setting,
            full_resolution,
        )

    def _finish_threshold_preview_refresh(
        self,
        path: str,
        threshold: int,
        reused: bool,
        changed_setting: str | None,
        full_resolution: bool,
    ):
        """Build current-T SolarData and publish the finalized refined preview."""
        self.threshold_refinement_job = None

        # A navigation or newer threshold change supersedes this queued stage.
        if self.current_path != path or self.gray_image is None:
            return
        if int(self.threshold.get()) != int(threshold):
            return

        state = self.image_state[path]
        try:
            refined_component = build_solar_data_at_threshold(
                self.gray_image,
                int(threshold),
                state,
            )
        except (ThresholdResolutionError, ValueError) as exc:
            self.status.set(
                f"Pure threshold remains displayed at T={threshold}; SolarData could not be "
                f"established ({exc})."
            )
            return

        self.threshold_preview = refined_component
        if hasattr(self, "threshold_canvas"):
            self.display_on_canvas(self.threshold_canvas, self.threshold_preview)

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
            )
'''
source = replace_function(
    source,
    "refresh_preview",
    refresh_preview,
    class_method=True,
)

apply_full_resolution = '''    def apply_full_resolution(self):
        if self.gray_image is None or self.current_path is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return
        self.refresh_preview(full_resolution=True)
'''
source = replace_function(
    source,
    "apply_full_resolution",
    apply_full_resolution,
    class_method=True,
)

SOURCE_PATH.write_text(source, encoding="utf-8")
