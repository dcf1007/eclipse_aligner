"""Apply the validated post-threshold SolarData extension to the monolithic detector.

The input must be the exact pre-SolarData ``circle_arc_detector.py``.  This script
uses asserted textual replacements so the existing automatic-threshold stage is not
silently rewritten while adding downstream solar persistence and GUI lifecycle
handling.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def apply(source: str) -> str:
    text = source

    text = replace_once(
        text,
        "import os\nimport tkinter as tk\n",
        "import os\nimport tkinter as tk\nimport zlib\n",
        "zlib import",
    )

    text = replace_once(
        text,
        "Each image entry is a normal dictionary with exactly two\n"
        "conceptual fields: ``settings`` is an ``ImageSettings`` instance containing only\n"
        "sparse overrides from that image's baseline values, while\n"
        "``auto_threshold_result`` caches the complete automatic threshold result. Ordinary controls use the application defaults as their baseline;\n",
        "Each image entry is a normal dictionary with three conceptual fields:\n"
        "``settings`` stores sparse ``ImageSettings`` overrides,\n"
        "``auto_threshold_result`` caches the complete automatic threshold result, and\n"
        "``solar_data`` caches post-threshold full-resolution solar geometry when it has\n"
        "been established for a specific T. Ordinary controls use the application defaults as their baseline;\n",
        "top-level image-state documentation",
    )

    text = replace_once(
        text,
        "``refresh_preview()`` is the explicit preview-processing entry point and is invoked\n"
        "when the user clicks Refresh Preview or when a readable image is loaded. At the\n"
        "current implementation stage its only image-processing result is still the B/W\n"
        "threshold preview, but later ellipse, arc, and horizon preview processing belongs\n"
        "there rather than in completed-setting commits. Apply Full Resolution remains a\n",
        "``refresh_preview()`` is the explicit processing boundary invoked by the user. A\n"
        "readable image load only draws the image panes after automatic thresholding; if\n"
        "automatic thresholding resolves, SolarData is built as a separate downstream stage\n"
        "after ``AutoThresholdResult`` has already been returned. If automatic thresholding\n"
        "is unresolved, image load stops after the fallback B/W preview and waits for an\n"
        "explicit Refresh Preview. Later ellipse, arc, and horizon preview processing belongs\n"
        "behind that same explicit boundary. Apply Full Resolution remains a\n",
        "refresh-preview documentation",
    )

    post_threshold_block = r'''

# ---------------------------------------------------------------------------
# Post-threshold full-resolution solar data
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SolarData:
    """Solar geometry established at exactly one full-resolution threshold T."""

    threshold: int
    seed_point: tuple[int, int]

    # Full-resolution masks encoded as np.packbits -> zlib level 1.
    component_mask: bytes
    roi_6_5_mask: bytes
    guard_19_5_mask: bytes

    # Ordered external full-resolution contour, shape (N, 2), dtype uint16.
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


def solar_component_from_seed_at_threshold(
    full_gray: np.ndarray,
    threshold: int,
    seed_point: tuple[int, int],
) -> np.ndarray:
    """Flood the enclosed full-resolution light component containing ``seed_point``."""
    threshold = int(threshold)
    seed_x, seed_y = map(int, seed_point)
    height, width = full_gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("Solar seed lies outside the image")

    binary = cv2.compare(full_gray, threshold, cv2.CMP_GT)
    if binary[seed_y, seed_x] == 0:
        raise ThresholdResolutionError(
            f"Solar seed is not light at threshold T={threshold}"
        )

    cv2.floodFill(binary, None, (seed_x, seed_y), 128, flags=8)
    component = binary == 128
    if _touches_image_border(component):
        raise ThresholdResolutionError(
            f"Seeded solar component touches the image border at T={threshold}"
        )
    return component


def establish_solar_component_at_threshold(
    full_gray: np.ndarray,
    threshold: int,
    preferred_seed: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], np.ndarray]:
    """Establish solar identity at one caller-selected T; never search for another T.

    A known solar seed is tried first. If it no longer identifies an enclosed light
    component, use the existing work-resolution enclosed-component machinery at the
    same T, map that identity to full resolution, and flood from the mapped seed.
    """
    threshold = int(threshold)

    if preferred_seed is not None:
        try:
            component = solar_component_from_seed_at_threshold(
                full_gray, threshold, preferred_seed
            )
            seed = tuple(map(int, preferred_seed))
            return seed, component
        except ThresholdResolutionError:
            pass

    work_gray = resize_gray_max_dim(full_gray)
    work_component = largest_enclosed_bright_component(work_gray > threshold)
    if work_component is None:
        raise ThresholdResolutionError(
            f"No enclosed solar component exists at selected T={threshold}"
        )

    work_mask = np.where(work_component, 255, 0).astype(np.uint8)
    full_seed = establish_full_resolution_seed(
        full_gray,
        work_gray.shape,
        work_mask,
        threshold,
    )
    component = solar_component_from_seed_at_threshold(
        full_gray,
        threshold,
        full_seed,
    )
    return full_seed, component


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


def build_solar_data(
    full_gray: np.ndarray,
    threshold: int,
    seed_point: tuple[int, int],
    component: np.ndarray,
) -> SolarData:
    """Build persistent full-resolution solar geometry after thresholding is complete."""
    threshold = int(threshold)
    component = np.asarray(component, dtype=bool)
    if component.shape != full_gray.shape:
        raise ValueError("solar component and grayscale image must have identical shapes")
    if not np.any(component):
        raise ThresholdResolutionError("Cannot build SolarData from an empty component")

    seed_x, seed_y = map(int, seed_point)
    height, width = full_gray.shape
    if not (0 <= seed_x < width and 0 <= seed_y < height):
        raise ThresholdResolutionError("SolarData seed lies outside the image")
    if not component[seed_y, seed_x]:
        raise ThresholdResolutionError("SolarData seed is outside the solar component")
    if int(full_gray[seed_y, seed_x]) <= threshold:
        raise ThresholdResolutionError("SolarData seed is not light at its threshold")

    image_scale = math.sqrt(float(width) * float(height))
    roi_6_5 = _full_mask_from_observation_region(
        full_gray,
        component,
        ROI_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
    )
    guard_19_5 = _full_mask_from_observation_region(
        full_gray,
        component,
        GUARD_DILATION_FRACTION * image_scale,
        (seed_x, seed_y),
    )
    contour = _ordered_external_component_contour(component)

    return SolarData(
        threshold=threshold,
        seed_point=(seed_x, seed_y),
        component_mask=compress_full_mask(component),
        roi_6_5_mask=compress_full_mask(roi_6_5),
        guard_19_5_mask=compress_full_mask(guard_19_5),
        component_contour=contour,
    )
'''

    text = replace_once(
        text,
        "\n\nclass DetectorApp:\n",
        post_threshold_block + "\n\nclass DetectorApp:\n",
        "post-threshold SolarData block",
    )

    text = replace_once(
        text,
        "        # Per-image state keeps a sparse ImageSettings instance plus the cached\n"
        "        # automatic threshold result. No ImageState wrapper class is needed: the\n"
        "        # outer dictionary directly expresses the image-to-state hierarchy.\n",
        "        # Per-image state keeps sparse settings, the cached automatic threshold\n"
        "        # result, and post-threshold SolarData when solar geometry has been built.\n"
        "        # No ImageState wrapper class is needed: the outer dictionary directly\n"
        "        # expresses the image-to-state hierarchy.\n",
        "image-state comment",
    )

    old_load = '''            restored = path in self.image_state
            if restored:
                state = self.image_state[path]
                result = state["auto_threshold_result"]
            else:
                result = auto_threshold(self.gray_image)
                state = {
                    "settings": ImageSettings(),
                    "auto_threshold_result": result,
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
            self.refresh_preview()
            selected_threshold = int(self.threshold.get())

            if restored:
                self.status.set(
                    f"Image loaded. Restored per-image settings at T={selected_threshold}."
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
'''

    new_load = '''            restored = path in self.image_state
            solar_build_error = None
            if restored:
                state = self.image_state[path]
                result = state["auto_threshold_result"]
                state.setdefault("solar_data", None)
            else:
                # Stage 1 completes fully before post-threshold solar persistence starts.
                result = auto_threshold(self.gray_image)
                solar_data = None
                if result.resolved and result.full_seed_point is not None:
                    try:
                        component = solar_component_from_seed_at_threshold(
                            self.gray_image,
                            result.threshold,
                            result.full_seed_point,
                        )
                        solar_data = build_solar_data(
                            self.gray_image,
                            result.threshold,
                            result.full_seed_point,
                            component,
                        )
                    except (ThresholdResolutionError, ValueError) as exc:
                        solar_build_error = str(exc)

                state = {
                    "settings": ImageSettings(),
                    "auto_threshold_result": result,
                    "solar_data": solar_data,
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
            # Loading only redraws the image panes. Refresh Preview remains the
            # explicit processing boundary for user-selected settings.
            self._refresh_display_images()
            selected_threshold = int(self.threshold.get())
            solar_data = state.get("solar_data")

            if restored:
                if isinstance(solar_data, SolarData) and solar_data.threshold == selected_threshold:
                    self.status.set(
                        f"Image loaded. Restored per-image settings at T={selected_threshold}; "
                        "SolarData is current."
                    )
                else:
                    self.status.set(
                        f"Image loaded. Restored per-image settings at T={selected_threshold}. "
                        "Press Refresh Preview to process the selected settings."
                    )
            elif not result.resolved:
                self.status.set(
                    "Automatic solar-component detection was unresolved; using "
                    f"rightmost-histogram left edge T={selected_threshold}. Adjust the "
                    "threshold if needed, then click Refresh Preview to continue."
                )
            elif solar_build_error is not None:
                self.status.set(
                    f"Automatic threshold T={selected_threshold} resolved, but SolarData "
                    f"could not be built ({solar_build_error}). Click Refresh Preview to retry."
                )
            else:
                self.status.set(
                    "Image loaded. Automatic grayscale threshold "
                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                    f"histogram start={result.histogram_left_edge}); SolarData prepared."
                )
'''

    text = replace_once(text, old_load, new_load, "image-load lifecycle")

    old_refresh_tail = '''        if hasattr(self, "threshold_canvas"):
            self.threshold_photo = self._show_image_on_canvas(
                self.threshold_canvas, self.threshold_preview
            )

    def _commit_setting_change(self, setting_name):
'''

    new_refresh_tail = '''        if hasattr(self, "threshold_canvas"):
            self.threshold_photo = self._show_image_on_canvas(
                self.threshold_canvas, self.threshold_preview
            )

    def _refresh_display_images(self):
        """Redraw the threshold and color panes without running processing stages."""
        self._refresh_threshold_image()
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)

    def _ensure_solar_data_for_current_threshold(self) -> tuple[SolarData, bool]:
        """Return current-T SolarData, rebuilding it only at an explicit processing boundary."""
        if self.gray_image is None or self.current_path is None:
            raise ThresholdResolutionError("No readable image is loaded")

        state = self.image_state[self.current_path]
        threshold = int(self.threshold.get())
        existing = state.get("solar_data")
        if isinstance(existing, SolarData) and existing.threshold == threshold:
            return existing, False

        result = state["auto_threshold_result"]
        preferred_seed = None
        if result.resolved and result.full_seed_point is not None and threshold == int(result.threshold):
            preferred_seed = result.full_seed_point
        elif isinstance(existing, SolarData):
            preferred_seed = existing.seed_point
        elif result.resolved and result.full_seed_point is not None:
            preferred_seed = result.full_seed_point

        seed_point, component = establish_solar_component_at_threshold(
            self.gray_image,
            threshold,
            preferred_seed=preferred_seed,
        )
        solar_data = build_solar_data(
            self.gray_image,
            threshold,
            seed_point,
            component,
        )
        state["solar_data"] = solar_data
        return solar_data, True

    def _commit_setting_change(self, setting_name):
'''

    text = replace_once(
        text,
        old_refresh_tail,
        new_refresh_tail,
        "display and SolarData GUI helpers",
    )

    text = replace_once(
        text,
        '        """Persist one changed per-image setting, then refresh only the B/W image."""\n',
        '        """Persist one changed setting; completed control changes stay lightweight."""\n',
        "setting-commit docstring",
    )

    old_refresh_preview = '''    def refresh_preview(self):
        """Run explicit preview processing for the currently loaded image."""
        if self.gray_image is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return

        self._refresh_threshold_image()
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)
        self.status.set(
            f"Threshold preview regenerated at T={int(self.threshold.get())}."
        )
'''

    new_refresh_preview = '''    def refresh_preview(self):
        """Explicitly process the current settings, beginning with current-T SolarData."""
        if self.gray_image is None or self.current_path is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return

        threshold = int(self.threshold.get())
        try:
            _solar_data, rebuilt = self._ensure_solar_data_for_current_threshold()
        except (ThresholdResolutionError, ValueError) as exc:
            # Keep any older SolarData object as a self-describing stale cache. It
            # cannot be consumed while its stored T differs from the current T.
            self._refresh_display_images()
            self.status.set(
                f"No enclosed solar component could be established at T={threshold}: {exc}. "
                "Adjust the threshold and click Refresh Preview again."
            )
            return

        self._refresh_display_images()
        if rebuilt:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; SolarData rebuilt for this T."
            )
        else:
            self.status.set(
                f"Threshold preview regenerated at T={threshold}; existing SolarData reused."
            )
'''

    text = replace_once(
        text,
        old_refresh_preview,
        new_refresh_preview,
        "explicit Refresh Preview lifecycle",
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.write_text(apply(source_path.read_text()))


if __name__ == "__main__":
    main()
