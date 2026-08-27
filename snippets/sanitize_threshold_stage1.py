from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"

text = SOURCE.read_text(encoding="utf-8")
original = text

# ---------------------------------------------------------------------------
# Module documentation and imports
# ---------------------------------------------------------------------------
image_types_at = text.index("IMAGE_FILE_TYPES = (")
new_prefix = '''"""Eclipse alignment GUI with grayscale automatic threshold selection.

This module combines the application's user-interface foundation with the tested
image-only automatic threshold finder. The GUI owns image navigation, per-image
processing settings, control interaction, preview lifecycle, and cached automatic
threshold results. The threshold finder itself remains independent of GUI state:
it accepts image data and returns an ``AutoThresholdResult`` describing the
selected threshold and the topology used to obtain it.

All processing controls are per-image. ``DetectorApp.image_state`` is keyed by the
absolute image path. Each image entry is a normal dictionary with exactly two
conceptual fields: ``settings`` contains only sparse overrides from that image's
baseline values, while ``auto_threshold_result`` caches the complete automatic
threshold result. Ordinary controls use the application defaults as their baseline;
the threshold uses the cached automatic threshold as its image-specific baseline.
Returning a control to its baseline removes that key from ``settings``. Reusing the
cached automatic result means the threshold algorithm is not rerun when Auto select
is clicked again for an unchanged loaded image.

Slider labels update continuously, but a setting is committed only after mouse
release or the final keyboard key release. Keyboard auto-repeat can emit temporary
release/press pairs on some Tk platforms, so releases are coalesced through a short
settling window. Checkboxes and radio buttons commit immediately. A completed
setting change calls ``_commit_setting_change(setting_name)``; that function updates
only the changed sparse override and then invokes ``_refresh_threshold_image()``.
That lightweight path performs only the current-T B/W conversion and threshold-pane
update. It does not run the broader Refresh Preview processing path.

``refresh_preview()`` is the explicit preview-processing entry point and is invoked
when the user clicks Refresh Preview or when a readable image is loaded. At the
current implementation stage its only image-processing result is still the B/W
threshold preview, but later ellipse, arc, and horizon preview processing belongs
there rather than in completed-setting commits. Apply Full Resolution remains a
separate explicit action. Radius auto-selection, ellipse fitting, horizon handling,
centering, and export are not implemented yet.

The threshold algorithm uses authoritative 8-bit grayscale with fixed semantics
``dark = gray <= T`` and ``light = gray > T``. It derives a <=1200-pixel working
raster with INTER_AREA, uses the rightmost locally smoothed histogram mode only to
establish a solar seed, tracks 8-connected component topology, maps the seed back
to full resolution, constructs rounded 6.5% and 19.5% observation regions, and
selects the lowest full-resolution threshold whose tracked component is genuinely
separated. If topology cannot be resolved, the deterministic fallback is the left
edge of the rightmost histogram peak. No HSV/color thresholding, Otsu thresholding,
ellipse-fit score, bright-pixel dominance, competitor gain, or horizon special case
is part of automatic threshold selection.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import math
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


'''
text = new_prefix + text[image_types_at:]

# Remove the old secondary threshold-description string and duplicate imports.
secondary_start = text.index(
    "# ---------------------------------------------------------------------------\n"
    "# Grayscale-only automatic threshold finder (tested implementation)\n"
    "# ---------------------------------------------------------------------------\n"
)
secondary_end_marker = "import numpy as np\n\nWORK_MAX_DIM = 1200"
secondary_end = text.index(secondary_end_marker, secondary_start) + len("import numpy as np\n\n")
text = (
    text[:secondary_start]
    + "# ---------------------------------------------------------------------------\n"
      "# Grayscale automatic threshold finder\n"
      "# ---------------------------------------------------------------------------\n"
    + text[secondary_end:]
)

old_types = '''IMAGE_FILE_TYPES = (\n    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),\n    ("All files", "*.*"),\n)\n'''
new_types = old_types + '''\nSLIDER_KEY_RELEASE_SETTLE_MS = 45\nPREVIEW_REDRAW_DELAY_MS = 60\n'''
assert old_types in text
text = text.replace(old_types, new_types, 1)

# ---------------------------------------------------------------------------
# Remove only the two already-approved dead algorithm parameters.
# ---------------------------------------------------------------------------
text = text.replace(
    '''def full_roi_seed_component(\n    full_gray: np.ndarray,\n    coarse_threshold: int,\n    seed_threshold: int,\n    seed_point: tuple[int, int],\n):''',
    '''def full_roi_seed_component(\n    full_gray: np.ndarray,\n    coarse_threshold: int,\n    seed_point: tuple[int, int],\n):''',
    1,
)
text = text.replace(
    '''def find_lowest_full_threshold(\n    full_gray: np.ndarray,\n    seed_point: tuple[int, int],\n    roi_seed_component: np.ndarray,\n    histogram_upper_hint: int,\n):''',
    '''def find_lowest_full_threshold(\n    full_gray: np.ndarray,\n    seed_point: tuple[int, int],\n    roi_seed_component: np.ndarray,\n):''',
    1,
)
text = text.replace(
    '''    used_guard = False\n    upper = max(0, min(255, int(histogram_upper_hint)))\n    for threshold in range(0, 256):\n        # The histogram upper hint is diagnostic rather than a hard limit. We keep\n        # scanning defensively above it if original-resolution topology requires it.\n''',
    '''    used_guard = False\n    for threshold in range(0, 256):\n''',
    1,
)
text = text.replace(
    '''        roi_seed_t, roi_seed_component = full_roi_seed_component(\n            full_gray, coarse.threshold, coarse.seed_threshold, full_seed\n        )''',
    '''        roi_seed_t, roi_seed_component = full_roi_seed_component(\n            full_gray, coarse.threshold, full_seed\n        )''',
    1,
)
text = text.replace(
    '''        final_t, used_guard = find_lowest_full_threshold(\n            full_gray,\n            full_seed,\n            roi_seed_component,\n            histogram_upper_hint=max(coarse.seed_threshold, roi_seed_t),\n        )''',
    '''        final_t, used_guard = find_lowest_full_threshold(\n            full_gray,\n            full_seed,\n            roi_seed_component,\n        )''',
    1,
)

# ---------------------------------------------------------------------------
# Adopt the validated GUI naming/interaction structure.
# ---------------------------------------------------------------------------
renames = {
    "build_navigation": "_build_navigation_bar",
    "build_controls": "_build_settings_panel",
    "build_previews": "_build_preview_panes",
    "add_scale": "_add_slider",
    "focus_scale": "_focus_slider",
    "begin_scale_mouse_change": "_begin_slider_mouse_change",
    "finish_scale_mouse_change": "_finish_slider_mouse_change",
    "begin_scale_key_change": "_begin_slider_keyboard_change",
    "defer_scale_key_refresh": "_schedule_slider_keyboard_commit",
    "finish_scale_key_change": "_finish_slider_keyboard_change",
    "release_scale_focus_if_outside": "_release_slider_focus_if_clicked_elsewhere",
    "pending": "_commit_setting_change",
    "center_target_changed": "_handle_center_target_change",
    "update_center_preview_label": "_update_center_preview_label",
    "render_threshold_preview": "_refresh_threshold_image",
    "schedule_redraw": "_schedule_preview_redraw",
    "redraw": "_redraw_previews",
    "show_image": "_show_image_on_canvas",
    "resize_job": "preview_redraw_job",
    "scale_key_release_job": "slider_keyboard_commit_job",
    "scale_key_widget": "slider_keyboard_widget",
    "scale_key_start_value": "slider_keyboard_start_value",
}
for old, new in renames.items():
    text = text.replace(old, new)
text = text.replace("_schedule_preview__redraw_previews", "_schedule_preview_redraw")

old_class_doc = '''    """GUI shell for the eclipse detector rebuild.\n\n    Only interface behavior is implemented here: image loading/navigation,\n    controls, preview panes, and the centering-target selector. Detector buttons\n    deliberately report that backend functionality has not yet been implemented.\n    """'''
new_class_doc = '''    """Own GUI state, per-image settings, cached auto thresholds, and previews.\n\n    Public methods represent application actions. Underscore-prefixed methods are\n    Tk callback or rendering internals. Automatic threshold selection remains a\n    stateless image algorithm outside this class; this class only caches its result\n    per image and manages the effective processing settings shown by the controls.\n    """'''
assert old_class_doc in text
text = text.replace(old_class_doc, new_class_doc, 1)

old_state = '''        # Threshold is stored per image so a manual override survives navigation.\n        # Automatic selection runs on first load and only reruns when Auto select is\n        # explicitly clicked for that image.\n        self.image_thresholds: dict[str, int] = {}\n        self.image_auto_results = {}\n'''
new_state = '''        # Per-image state keeps sparse setting overrides plus the cached automatic\n        # threshold result. No additional ImageState class is needed: the nested\n        # dictionaries directly express the state hierarchy.\n        self.image_state: dict[str, dict[str, object]] = {}\n'''
assert old_state in text
text = text.replace(old_state, new_state, 1)

old_vars = '''        self.center_target = tk.StringVar(value="light")\n        self.center_preview_label = tk.StringVar()\n'''
new_vars = '''        self.center_target = tk.StringVar(value="light")\n        self.center_preview_label = tk.StringVar()\n\n        # Every processing control is per-image. Ordinary controls use these values\n        # as their baseline; threshold instead uses the cached automatic T for the\n        # current image. Only deviations from those baselines are stored.\n        self.default_settings = {\n            "min_radius": int(self.min_radius.get()),\n            "max_radius": int(self.max_radius.get()),\n            "max_error": float(self.max_error.get()),\n            "min_coverage": int(self.min_coverage.get()),\n            "morphology": bool(self.morphology.get()),\n            "outer_limb_assistance": bool(self.outer_limb_assistance.get()),\n            "use_horizon": bool(self.use_horizon.get()),\n            "center_target": self.center_target.get(),\n        }\n        self.setting_variables = {\n            "threshold": self.threshold,\n            "min_radius": self.min_radius,\n            "max_radius": self.max_radius,\n            "max_error": self.max_error,\n            "min_coverage": self.min_coverage,\n            "morphology": self.morphology,\n            "outer_limb_assistance": self.outer_limb_assistance,\n            "use_horizon": self.use_horizon,\n            "center_target": self.center_target,\n        }\n'''
assert old_vars in text
text = text.replace(old_vars, new_vars, 1)

text = text.replace(
    "self.slider_keyboard_commit_job = self.root.after(45, self._finish_slider_keyboard_change)",
    "self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )",
    1,
)
text = text.replace(
    "self.preview_redraw_job = self.root.after(60, self._redraw_previews)",
    "self.preview_redraw_job = self.root.after(\n            PREVIEW_REDRAW_DELAY_MS, self._redraw_previews\n        )",
    1,
)

old_keyboard = '''    def _begin_slider_keyboard_change(self, event):\n        """Start/coalesce a keyboard slider interaction without refreshing."""\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n            self.slider_keyboard_commit_job = None\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n\n    def _schedule_slider_keyboard_commit(self, event):\n        """Refresh after the final KeyRelease, not during key auto-repeat."""\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n        self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )\n'''
new_keyboard = '''    def _cancel_pending_slider_keyboard_commit(self):\n        """Cancel a release callback superseded by continuing keyboard input."""\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n            self.slider_keyboard_commit_job = None\n\n    def _begin_slider_keyboard_change(self, event):\n        """Begin or continue one keyboard slider interaction."""\n        self._cancel_pending_slider_keyboard_commit()\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n\n    def _schedule_slider_keyboard_commit(self, event):\n        """Schedule completion after a KeyRelease survives the repeat window."""\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n        self._cancel_pending_slider_keyboard_commit()\n        self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )\n'''
assert old_keyboard in text
text = text.replace(old_keyboard, new_keyboard, 1)

# ---------------------------------------------------------------------------
# Restore/create all per-image settings and use image load as a full preview entry.
# ---------------------------------------------------------------------------
old_load_state = '''            restored = path in self.image_thresholds\n            if restored:\n                selected_threshold = int(self.image_thresholds[path])\n            else:\n                result = auto_threshold_from_gray(self.gray_image)\n                selected_threshold = int(result.threshold)\n                self.image_thresholds[path] = selected_threshold\n                self.image_auto_results[path] = result\n\n            # Setting the Tk variable updates the displayed slider value only.\n            # Image loading explicitly regenerates the black/white threshold raster\n            # immediately after restoring or automatically selecting T.\n            self.threshold.set(selected_threshold)\n            self._refresh_threshold_image()\n            self._redraw_previews()\n\n            if restored:\n                self.status.set(\n                    f"Image loaded. Restored stored threshold T={selected_threshold}."\n                )\n            elif result.resolved:\n                self.status.set(\n                    "Image loaded. Automatic grayscale threshold "\n                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "\n                    f"histogram start={result.histogram_left_edge})."\n                )\n            else:\n                self.status.set(\n                    "Image loaded. Automatic component tracking was unresolved; "\n                    f"using rightmost-histogram left edge T={selected_threshold}."\n                )\n'''
new_load_state = '''            restored = path in self.image_state\n            if restored:\n                state = self.image_state[path]\n                result = state["auto_threshold_result"]\n            else:\n                result = auto_threshold_from_gray(self.gray_image)\n                state = {\n                    "settings": {},\n                    "auto_threshold_result": result,\n                }\n                self.image_state[path] = state\n\n            settings = state["settings"]\n            for setting_name, variable in self.setting_variables.items():\n                if setting_name == "threshold":\n                    baseline = int(result.threshold)\n                else:\n                    baseline = self.default_settings[setting_name]\n                variable.set(settings.get(setting_name, baseline))\n\n            self._update_center_preview_label()\n            self.refresh_preview()\n            selected_threshold = int(self.threshold.get())\n\n            if restored:\n                self.status.set(\n                    f"Image loaded. Restored per-image settings at T={selected_threshold}."\n                )\n            elif result.resolved:\n                self.status.set(\n                    "Image loaded. Automatic grayscale threshold "\n                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "\n                    f"histogram start={result.histogram_left_edge})."\n                )\n            else:\n                self.status.set(\n                    "Image loaded. Automatic component tracking was unresolved; "\n                    f"using rightmost-histogram left edge T={selected_threshold}."\n                )\n'''
assert old_load_state in text
text = text.replace(old_load_state, new_load_state, 1)

# ---------------------------------------------------------------------------
# Settings panel: identify every slider/control by its per-image setting name.
# ---------------------------------------------------------------------------
old_rows = '''        rows = [\n            ("Brightness threshold (dark <= T, light > T)", self.threshold,\n             0, 255, 1, lambda v: str(int(float(v)))),\n            ("Minimum FINAL fitted semi-axis radius (px)", self.min_radius,\n             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),\n            ("Maximum FINAL fitted semi-axis radius (px)", self.max_radius,\n             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),\n            ("Maximum average normalized ellipse error (%)", self.max_error,\n             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),\n            ("Minimum TOTAL supported ellipse arc (%)", self.min_coverage,\n             0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),\n        ]\n        for row, spec in enumerate(rows):\n            self._add_slider(frame, row, *spec)\n'''
new_rows = '''        slider_specs = [\n            ("threshold", "Brightness threshold (dark <= T, light > T)", self.threshold,\n             0, 255, 1, lambda v: str(int(float(v)))),\n            ("min_radius", "Minimum FINAL fitted semi-axis radius (px)", self.min_radius,\n             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),\n            ("max_radius", "Maximum FINAL fitted semi-axis radius (px)", self.max_radius,\n             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),\n            ("max_error", "Maximum average normalized ellipse error (%)", self.max_error,\n             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),\n            ("min_coverage", "Minimum TOTAL supported ellipse arc (%)", self.min_coverage,\n             0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),\n        ]\n        for row, spec in enumerate(slider_specs):\n            self._add_slider(frame, row, *spec)\n'''
assert old_rows in text
text = text.replace(old_rows, new_rows, 1)

text = text.replace("command=self._commit_setting_change,", 'command=lambda: self._commit_setting_change("morphology"),', 1)
text = text.replace("command=self._commit_setting_change,", 'command=lambda: self._commit_setting_change("outer_limb_assistance"),', 1)
text = text.replace("command=self._commit_setting_change,", 'command=lambda: self._commit_setting_change("use_horizon"),', 1)

old_signature = "    def _add_slider(self, parent, row, text, variable, low, high, resolution, formatter):"
new_signature = "    def _add_slider(self, parent, row, setting_name, text, variable, low, high, resolution, formatter):"
assert old_signature in text
text = text.replace(old_signature, new_signature, 1)

slider_start = text.index("    def _add_slider(")
slider_end = text.index("    @staticmethod\n    def _focus_slider", slider_start)
slider_block = text[slider_start:slider_end]
slider_block = slider_block.replace("scale = tk.Scale(", "slider = tk.Scale(")
slider_block = slider_block.replace("scale.grid(", "slider.grid(")
slider_block = slider_block.replace("scale.bind(", "slider.bind(")
slider_block = slider_block.replace(
    '        slider.grid(row=row, column=1, sticky="ew", pady=2)\n',
    '        slider.grid(row=row, column=1, sticky="ew", pady=2)\n        slider._setting_name = setting_name\n',
    1,
)
text = text[:slider_start] + slider_block + text[slider_end:]

text = text.replace(
    "            self._commit_setting_change()\n\n    def _cancel_pending_slider_keyboard_commit",
    "            self._commit_setting_change(event.widget._setting_name)\n\n    def _cancel_pending_slider_keyboard_commit",
    1,
)
text = text.replace(
    "            self._commit_setting_change()\n\n    def _release_slider_focus_if_clicked_elsewhere",
    "            self._commit_setting_change(widget._setting_name)\n\n    def _release_slider_focus_if_clicked_elsewhere",
    1,
)

# ---------------------------------------------------------------------------
# Application actions/state persistence and lightweight threshold refresh.
# ---------------------------------------------------------------------------
old_auto = '''    def auto_select_threshold(self):\n        """Rerun image-only automatic T selection and regenerate the preview."""\n        if self.gray_image is None:\n            self.status.set("Auto select threshold: no readable image is loaded.")\n            return\n\n        result = auto_threshold_from_gray(self.gray_image)\n        selected_threshold = int(result.threshold)\n        if self.current_path is not None:\n            self.image_thresholds[self.current_path] = selected_threshold\n            self.image_auto_results[self.current_path] = result\n\n        # The threshold variable trace updates only the displayed slider value.\n        # Auto select is a completed setting change, so regenerate through the same\n        # refresh path used by the other controls after storing the new T.\n        self.threshold.set(selected_threshold)\n        self.refresh_preview()\n        if result.resolved:\n'''
new_auto = '''    def auto_select_threshold(self):\n        """Restore the cached image-only automatic T and refresh the B/W image."""\n        if self.gray_image is None or self.current_path is None:\n            self.status.set("Auto select threshold: no readable image is loaded.")\n            return\n\n        state = self.image_state[self.current_path]\n        result = state["auto_threshold_result"]\n        if result is None:\n            result = auto_threshold_from_gray(self.gray_image)\n            state["auto_threshold_result"] = result\n\n        selected_threshold = int(result.threshold)\n        self.threshold.set(selected_threshold)\n        self._commit_setting_change("threshold")\n        if result.resolved:\n'''
assert old_auto in text
text = text.replace(old_auto, new_auto, 1)

# Remove obsolete transparent-preview invalidation method.
clear_start = text.index("    def clear_threshold_preview(self):")
clear_end = text.index("    def _commit_setting_change", clear_start)
text = text[:clear_start] + text[clear_end:]

old_commit = '''    def _commit_setting_change(self, *_args):\n        """Store the current T and refresh after a completed/discrete setting change."""\n        if self.current_path is not None and self.gray_image is not None:\n            self.image_thresholds[self.current_path] = int(self.threshold.get())\n        self.refresh_preview()\n\n    def _handle_center_target_change(self):\n        self._update_center_preview_label()\n        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n        self.refresh_preview()\n        self.status.set(\n            f"Centering target set to {target}. Actual centering will be implemented with ellipse detection."\n        )\n\n    def _update_center_preview_label(self):\n        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n        self.center_preview_label.set(f"Full-color image — center on {target}")\n'''
new_commit = '''    def _commit_setting_change(self, setting_name):\n        """Persist one changed per-image setting, then refresh only the B/W image."""\n        if self.current_path is not None and self.current_path in self.image_state:\n            state = self.image_state[self.current_path]\n            settings = state["settings"]\n            value = self.setting_variables[setting_name].get()\n\n            if setting_name == "threshold":\n                baseline = int(state["auto_threshold_result"].threshold)\n            else:\n                baseline = self.default_settings[setting_name]\n\n            if value == baseline:\n                settings.pop(setting_name, None)\n            else:\n                settings[setting_name] = value\n\n        self._refresh_threshold_image()\n\n    def _selected_center_target_name(self):\n        """Return the user-facing name of the selected centering target."""\n        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n\n    def _handle_center_target_change(self):\n        self._update_center_preview_label()\n        self._commit_setting_change("center_target")\n        self.status.set(\n            f"Centering target set to {self._selected_center_target_name()}. "\n            "Actual centering will be implemented with ellipse detection."\n        )\n\n    def _update_center_preview_label(self):\n        target = self._selected_center_target_name()\n        self.center_preview_label.set(f"Full-color image — center on {target}")\n'''
assert old_commit in text
text = text.replace(old_commit, new_commit, 1)

old_renderer = '''    def _refresh_threshold_image(self):\n        """Render authoritative black/white mask for the currently selected T."""\n        if self.gray_image is None:\n            self.threshold_preview = transparent_bgra()\n            self.threshold_photo = None\n            return\n\n        # Exact threshold semantics: dark = gray <= T, light = gray > T.\n        light_mask = cv2.compare(\n            self.gray_image,\n            int(self.threshold.get()),\n            cv2.CMP_GT,\n        )\n        preview = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGRA)\n        preview[:, :, 3] = 255\n        self.threshold_preview = preview\n        self.threshold_photo = None\n'''
new_renderer = '''    def _refresh_threshold_image(self):\n        """Rebuild and display only the B/W raster for the current threshold T."""\n        if self.gray_image is None:\n            self.threshold_preview = transparent_bgra()\n            self.threshold_photo = None\n        else:\n            # Exact semantics: dark = gray <= T, light = gray > T.\n            light_mask = cv2.compare(\n                self.gray_image,\n                int(self.threshold.get()),\n                cv2.CMP_GT,\n            )\n            preview = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGRA)\n            preview[:, :, 3] = 255\n            self.threshold_preview = preview\n            self.threshold_photo = None\n\n        if hasattr(self, "threshold_canvas"):\n            self.threshold_photo = self._show_image_on_canvas(\n                self.threshold_canvas, self.threshold_preview\n            )\n'''
assert old_renderer in text
text = text.replace(old_renderer, new_renderer, 1)

old_refresh = '''    def refresh_preview(self):\n        if self.gray_image is None:\n            self.status.set("Refresh Preview: no readable image is loaded.")\n            return\n        self._refresh_threshold_image()\n        self._redraw_previews()\n        self.status.set(\n            f"Threshold preview regenerated at T={int(self.threshold.get())}."\n        )\n'''
new_refresh = '''    def refresh_preview(self):\n        """Run explicit preview processing for the currently loaded image."""\n        if self.gray_image is None:\n            self.status.set("Refresh Preview: no readable image is loaded.")\n            return\n\n        self._refresh_threshold_image()\n        if hasattr(self, "color_canvas"):\n            image = self.color_image if self.color_image is not None else transparent_bgra()\n            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)\n        self.status.set(\n            f"Threshold preview regenerated at T={int(self.threshold.get())}."\n        )\n'''
assert old_refresh in text
text = text.replace(old_refresh, new_refresh, 1)

text = text.replace("self._refresh_threshold_image()\n        self._redraw_previews()\n        self.status.set(\n            \"Full-resolution threshold preview applied", "self._refresh_threshold_image()\n        self.status.set(\n            \"Full-resolution threshold preview applied", 1)

old_color = '''        if hasattr(self, "color_canvas"):\n            if self.color_image is None:\n                self.color_photo = self._show_image_on_canvas(\n                    self.color_canvas, transparent_bgra()\n                )\n            else:\n                self.color_photo = self._show_image_on_canvas(self.color_canvas, self.color_image)\n'''
new_color = '''        if hasattr(self, "color_canvas"):\n            image = self.color_image if self.color_image is not None else transparent_bgra()\n            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)\n'''
assert old_color in text
text = text.replace(old_color, new_color, 1)

text = text.replace("GUI-only milestone. Load images to inspect the interface.",
                    "Threshold finder integrated. Load images to inspect automatic T selection.")
text = text.replace("Ellipse / Arc Detector — GUI milestone", "Ellipse / Arc Detector — threshold finder")
text = text.replace("GUI milestone", "threshold-finder stage")
text = text.replace("GUI-only actions", "Application actions and threshold preview")
text = text.replace("Selection buttons are GUI placeholders at this milestone.",
                    "Threshold Auto select is implemented; radius Auto select remains a placeholder.")

# ---------------------------------------------------------------------------
# Rewrite threshold integration/event tests around behavior and the new state model.
# Historical snippets are never modified or deleted.
# ---------------------------------------------------------------------------
tests = ROOT / "tests"

(tests / "test_gui_threshold_integration.py").write_text('''"""Behavioral integration checks for threshold state and preview routing."""\n\nfrom types import SimpleNamespace\n\nimport numpy as np\n\nimport circle_arc_detector as appmod\n\n\nclass FakeVariable:\n    def __init__(self, value):\n        self.value = value\n\n    def get(self):\n        return self.value\n\n    def set(self, value):\n        self.value = value\n\n\ndef make_state_app():\n    app = object.__new__(appmod.DetectorApp)\n    app.current_path = "/tmp/image.jpg"\n    app.gray_image = np.array([[0, 9, 10, 255]], dtype=np.uint8)\n    app.threshold = FakeVariable(10)\n    app.min_radius = FakeVariable(1000)\n    app.max_radius = FakeVariable(1500)\n    app.max_error = FakeVariable(8.0)\n    app.min_coverage = FakeVariable(8)\n    app.morphology = FakeVariable(False)\n    app.outer_limb_assistance = FakeVariable(False)\n    app.use_horizon = FakeVariable(True)\n    app.center_target = FakeVariable("light")\n    app.default_settings = {\n        "min_radius": 1000, "max_radius": 1500, "max_error": 8.0,\n        "min_coverage": 8, "morphology": False,\n        "outer_limb_assistance": False, "use_horizon": True,\n        "center_target": "light",\n    }\n    app.setting_variables = {\n        "threshold": app.threshold, "min_radius": app.min_radius,\n        "max_radius": app.max_radius, "max_error": app.max_error,\n        "min_coverage": app.min_coverage, "morphology": app.morphology,\n        "outer_limb_assistance": app.outer_limb_assistance,\n        "use_horizon": app.use_horizon, "center_target": app.center_target,\n    }\n    result = appmod.AutoThresholdResult(\n        threshold=10, histogram_peak=20, histogram_left_edge=10, seed_threshold=15,\n        coarse_threshold=12, roi_seed_threshold=12, full_seed_point=(1, 1),\n        used_guard=False, resolved=True,\n    )\n    app.image_state = {app.current_path: {"settings": {}, "auto_threshold_result": result}}\n    app.refresh_count = 0\n    app._refresh_threshold_image = lambda: setattr(app, "refresh_count", app.refresh_count + 1)\n    return app\n\n\ndef test_changed_setting_is_stored_sparsely_and_refreshes_bw_once():\n    app = make_state_app()\n    app.min_radius.set(900)\n    app._commit_setting_change("min_radius")\n    assert app.image_state[app.current_path]["settings"] == {"min_radius": 900}\n    assert app.refresh_count == 1\n\n\ndef test_setting_returned_to_default_removes_override():\n    app = make_state_app()\n    app.image_state[app.current_path]["settings"]["min_radius"] = 900\n    app.min_radius.set(1000)\n    app._commit_setting_change("min_radius")\n    assert "min_radius" not in app.image_state[app.current_path]["settings"]\n\n\ndef test_threshold_uses_cached_auto_result_as_its_baseline():\n    app = make_state_app()\n    app.threshold.set(14)\n    app._commit_setting_change("threshold")\n    assert app.image_state[app.current_path]["settings"]["threshold"] == 14\n    app.threshold.set(10)\n    app._commit_setting_change("threshold")\n    assert "threshold" not in app.image_state[app.current_path]["settings"]\n\n\ndef test_bw_renderer_uses_exact_gray_greater_than_t_semantics():\n    app = make_state_app()\n    app.threshold_preview = appmod.transparent_bgra()\n    app.threshold_photo = None\n    app._refresh_threshold_image = appmod.DetectorApp._refresh_threshold_image.__get__(app)\n    app._refresh_threshold_image()\n    assert app.threshold_preview.shape == (1, 4, 4)\n    assert app.threshold_preview[0, 0, 0] == 0\n    assert app.threshold_preview[0, 1, 0] == 0\n    assert app.threshold_preview[0, 2, 0] == 0\n    assert app.threshold_preview[0, 3, 0] == 255\n    assert np.all(app.threshold_preview[:, :, 3] == 255)\n''', encoding="utf-8")

(tests / "test_threshold_gui_deferred_preview_refresh.py").write_text('''"""Behavioral checks for deferred slider commits on the threshold branch."""\n\nfrom types import SimpleNamespace\n\nimport circle_arc_detector as appmod\n\n\nclass FakeRoot:\n    def __init__(self):\n        self.jobs = {}\n        self.cancelled = []\n        self.next_job = 1\n\n    def after(self, delay, callback):\n        job = self.next_job\n        self.next_job += 1\n        self.jobs[job] = (delay, callback)\n        return job\n\n    def after_cancel(self, job):\n        self.cancelled.append(job)\n        self.jobs.pop(job, None)\n\n\nclass FakeSlider:\n    def __init__(self, value, setting_name="threshold"):\n        self.value = value\n        self._setting_name = setting_name\n\n    def get(self):\n        return self.value\n\n\ndef make_app():\n    app = object.__new__(appmod.DetectorApp)\n    app.root = FakeRoot()\n    app.slider_keyboard_commit_job = None\n    app.slider_keyboard_widget = None\n    app.slider_keyboard_start_value = None\n    app.commits = []\n    app._commit_setting_change = lambda name: app.commits.append(name)\n    return app\n\n\ndef test_mouse_release_commits_only_the_changed_slider_setting():\n    app = make_app()\n    slider = FakeSlider(8, "threshold")\n    event = SimpleNamespace(widget=slider)\n    app._begin_slider_mouse_change(event)\n    slider.value = 12\n    app._finish_slider_mouse_change(event)\n    assert app.commits == ["threshold"]\n\n\ndef test_mouse_release_without_change_does_not_commit():\n    app = make_app()\n    slider = FakeSlider(8, "min_radius")\n    event = SimpleNamespace(widget=slider)\n    app._begin_slider_mouse_change(event)\n    app._finish_slider_mouse_change(event)\n    assert app.commits == []\n\n\ndef test_keyboard_repeat_release_is_cancelled_until_final_release():\n    app = make_app()\n    slider = FakeSlider(8, "threshold")\n    event = SimpleNamespace(widget=slider)\n    app._begin_slider_keyboard_change(event)\n    slider.value = 9\n    app._schedule_slider_keyboard_commit(event)\n    first_job = app.slider_keyboard_commit_job\n    app._begin_slider_keyboard_change(event)\n    assert first_job in app.root.cancelled\n    slider.value = 12\n    app._schedule_slider_keyboard_commit(event)\n    final_job = app.slider_keyboard_commit_job\n    delay, callback = app.root.jobs[final_job]\n    assert delay == appmod.SLIDER_KEY_RELEASE_SETTLE_MS\n    callback()\n    assert app.commits == ["threshold"]\n''', encoding="utf-8")

preview_clear = tests / "test_gui_preview_clear.py"
if preview_clear.exists():
    preview_clear.unlink()

# Remove the obsolete clear-preview assertion from the alpha test while retaining
# the real alpha-channel behavior tests.
alpha_path = tests / "test_gui_alpha_preview.py"
alpha = alpha_path.read_text(encoding="utf-8")
marker = "def test_setting_changes_replace_threshold_with_transparent_frame():"
if marker in alpha:
    start = alpha.index(marker)
    end_marker = "\ndef test_canvas_is_only_display_surface():"
    end = alpha.index(end_marker, start)
    alpha = alpha[:start] + alpha[end + 1:]
    alpha_path.write_text(alpha, encoding="utf-8")

# Normalize the legacy GUI static tests to the sanitized names without adding
# implementation-prohibition assertions.
for name in ("test_gui_focus_autoselect.py", "test_gui_focus_release.py"):
    path = tests / name
    data = path.read_text(encoding="utf-8")
    for old, new in renames.items():
        data = data.replace(old, new)
    data = data.replace("scale.bind(", "slider.bind(")
    path.write_text(data, encoding="utf-8")

assert text != original
SOURCE.write_text(text, encoding="utf-8")
print("Applied agreed threshold sanitation stage 1")
