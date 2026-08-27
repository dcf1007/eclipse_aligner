"""GUI foundation for the eclipse alignment application.

This branch defines the user-interface contract independently from the detector
backend. It intentionally contains image loading and navigation, control layout,
preview surfaces, focus behavior, completed-setting event handling, and backend
hooks, while ellipse fitting, horizon detection, centering, export, automatic
threshold selection, automatic radius selection, and threshold-raster generation
remain outside this GUI-only implementation. The interface descends from the
``refactor/cleanup-performance`` GUI and retains its established layout and
keyboard interaction model.

The centering target is represented by one shared Tk ``StringVar`` and two mutually
exclusive radio buttons: Light ellipse is the default and Dark ellipse is the
alternative. Sliders retain keyboard focus after a click so arrow keys can make
precise adjustments; clicking elsewhere releases that focus. A slider's numeric
label updates continuously while it moves, but setting completion is delayed until
mouse release or the final keyboard key release. Keyboard auto-repeat may emit
intermediate release/press pairs on some Tk platforms, so those releases are
coalesced through a short settling window. Checkboxes, radio buttons, and Auto
select controls commit immediately because their interaction is already discrete.

Completed setting changes are deliberately separate from explicit preview
processing. They call ``_commit_setting_change()``, whose only responsibility is
to invoke the lightweight ``_refresh_threshold_image()`` hook. In a backend-enabled
branch that hook may rebuild the B/W image from the already-loaded grayscale image
and the current threshold, but it must not run ellipse, horizon, candidate, or
other Refresh Preview processing. The full ``refresh_preview()`` action is reserved
for an explicit Refresh Preview request or for initialization after a readable
image is loaded. In this GUI-only branch both backend hooks remain placeholders and
no grayscale or B/W threshold backend is implemented.

Preview rasters contain no placeholder text. Empty preview regions are represented
as real BGRA image data with alpha=0, while loaded source-image pixels are made
fully opaque. Transparency is retained in the raster itself rather than simulated
by matching the Tk canvas background. The Save centered images control, threshold
Auto select, radius Auto select, Refresh Preview, and Apply Full Resolution actions
remain present as interface/backend boundaries for later implementation.
"""

from __future__ import annotations

import argparse
import base64
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)

SLIDER_KEY_RELEASE_SETTLE_MS = 45
PREVIEW_REDRAW_DELAY_MS = 60


def transparent_bgra(width: int = 1, height: int = 1) -> np.ndarray:
    """Return a BGRA frame whose pixels are fully transparent (alpha = 0)."""
    width = max(1, int(width))
    height = max(1, int(height))
    return np.zeros((height, width, 4), dtype=np.uint8)


def opaque_bgra(bgr: np.ndarray) -> np.ndarray:
    """Convert a normal OpenCV BGR image to BGRA with fully opaque image pixels."""
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = 255
    return bgra


class DetectorApp:
    """Own the GUI state, widgets, interaction rules, and preview hooks.

    The class manages the ordered image list, alpha-aware preview rasters, settings,
    navigation, slider focus, completed-setting commits, and canvas redraws. Public
    methods correspond to application actions; underscore-prefixed methods are
    implementation details used by Tk callbacks or rendering internals. Detector
    work is intentionally represented only by backend hooks in this branch.
    """

    def __init__(self, root: tk.Tk, image_paths: list[str], args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path: str | None = None
        self.color_image = None
        self.threshold_preview = transparent_bgra()

        self.threshold_photo = None
        self.color_photo = None
        self.preview_redraw_job = None

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

        self.status = tk.StringVar(value="GUI foundation. Load images to inspect the interface.")
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector — GUI foundation")
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
        root.bind("<Return>", lambda _event: self.apply_full_resolution())
        root.bind("<Escape>", lambda _event: self.close())

        if self.image_paths:
            self.load_image_at(0)
        else:
            self.update_navigation_state()

    # ------------------------------------------------------------------
    # Image list / navigation (GUI support only)
    # ------------------------------------------------------------------
    def load_images(self):
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
        self.threshold_preview = transparent_bgra()
        self.load_image_at(0)

    def load_image_at(self, index: int):
        if not 0 <= index < len(self.image_paths):
            return

        path = self.image_paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.current_index = index
        self.current_path = path

        if image is None:
            self.color_image = None
            self.threshold_preview = transparent_bgra()
            self.threshold_photo = None
            self.color_photo = None
            self._redraw_previews()
            self.status.set(f"Could not load image: {path}")
        else:
            # The preview pipeline is alpha-aware from the start. Source pixels are
            # opaque; future centered output may use alpha=0 outside image data.
            self.color_image = opaque_bgra(image)
            height, width = image.shape[:2]
            self.threshold_preview = transparent_bgra(width, height)
            self._redraw_previews()
            self.refresh_preview()
            self.status.set(
                "Image loaded. Detector functionality is intentionally not implemented in the GUI foundation."
            )

        self.update_navigation_state()

    def previous_image(self):
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_image(self):
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    def update_navigation_state(self):
        count = len(self.image_paths)
        has_current = 0 <= self.current_index < count
        readable = has_current and self.color_image is not None

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

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_navigation_bar(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(4, weight=1)

        tk.Button(frame, text="Load images...", width=14, command=self.load_images).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.save_centered_button = tk.Button(
            frame,
            text="Save centered images",
            width=20,
            command=self.save_centered_images,
        )
        self.save_centered_button.grid(row=0, column=1, padx=(0, 10))
        self.previous_button = tk.Button(
            frame, text="◀ Previous", width=12, command=self.previous_image
        )
        self.previous_button.grid(row=0, column=2, padx=(0, 5))
        self.next_button = tk.Button(
            frame, text="Next ▶", width=12, command=self.next_image
        )
        self.next_button.grid(row=0, column=3, padx=(0, 10))
        tk.Label(frame, textvariable=self.image_info, anchor="w").grid(
            row=0, column=4, sticky="ew"
        )

    def _build_settings_panel(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        radius_limit = max(1600, round(max(self.args.max_radius, self.args.min_radius) * 1.5))

        slider_specs = [
            ("Brightness threshold (dark <= T, light > T)", self.threshold,
             0, 255, 1, lambda v: str(int(float(v)))),
            ("Minimum FINAL fitted semi-axis radius (px)", self.min_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum FINAL fitted semi-axis radius (px)", self.max_radius,
             1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum average normalized ellipse error (%)", self.max_error,
             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),
            ("Minimum TOTAL supported ellipse arc (%)", self.min_coverage,
             0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(slider_specs):
            self._add_slider(frame, row, *spec)

        # Auto-select controls are backend boundaries. Threshold has its own
        # button; radius selection is one operation spanning the paired
        # minimum/maximum radius rows.
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
            command=self._commit_setting_change,
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=self._commit_setting_change,
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=self._commit_setting_change,
            state=tk.DISABLED,
        )
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")

        # New requested alignment target. Sharing center_target makes these two
        # controls mutually exclusive without extra synchronization logic.
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

    def _add_slider(self, parent, row, text, variable, low, high, resolution, formatter):
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
            # Do not refresh here: this trace fires continuously while the Scale is
            # dragged or while an arrow key auto-repeats.
            value_label.config(text=formatter(variable.get()))

        variable.trace_add("write", update_value)
        update_value()

    @staticmethod
    def _focus_slider(event):
        """Keep a clicked slider focused so arrow keys continue to adjust it."""
        event.widget.focus_set()

    @staticmethod
    def _begin_slider_mouse_change(event):
        """Remember the value before a mouse slider interaction begins."""
        event.widget._preview_mouse_start_value = event.widget.get()

    def _finish_slider_mouse_change(self, event):
        """Refresh once after a mouse slider change has actually finished."""
        start_value = getattr(event.widget, "_preview_mouse_start_value", None)
        if start_value is not None and event.widget.get() != start_value:
            self._commit_setting_change()

    def _cancel_pending_slider_keyboard_commit(self):
        """Cancel a release callback superseded by continuing keyboard input."""
        if self.slider_keyboard_commit_job is not None:
            self.root.after_cancel(self.slider_keyboard_commit_job)
            self.slider_keyboard_commit_job = None

    def _begin_slider_keyboard_change(self, event):
        """Begin or continue one keyboard slider interaction."""
        self._cancel_pending_slider_keyboard_commit()
        if self.slider_keyboard_widget is not event.widget:
            self.slider_keyboard_widget = event.widget
            self.slider_keyboard_start_value = event.widget.get()

    def _schedule_slider_keyboard_commit(self, event):
        """Schedule completion after a KeyRelease survives the repeat window."""
        if self.slider_keyboard_widget is not event.widget:
            self.slider_keyboard_widget = event.widget
            self.slider_keyboard_start_value = event.widget.get()
        self._cancel_pending_slider_keyboard_commit()
        self.slider_keyboard_commit_job = self.root.after(
            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change
        )

    def _finish_slider_keyboard_change(self):
        """Commit one preview refresh after keyboard slider input becomes idle."""
        self.slider_keyboard_commit_job = None
        widget = self.slider_keyboard_widget
        start_value = self.slider_keyboard_start_value
        self.slider_keyboard_widget = None
        self.slider_keyboard_start_value = None
        if widget is not None and start_value is not None and widget.get() != start_value:
            self._commit_setting_change()

    def _release_slider_focus_if_clicked_elsewhere(self, event):
        """Release slider focus immediately when the mouse clicks elsewhere.

        Clicking the focused slider itself keeps focus. Clicking a different slider
        transfers focus through that slider's own ButtonPress binding, so this
        handler also leaves it alone. Any non-slider click removes the keyboard
        focus ring from the previously focused slider.
        """
        focused = self.root.focus_get()
        if isinstance(focused, tk.Scale) and event.widget is not focused:
            self.root.focus_set()

    def _build_preview_panes(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(
            frame,
            text="Threshold preview: arcs / ellipses / detected horizon",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            textvariable=self.center_preview_label,
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        # The canvas is only a display surface. Transparency is retained in the
        # BGRA preview raster itself; it is not simulated by matching Tk colors.
        self.threshold_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.color_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.threshold_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.color_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.threshold_canvas.bind("<Configure>", self._schedule_preview_redraw)
        self.color_canvas.bind("<Configure>", self._schedule_preview_redraw)

    # ------------------------------------------------------------------
    # GUI-only actions
    # ------------------------------------------------------------------
    def save_centered_images(self):
        self.status.set(
            "Save centered images: export functionality is not implemented in the GUI foundation."
        )

    def auto_select_threshold(self):
        self._commit_setting_change()
        self.status.set(
            "Auto select threshold: algorithm not implemented in the GUI foundation."
        )

    def auto_select_radius(self):
        self.status.set(
            "Auto select radius range: algorithm not implemented in the GUI foundation."
        )

    def _commit_setting_change(self, *_args):
        """Apply only the lightweight threshold-image consequence of a setting change."""
        self._refresh_threshold_image()

    def _selected_center_target_name(self):
        """Return the user-facing name of the selected centering target."""
        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"

    def _handle_center_target_change(self):
        self._update_center_preview_label()
        self._commit_setting_change()
        self.status.set(
            f"Centering target set to {self._selected_center_target_name()}. "
            "Actual centering will be implemented with ellipse detection."
        )

    def _update_center_preview_label(self):
        target = self._selected_center_target_name()
        self.center_preview_label.set(f"Full-color image — center on {target}")

    def _refresh_threshold_image(self):
        """Backend hook for rebuilding only the current-threshold B/W image.

        The GUI-only branch intentionally has no grayscale/threshold backend.
        Backend-enabled branches implement this hook without running the broader
        Refresh Preview processing pipeline.
        """

    def refresh_preview(self):
        self.status.set("Refresh Preview: detector backend not implemented in the GUI foundation.")

    def apply_full_resolution(self):
        self.status.set("Apply Full Resolution: detector backend not implemented in the GUI foundation.")

    def _schedule_preview_redraw(self, _event=None):
        if self.preview_redraw_job is not None:
            self.root.after_cancel(self.preview_redraw_job)
        self.preview_redraw_job = self.root.after(
            PREVIEW_REDRAW_DELAY_MS, self._redraw_previews
        )

    def _redraw_previews(self):
        self.preview_redraw_job = None
        if hasattr(self, "threshold_canvas"):
            self.threshold_photo = self._show_image_on_canvas(
                self.threshold_canvas, self.threshold_preview
            )
        if hasattr(self, "color_canvas"):
            image = self.color_image if self.color_image is not None else transparent_bgra()
            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)

    @staticmethod
    def _show_image_on_canvas(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(image, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
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
        return photo

    def close(self):
        self.root.destroy()


def build_parser():
    parser = argparse.ArgumentParser(description="GUI foundation for the eclipse detector rebuild.")
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional ordered input image list; more images can be loaded in the GUI",
    )
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--min-radius", type=float, default=1000.0)
    parser.add_argument("--max-radius", type=float, default=1500.0)
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    return parser


def validate_args(args, parser):
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be 0..255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be > 0")
    if args.max_radius <= 0:
        parser.error("--max-radius must be > 0")
    if args.max_radius < args.min_radius:
        parser.error("--max-radius must be >= --min-radius")
    if args.max_error <= 0:
        parser.error("--max-error must be > 0")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be 0..1")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    root = tk.Tk()
    DetectorApp(root, args.images, args)
    root.mainloop()


if __name__ == "__main__":
    main()
