"""GUI-first eclipse detector shell.

This milestone intentionally contains the user interface only. Detection,
threshold candidate generation, ellipse fitting, horizon handling, and image
centering are not implemented yet. The interface is based on the latest GUI from
``refactor/cleanup-performance`` and adds a mutually exclusive centering target:
light ellipse (default) or dark ellipse. A clicked slider retains keyboard focus
for arrow-key adjustment until the mouse is clicked anywhere outside that slider.
GUI-only Auto select buttons are provided for threshold and radius, and a
Save centered images placeholder is available beside the image-loading control.
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


class DetectorApp:
    """GUI shell for the eclipse detector rebuild.

    Only interface behavior is implemented here: image loading/navigation,
    controls, preview panes, and the centering-target selector. Detector buttons
    deliberately report that backend functionality has not yet been implemented.
    """

    def __init__(self, root: tk.Tk, image_paths: list[str], args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path: str | None = None
        self.color_image = None

        self.color_photo = None
        self.resize_job = None

        self.threshold = tk.IntVar(value=args.threshold)
        self.min_radius = tk.IntVar(value=round(args.min_radius))
        self.max_radius = tk.IntVar(value=round(args.max_radius))
        self.max_error = tk.DoubleVar(value=args.max_error * 100.0)
        self.min_coverage = tk.IntVar(value=round(args.min_coverage * 100.0))
        self.morphology = tk.BooleanVar(value=False)
        self.outer_limb_assistance = tk.BooleanVar(value=False)
        self.use_horizon = tk.BooleanVar(value=True)

        self.center_target = tk.StringVar(value="light")
        self.center_preview_label = tk.StringVar()

        self.status = tk.StringVar(value="GUI-only milestone. Load images to inspect the interface.")
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector — GUI milestone")
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.build_navigation()
        self.build_controls()
        self.build_previews()
        self.update_center_preview_label()

        root.bind("<ButtonPress-1>", self.release_scale_focus_if_outside, add="+")
        root.bind("<Return>", lambda _event: self.apply_full_resolution())
        root.bind("<Escape>", lambda _event: self.close())

        if self.image_paths:
            self.load_image_at(0)
        else:
            self.update_navigation_state()

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
        self.load_image_at(0)

    def load_image_at(self, index: int):
        if not 0 <= index < len(self.image_paths):
            return
        path = self.image_paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.current_index = index
        self.current_path = path
        self.threshold_canvas.delete("all")
        self.color_canvas.delete("all")
        if image is None:
            self.color_image = None
            self.placeholder(self.threshold_canvas, "Threshold preview\n(detector not implemented)")
            self.placeholder(self.color_canvas, "Unreadable image")
            self.status.set(f"Could not load image: {path}")
        else:
            self.color_image = image
            self.placeholder(self.threshold_canvas, "Threshold preview\n(detector not implemented)")
            self.redraw()
            self.status.set("Image loaded. Detector functionality is intentionally not implemented in this milestone.")
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
        self.previous_button.config(state=tk.NORMAL if has_current and self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if has_current and self.current_index < count - 1 else tk.DISABLED)
        self.preview_button.config(state=tk.NORMAL if readable else tk.DISABLED)
        self.full_button.config(state=tk.NORMAL if readable else tk.DISABLED)
        if has_current:
            self.image_info.set(f"{self.current_index + 1} / {count}   {os.path.basename(self.current_path or '')}")
        else:
            self.image_info.set("No image loaded" if not count else f"0 / {count}")

    def build_navigation(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(4, weight=1)
        tk.Button(frame, text="Load images...", width=14, command=self.load_images).grid(row=0, column=0, padx=(0, 8))
        self.save_centered_button = tk.Button(frame, text="Save centered images", width=20, command=self.save_centered_images)
        self.save_centered_button.grid(row=0, column=1, padx=(0, 10))
        self.previous_button = tk.Button(frame, text="◀ Previous", width=12, command=self.previous_image)
        self.previous_button.grid(row=0, column=2, padx=(0, 5))
        self.next_button = tk.Button(frame, text="Next ▶", width=12, command=self.next_image)
        self.next_button.grid(row=0, column=3, padx=(0, 10))
        tk.Label(frame, textvariable=self.image_info, anchor="w").grid(row=0, column=4, sticky="ew")

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)
        radius_limit = max(1600, round(max(self.args.max_radius, self.args.min_radius) * 1.5))
        rows = [
            ("Brightness threshold (dark <= T, light > T)", self.threshold, 0, 255, 1, lambda v: str(int(float(v)))),
            ("Minimum FINAL fitted semi-axis radius (px)", self.min_radius, 1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum FINAL fitted semi-axis radius (px)", self.max_radius, 1, radius_limit, 1, lambda v: f"{int(float(v))} px"),
            ("Maximum average normalized ellipse error (%)", self.max_error, 0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),
            ("Minimum TOTAL supported ellipse arc (%)", self.min_coverage, 0, 100, 1, lambda v: f"{int(float(v))}% (~{float(v) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)
        self.threshold_auto_button = tk.Button(frame, text="Auto select", width=12, command=self.auto_select_threshold)
        self.threshold_auto_button.grid(row=0, column=3, sticky="ns", padx=(10, 0), pady=2)
        self.radius_auto_button = tk.Button(frame, text="Auto select", width=12, command=self.auto_select_radius)
        self.radius_auto_button.grid(row=1, column=3, rowspan=2, sticky="nsew", padx=(10, 0), pady=2)
        options = tk.Frame(frame)
        options.grid(row=5, column=0, columnspan=4, sticky="w", pady=(4, 3))
        tk.Checkbutton(options, text="Morphology cleanup for candidate search", variable=self.morphology, command=self.pending).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(options, text="Outer-limb assistance", variable=self.outer_limb_assistance, command=self.pending).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(options, text="Use detected horizon", variable=self.use_horizon, command=self.pending, state=tk.DISABLED)
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")
        center_frame = tk.Frame(frame)
        center_frame.grid(row=6, column=0, columnspan=4, sticky="w", pady=(2, 5))
        tk.Label(center_frame, text="Center full-color image on:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Radiobutton(center_frame, text="Light ellipse", variable=self.center_target, value="light", command=self.center_target_changed).grid(row=0, column=1, sticky="w", padx=(0, 14))
        tk.Radiobutton(center_frame, text="Dark ellipse", variable=self.center_target, value="dark", command=self.center_target_changed).grid(row=0, column=2, sticky="w")
        tk.Label(frame, text="Useful threshold candidates:").grid(row=7, column=0, sticky="nw")
        self.palette_frame = tk.Frame(frame)
        self.palette_frame.grid(row=7, column=1, columnspan=3, sticky="w", pady=(0, 8))
        tk.Label(self.palette_frame, text="Not implemented yet", fg="#666666").grid(row=0, column=0, sticky="w")
        button_frame = tk.Frame(frame)
        button_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.preview_button = tk.Button(button_frame, text="Refresh Preview", width=16, command=self.refresh_preview)
        self.preview_button.grid(row=0, column=0, padx=(0, 8))
        self.full_button = tk.Button(button_frame, text="Apply Full Resolution", width=20, command=self.apply_full_resolution)
        self.full_button.grid(row=0, column=1)
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=1150).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=42, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        scale = tk.Scale(parent, from_=low, to=high, orient=tk.HORIZONTAL, resolution=resolution, variable=variable, showvalue=False, length=420, takefocus=True, highlightthickness=1)
        scale.grid(row=row, column=1, sticky="ew", pady=2)
        scale.bind("<ButtonPress-1>", self.focus_scale, add="+")
        scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")
        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)
        def update_value(*_args):
            value_label.config(text=formatter(variable.get()))
            self.pending()
        variable.trace_add("write", update_value)
        update_value()

    @staticmethod
    def focus_scale(event):
        event.widget.focus_set()

    def release_scale_focus_if_outside(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, tk.Scale) and event.widget is not focused:
            self.root.focus_set()

    def build_previews(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")
        tk.Label(frame, text="Threshold preview: arcs / ellipses / detected horizon").grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(frame, textvariable=self.center_preview_label).grid(row=0, column=1, sticky="w", pady=(0, 4))
        self.threshold_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.color_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.threshold_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.color_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.threshold_canvas.bind("<Configure>", self.schedule_redraw)
        self.color_canvas.bind("<Configure>", self.schedule_redraw)
        self.placeholder(self.threshold_canvas, "Threshold preview\n(detector not implemented)")
        self.placeholder(self.color_canvas, "Color image")

    def save_centered_images(self):
        self.status.set("Save centered images: export functionality is not implemented in the GUI milestone.")

    def auto_select_threshold(self):
        self.status.set("Auto select threshold: algorithm not implemented in the GUI milestone.")

    def auto_select_radius(self):
        self.status.set("Auto select radius range: algorithm not implemented in the GUI milestone.")

    def pending(self, *_args):
        self.status.set("Settings changed. This milestone contains the GUI only; no detector recomputation is performed.")

    def center_target_changed(self):
        self.update_center_preview_label()
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.status.set(f"Centering target set to {target}. Actual centering will be implemented with ellipse detection.")

    def update_center_preview_label(self):
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.center_preview_label.set(f"Full-color image — center on {target}")

    def refresh_preview(self):
        self.status.set("Refresh Preview: detector backend not implemented in the GUI milestone.")

    def apply_full_resolution(self):
        self.status.set("Apply Full Resolution: detector backend not implemented in the GUI milestone.")

    def schedule_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        self.resize_job = None
        if self.color_image is not None:
            self.color_photo = self.show_image(self.color_canvas, self.color_image)

    @staticmethod
    def show_image(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(image, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(canvas_width // 2 + 1, canvas_height // 2 + 1, image=photo, anchor="center")
        return photo

    @staticmethod
    def placeholder(canvas, text):
        canvas.create_text(160, 120, text=text, fill="#cccccc", justify="center")

    def close(self):
        self.root.destroy()


def build_parser():
    parser = argparse.ArgumentParser(description="GUI milestone for the eclipse detector rebuild.")
    parser.add_argument("images", nargs="*", help="Optional ordered input image list; more images can be loaded in the GUI")
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
