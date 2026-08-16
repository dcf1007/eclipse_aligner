import argparse
import math
import time
import tkinter as tk

import cv2
import numpy as np


MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)


def fit_circle_least_squares(points):
    """Fit a circle to Nx2 points with a fast centered algebraic fit."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]
    xm = float(x.mean())
    ym = float(y.mean())
    u = x - xm
    v = y - ym
    z = u * u + v * v

    suu = float(np.dot(u, u))
    svv = float(np.dot(v, v))
    suv = float(np.dot(u, v))
    suz = float(np.dot(u, z))
    svz = float(np.dot(v, z))

    det = suu * svv - suv * suv
    if abs(det) <= 1e-12 * (suu * svv + 1.0):
        return None

    uc = 0.5 * (suz * svv - svz * suv) / det
    vc = 0.5 * (svz * suu - suz * suv) / det
    cx = xm + uc
    cy = ym + vc

    radius_squared = float(z.mean()) + uc * uc + vc * vc
    if radius_squared <= 0.0 or not np.isfinite(radius_squared):
        return None

    radius = math.sqrt(radius_squared)
    distances = np.hypot(x - cx, y - cy)
    mean_error = float(np.mean(np.abs(distances - radius)))
    if not all(np.isfinite(value) for value in (cx, cy, radius, mean_error)):
        return None

    return float(cx), float(cy), float(radius), mean_error


def angular_coverage(points, cx, cy):
    """Estimate angular coverage of an ordered contiguous contour region."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    unwrapped = np.unwrap(angles)
    return float(np.clip(np.ptp(unwrapped) / (2.0 * np.pi), 0.0, 1.0))


def _candidate_window_lengths(point_count, min_region_points):
    min_region_points = max(3, min(min_region_points, point_count))
    lengths = {min_region_points, point_count}
    length = min_region_points

    while length < point_count:
        lengths.add(length)
        length = min(point_count, max(length + 1, int(round(length * 1.45))))

    return sorted(lengths)


def _evaluate_circular_region(
    extended_points,
    contour_point_count,
    start,
    length,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
):
    if length < 3 or length > contour_point_count:
        return None

    start %= contour_point_count
    region = extended_points[start : start + length]
    fitted = fit_circle_least_squares(region)
    if fitted is None:
        return None

    cx, cy, radius, mean_error = fitted
    if radius < min_radius or radius > max_radius:
        return None

    relative_error = mean_error / radius
    if relative_error > max_relative_error:
        return None

    coverage = angular_coverage(region, cx, cy)
    if coverage < min_coverage:
        return None

    support_fraction = length / contour_point_count
    score = (
        (coverage**1.5)
        * math.sqrt(float(length))
        * (0.75 + 0.25 * math.sqrt(support_fraction))
        / (relative_error + 0.002)
    )

    return {
        "center": (cx, cy),
        "radius": radius,
        "error": mean_error,
        "relative_error": relative_error,
        "coverage": coverage,
        "points": region.copy(),
        "region_start": start,
        "region_length": length,
        "contour_points": contour_point_count,
        "score": score,
    }


def _hill_climb_region(
    candidate,
    extended_points,
    contour_point_count,
    min_region_points,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
):
    best = candidate
    moves = ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0))

    for _ in range(40):
        improved = False
        current_start = best["region_start"]
        current_length = best["region_length"]

        for start_delta, length_delta in moves:
            new_length = current_length + length_delta
            if new_length < min_region_points or new_length > contour_point_count:
                continue

            trial = _evaluate_circular_region(
                extended_points,
                contour_point_count,
                current_start + start_delta,
                new_length,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1.0 + 1e-9):
                best = trial
                improved = True

        if not improved:
            break

    return best


def find_best_circular_region(
    points,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    min_region_points=12,
):
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    if point_count < max(3, min_region_points):
        return None

    min_region_points = min(min_region_points, point_count)
    extended_points = np.vstack((points, points))
    coarse_candidates = []

    for length in _candidate_window_lengths(point_count, min_region_points):
        step = max(1, length // 5)
        for start in range(0, point_count, step):
            candidate = _evaluate_circular_region(
                extended_points,
                point_count,
                start,
                length,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
            )
            if candidate is not None:
                coarse_candidates.append(candidate)

    if not coarse_candidates:
        return None

    coarse_candidates.sort(key=lambda item: item["score"], reverse=True)
    best = None
    for seed in coarse_candidates[:4]:
        refined = _hill_climb_region(
            seed,
            extended_points,
            point_count,
            min_region_points,
            min_radius,
            max_radius,
            max_relative_error,
            min_coverage,
        )
        if best is None or refined["score"] > best["score"]:
            best = refined

    return best


def _downsample_contour_points(points, max_search_points):
    point_count = len(points)
    if max_search_points <= 0 or point_count <= max_search_points:
        return points

    indices = np.linspace(0, point_count - 1, max_search_points, dtype=np.int32)
    return points[indices]


def detect_circular_objects(
    binary,
    max_objects=2,
    min_contour_points=12,
    min_radius=10.0,
    max_radius=None,
    max_relative_error=0.08,
    min_coverage=0.12,
    max_contours=100,
    max_search_points=500,
):
    height, width = binary.shape[:2]
    if max_radius is None:
        max_radius = float(max(width, height))

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_region_points = max(8, min_contour_points)

    usable = []
    for contour in contours:
        if len(contour) < min_region_points:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter < max(12.0, min_radius * min_coverage * math.pi):
            continue
        usable.append((perimeter, contour))

    usable.sort(key=lambda item: item[0], reverse=True)
    if max_contours > 0:
        usable = usable[:max_contours]

    candidates = []
    for _, contour in usable:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        points = _downsample_contour_points(points, max_search_points)
        candidate = find_best_circular_region(
            points,
            min_radius=min_radius,
            max_radius=max_radius,
            max_relative_error=max_relative_error,
            min_coverage=min_coverage,
            min_region_points=min_region_points,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    selected = []

    for candidate in candidates:
        cx, cy = candidate["center"]
        radius = candidate["radius"]
        duplicate = False

        for existing in selected:
            ex, ey = existing["center"]
            existing_radius = existing["radius"]
            scale = max(radius, existing_radius)
            if (
                math.hypot(cx - ex, cy - ey) < 0.15 * scale
                and abs(radius - existing_radius) < 0.15 * scale
            ):
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_objects:
            break

    return selected


def process_image(
    image,
    threshold_value,
    invert=False,
    min_radius=10.0,
    max_radius=None,
    max_relative_error=0.08,
    min_coverage=0.12,
    morphology=False,
    max_contours=100,
    max_search_points=500,
    gray=None,
):
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, int(threshold_value), 255, threshold_mode)

    binary_for_detection = binary
    if morphology:
        binary_for_detection = cv2.morphologyEx(binary, cv2.MORPH_OPEN, MORPH_KERNEL)

    detections = detect_circular_objects(
        binary_for_detection,
        max_objects=2,
        min_contour_points=12,
        min_radius=min_radius,
        max_radius=max_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
        max_contours=max_contours,
        max_search_points=max_search_points,
    )

    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        cx, cy = detection["center"]
        radius = detection["radius"]
        center = (int(round(cx)), int(round(cy)))
        radius_int = int(round(radius))

        support_points = np.rint(detection["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support_points) >= 2:
            cv2.polylines(output, [support_points], False, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.circle(output, center, radius_int, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

        coverage_degrees = detection["coverage"] * 360.0
        label = (
            f"{index}: r={radius:.1f} "
            f"arc={coverage_degrees:.0f}deg "
            f"err={detection['relative_error']:.3f} "
            f"support={detection['region_length']}/{detection['contour_points']}"
        )
        cv2.putText(
            output,
            label,
            (center[0] + 10, max(18, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return binary, binary_for_detection, output, detections


def build_threshold_palette(image, gray, max_colors=12, min_shade_gap=15):
    """Return dominant source colors whose grayscale shades are meaningfully distinct."""
    if max_colors <= 0:
        return []

    pixel_count = gray.size
    sample_limit = 200_000
    stride = max(1, int(math.ceil(math.sqrt(pixel_count / sample_limit))))
    sample_gray = gray[::stride, ::stride].reshape(-1)
    sample_bgr = image[::stride, ::stride].reshape(-1, 3)

    hist = np.bincount(sample_gray, minlength=256).astype(np.float64)
    smooth_radius = max(2, min_shade_gap // 3)
    kernel = np.ones(2 * smooth_radius + 1, dtype=np.float64)
    density = np.convolve(hist, kernel, mode="same")

    selected_shades = []
    for shade in np.argsort(density)[::-1]:
        shade = int(shade)
        if density[shade] <= 0:
            break
        if any(abs(shade - existing) < min_shade_gap for existing in selected_shades):
            continue
        selected_shades.append(shade)
        if len(selected_shades) >= max_colors:
            break

    palette = []
    color_radius = max(3, min_shade_gap // 3)
    sample_gray_i16 = sample_gray.astype(np.int16, copy=False)

    for shade in selected_shades:
        mask = np.abs(sample_gray_i16 - shade) <= color_radius
        if np.any(mask):
            matching_gray = sample_gray[mask]
            matching_bgr = sample_bgr[mask]
            threshold = int(round(float(np.average(matching_gray))))
            bgr = tuple(int(round(value)) for value in np.median(matching_bgr, axis=0))
        else:
            threshold = shade
            bgr = (shade, shade, shade)

        palette.append({"threshold": int(np.clip(threshold, 0, 255)), "bgr": bgr})

    palette.sort(key=lambda item: item["threshold"])
    deduped = []
    for item in palette:
        if not deduped or item["threshold"] - deduped[-1]["threshold"] >= min_shade_gap:
            deduped.append(item)
        elif hist[item["threshold"]] > hist[deduped[-1]["threshold"]]:
            deduped[-1] = item

    return deduped


def _bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _contrasting_text_color(bgr):
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return "#111111" if luminance >= 150 else "#f7f7f7"


def _make_display(binary_for_detection, result):
    """Build the OpenCV preview without any interactive controls."""
    threshold_preview = cv2.cvtColor(binary_for_detection, cv2.COLOR_GRAY2BGR)
    body = np.hstack((threshold_preview, result))

    max_width = 1600
    max_height = 900
    scale = min(1.0, max_width / body.shape[1], max_height / body.shape[0])
    if scale < 1.0:
        body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return body


def _print_detections(detections):
    for index, detection in enumerate(detections, start=1):
        cx, cy = detection["center"]
        print(
            f"Object {index}: "
            f"center=({cx:.2f}, {cy:.2f}), "
            f"radius={detection['radius']:.2f}, "
            f"coverage={detection['coverage'] * 360.0:.1f} deg, "
            f"relative_error={detection['relative_error']:.4f}, "
            f"support_points={detection['region_length']}/{detection['contour_points']}"
        )


class ToolTip:
    """Small delayed hover description for a Tk widget."""

    def __init__(self, widget, text, delay_ms=450, wrap_length=360):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wrap_length = wrap_length
        self._after_id = None
        self._tip_window = None

        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._tip_window is not None or not self.text:
            return

        try:
            x = self.widget.winfo_pointerx() + 14
            y = self.widget.winfo_pointery() + 18
        except tk.TclError:
            return

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=self.wrap_length,
            background="#fffde7",
            foreground="#202020",
        )
        label.pack()
        self._tip_window = tip

    def _hide(self, _event=None):
        self._cancel()
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except tk.TclError:
                pass
            self._tip_window = None


def _attach_tooltip(widgets, text):
    for widget in widgets:
        ToolTip(widget, text)


class CircleArcDetectorApp:
    def __init__(self, root, image, gray, palette, args):
        self.root = root
        self.image = image
        self.gray = gray
        self.palette = palette
        self.args = args
        self.window_name = "Circle / Arc Detection"

        self.result = None
        self.detections = []
        self.applied_values = None
        self.palette_buttons = []
        self._closing = False

        image_max_dim = max(gray.shape)
        initial_max_radius = args.max_radius if args.max_radius > 0 else float(image_max_dim)
        initial_max_radius = max(initial_max_radius, args.min_radius)
        self.radius_slider_limit = max(
            100,
            int(math.ceil(image_max_dim * 2.0)),
            int(math.ceil(args.min_radius)),
            int(math.ceil(initial_max_radius)),
        )

        self.threshold_var = tk.IntVar(value=args.threshold)
        self.min_radius_var = tk.DoubleVar(value=args.min_radius)
        self.max_radius_var = tk.DoubleVar(value=initial_max_radius)
        self.max_error_percent_var = tk.DoubleVar(value=args.max_error * 100.0)
        self.min_coverage_percent_var = tk.DoubleVar(value=args.min_coverage * 100.0)

        self.status_var = tk.StringVar(value="Adjust settings, then click Apply.")

        self._configure_window()
        self._build_controls()
        self._bind_pending_change_tracking()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Return>", lambda _event: self.apply())
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Escape>", lambda _event: self.close())

        self.root.after(0, self.apply)
        self.root.after(30, self._pump_opencv)

    def _configure_window(self):
        self.root.title("Circle / Arc Detector Controls")
        self.root.resizable(False, False)

    def _build_controls(self):
        outer = tk.Frame(self.root, padx=14, pady=12)
        outer.pack(fill="both", expand=True)

        title = tk.Label(
            outer,
            text="Circle / Arc Detection Settings",
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        row = 1
        self._add_slider_row(
            outer,
            row,
            "Brightness threshold (0=black, 255=white)",
            self.threshold_var,
            0,
            255,
            1,
            lambda value: f"{int(round(float(value)))}",
            (
                "Pixels brighter than this threshold become white and pixels darker than it "
                "become black (reversed when --invert is used). You can also set this value "
                "by clicking one of the sampled colors below. Changing it does not process "
                "the image until Apply is clicked."
            ),
        )
        row += 1

        self._add_slider_row(
            outer,
            row,
            "Minimum fitted circle radius (px)",
            self.min_radius_var,
            1,
            self.radius_slider_limit,
            1,
            lambda value: f"{float(value):.0f} px",
            (
                "Reject fitted circles smaller than this radius. This limits the physical "
                "size of circles/arcs that can be accepted; it does not control how much of "
                "the circle must be visible."
            ),
        )
        row += 1

        self._add_slider_row(
            outer,
            row,
            "Maximum fitted circle radius (px)",
            self.max_radius_var,
            1,
            self.radius_slider_limit,
            1,
            lambda value: f"{float(value):.0f} px",
            (
                "Reject fitted circles larger than this radius. If it is set below the minimum "
                "radius, Apply automatically raises it to match the minimum."
            ),
        )
        row += 1

        self._add_slider_row(
            outer,
            row,
            "Maximum average radial error (% of radius)",
            self.max_error_percent_var,
            0.1,
            50.0,
            0.1,
            lambda value: f"{float(value):.1f}%",
            (
                "Controls how closely contour points must follow the fitted circle. The detector "
                "computes each point's radial distance from the fitted center, measures its "
                "absolute difference from the fitted radius, averages those differences, then "
                "divides by the radius. Example: 8% permits about 8 px average radial deviation "
                "for a 100 px radius circle. Lower values are stricter; higher values accept "
                "rougher or less circular contours."
            ),
        )
        row += 1

        self._add_slider_row(
            outer,
            row,
            "Minimum visible circle arc (%)",
            self.min_coverage_percent_var,
            0,
            100,
            1,
            lambda value: f"{float(value):.0f}%  (~{float(value) * 3.6:.0f}°)",
            (
                "Minimum percentage of the full 360° circle that the contour must cover. "
                "For example, 25% requires roughly a 90° arc and 50% requires roughly 180°. "
                "Higher values reduce false positives from very short curved fragments."
            ),
        )
        row += 1

        palette_label = tk.Label(outer, text="Pick threshold from image colors", anchor="w")
        palette_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 4))
        ToolTip(
            palette_label,
            "Representative colors sampled from the input image. Similar brightness shades are "
            "merged so the list stays useful. Clicking a swatch changes only the pending "
            "threshold; click Apply to recompute the image.",
        )
        row += 1

        palette_frame = tk.Frame(outer)
        palette_frame.grid(row=row, column=0, columnspan=3, sticky="w")
        self._build_palette_buttons(palette_frame)
        row += 1

        button_frame = tk.Frame(outer)
        button_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(14, 8))

        self.apply_button = tk.Button(
            button_frame,
            text="Apply settings",
            width=16,
            command=self.apply,
        )
        self.apply_button.pack(side="left")
        ToolTip(
            self.apply_button,
            "Run thresholding, contour analysis, circle fitting, and redraw the OpenCV result "
            "using the current pending settings.",
        )

        self.save_button = tk.Button(
            button_frame,
            text="Save current result",
            width=18,
            command=self.save,
            state=tk.DISABLED,
        )
        self.save_button.pack(side="left", padx=(8, 0))
        ToolTip(
            self.save_button,
            f"Save the most recently applied annotated result to: {self.args.output}",
        )

        status = tk.Label(
            outer,
            textvariable=self.status_var,
            justify="left",
            anchor="w",
            wraplength=720,
        )
        status.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(2, 0))

        outer.grid_columnconfigure(1, weight=1)

    def _add_slider_row(
        self,
        parent,
        row,
        label_text,
        variable,
        minimum,
        maximum,
        resolution,
        formatter,
        tooltip_text,
    ):
        label = tk.Label(parent, text=label_text, width=42, anchor="w")
        label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)

        scale = tk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=variable,
            showvalue=False,
            length=360,
            highlightthickness=0,
        )
        scale.grid(row=row, column=1, sticky="ew", pady=3)

        value_label = tk.Label(parent, width=15, anchor="e")
        value_label.grid(row=row, column=2, sticky="e", padx=(8, 0), pady=3)

        def refresh_value(*_args):
            value_label.config(text=formatter(variable.get()))

        variable.trace_add("write", refresh_value)
        refresh_value()
        _attach_tooltip((label, scale, value_label), tooltip_text)

    def _build_palette_buttons(self, parent):
        self.palette_buttons.clear()

        for index, item in enumerate(self.palette):
            threshold = item["threshold"]
            bgr = item["bgr"]
            button = tk.Button(
                parent,
                text=str(threshold),
                width=4,
                height=2,
                background=_bgr_to_hex(bgr),
                foreground=_contrasting_text_color(bgr),
                activebackground=_bgr_to_hex(bgr),
                activeforeground=_contrasting_text_color(bgr),
                command=lambda value=threshold: self._select_palette_threshold(value),
                relief=tk.RAISED,
                borderwidth=2,
            )
            button.grid(row=index // 8, column=index % 8, padx=3, pady=3)
            ToolTip(
                button,
                f"Set the pending threshold to brightness {threshold}. This color is sampled "
                "from the input image. The image will not be reprocessed until Apply is clicked.",
            )
            self.palette_buttons.append((threshold, button))

        self._refresh_palette_selection()

    def _select_palette_threshold(self, threshold):
        self.threshold_var.set(int(threshold))
        self._refresh_palette_selection()
        self.status_var.set(
            f"Pending threshold set to {threshold} from the color palette. Click Apply to recompute."
        )

    def _refresh_palette_selection(self):
        selected_threshold = int(round(self.threshold_var.get()))
        for threshold, button in self.palette_buttons:
            if threshold == selected_threshold:
                button.config(relief=tk.SUNKEN, borderwidth=4)
            else:
                button.config(relief=tk.RAISED, borderwidth=2)

    def _bind_pending_change_tracking(self):
        variables = (
            self.threshold_var,
            self.min_radius_var,
            self.max_radius_var,
            self.max_error_percent_var,
            self.min_coverage_percent_var,
        )

        for variable in variables:
            variable.trace_add("write", self._mark_pending)

    def _mark_pending(self, *_args):
        self._refresh_palette_selection()
        if self.applied_values is not None:
            self.status_var.set("Settings changed. Click Apply settings to recompute the result.")

    def _current_settings(self):
        threshold = int(round(self.threshold_var.get()))
        min_radius = max(1.0, float(self.min_radius_var.get()))
        max_radius = max(1.0, float(self.max_radius_var.get()))
        if max_radius < min_radius:
            max_radius = min_radius
            self.max_radius_var.set(max_radius)

        max_relative_error = max(0.001, float(self.max_error_percent_var.get()) / 100.0)
        min_coverage = float(np.clip(self.min_coverage_percent_var.get() / 100.0, 0.0, 1.0))

        return threshold, min_radius, max_radius, max_relative_error, min_coverage

    def apply(self):
        if self._closing:
            return

        threshold, min_radius, max_radius, max_relative_error, min_coverage = self._current_settings()

        self.apply_button.config(state=tk.DISABLED)
        self.status_var.set("Processing current settings...")
        self.root.update_idletasks()

        started = time.perf_counter()
        try:
            _, binary_for_detection, result, detections = process_image(
                self.image,
                threshold,
                invert=self.args.invert,
                min_radius=min_radius,
                max_radius=max_radius,
                max_relative_error=max_relative_error,
                min_coverage=min_coverage,
                morphology=self.args.morphology,
                max_contours=self.args.max_contours,
                max_search_points=self.args.max_search_points,
                gray=self.gray,
            )
        finally:
            self.apply_button.config(state=tk.NORMAL)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.result = result
        self.detections = detections
        self.applied_values = (
            threshold,
            min_radius,
            max_radius,
            max_relative_error,
            min_coverage,
        )

        display = _make_display(binary_for_detection, result)
        cv2.imshow(self.window_name, display)
        cv2.waitKey(1)

        self.save_button.config(state=tk.NORMAL)
        self._refresh_palette_selection()
        self.status_var.set(
            f"Applied: threshold={threshold}; radius={min_radius:.0f}-{max_radius:.0f}px; "
            f"max avg radial error={max_relative_error:.1%}; minimum visible arc={min_coverage:.0%}; "
            f"{len(detections)} object(s); {elapsed_ms:.1f} ms."
        )

        print(
            f"Applied: threshold={threshold}, "
            f"min_radius={min_radius:.0f}, max_radius={max_radius:.0f}, "
            f"max_relative_error={max_relative_error:.3f}, "
            f"min_coverage={min_coverage:.2f}: "
            f"{len(detections)} object(s), {elapsed_ms:.1f} ms"
        )
        _print_detections(detections)

    def save(self):
        if self.result is None:
            self.status_var.set("Nothing to save yet. Click Apply settings first.")
            return

        if not cv2.imwrite(self.args.output, self.result):
            self.status_var.set(f"Could not write output image: {self.args.output}")
            raise RuntimeError(f"Could not write output image: {self.args.output}")

        self.status_var.set(f"Saved current applied result: {self.args.output}")
        print(f"Saved: {self.args.output}")

    def _pump_opencv(self):
        if self._closing:
            return

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.close()
            return
        if key == ord("s"):
            self.save()

        self.root.after(30, self._pump_opencv)

    def close(self):
        if self._closing:
            return
        self._closing = True
        cv2.destroyAllWindows()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image and detect up to two circles by finding the "
            "contiguous region of each contour that follows a circle best."
        )
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--threshold", type=int, default=128, help="Initial threshold, 0-255")
    parser.add_argument("--invert", action="store_true", help="Use inverted binary thresholding")
    parser.add_argument("--min-radius", type=float, default=10.0, help="Initial minimum circle radius")
    parser.add_argument(
        "--max-radius",
        type=float,
        default=0.0,
        help="Initial maximum circle radius; 0 uses the largest image dimension",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.08,
        help="Initial maximum mean radial error divided by radius",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.12,
        help="Initial minimum angular coverage of selected arc, 0..1",
    )
    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Apply a 3x3 morphological opening before contour detection",
    )
    parser.add_argument(
        "--max-contours",
        type=int,
        default=100,
        help="Maximum largest contours searched; 0 means unlimited",
    )
    parser.add_argument(
        "--max-search-points",
        type=int,
        default=500,
        help="Maximum ordered points searched per contour; 0 means unlimited",
    )
    parser.add_argument(
        "--palette-size",
        type=int,
        default=12,
        help="Maximum number of threshold color choices (default: 12)",
    )
    parser.add_argument(
        "--palette-min-gap",
        type=int,
        default=15,
        help="Minimum grayscale difference between color choices, 1-255 (default: 15)",
    )
    parser.add_argument(
        "--output",
        default="detected_circles.png",
        help="Output image written by Save current result or the S key",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be greater than zero")
    if args.max_radius < 0:
        parser.error("--max-radius must be zero or greater")
    if args.max_error <= 0:
        parser.error("--max-error must be greater than zero")
    if not 0.0 <= args.min_coverage <= 1.0:
        parser.error("--min-coverage must be between 0 and 1")
    if args.max_contours < 0 or args.max_search_points < 0:
        parser.error("--max-contours and --max-search-points must be zero or greater")
    if args.palette_size < 1:
        parser.error("--palette-size must be at least 1")
    if not 1 <= args.palette_min_gap <= 255:
        parser.error("--palette-min-gap must be between 1 and 255")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    palette = build_threshold_palette(
        image,
        gray,
        max_colors=args.palette_size,
        min_shade_gap=args.palette_min_gap,
    )

    root = tk.Tk()
    CircleArcDetectorApp(root, image, gray, palette, args)
    root.mainloop()


if __name__ == "__main__":
    main()
