import argparse
import base64
import math
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)
MIN_INTERIOR_FRACTION = 0.50
MAX_CLASS_CANDIDATES = 30
MAX_REGIONS_PER_CONTOUR = 4


def _fit_circle(points):
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

    radius_sq = float(z.mean()) + uc * uc + vc * vc
    if radius_sq <= 0.0 or not np.isfinite(radius_sq):
        return None

    radius = math.sqrt(radius_sq)
    mean_error = float(np.mean(np.abs(np.hypot(x - cx, y - cy) - radius)))
    if not all(np.isfinite(value) for value in (cx, cy, radius, mean_error)):
        return None

    return float(cx), float(cy), float(radius), mean_error


def _angular_coverage(points, cx, cy):
    if len(points) < 2:
        return 0.0
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    return float(np.clip(np.ptp(np.unwrap(angles)) / (2.0 * np.pi), 0.0, 1.0))


def _candidate_window_lengths(point_count, min_region_points):
    min_region_points = max(3, min(min_region_points, point_count))
    lengths = {min_region_points, point_count}
    length = min_region_points
    while length < point_count:
        lengths.add(length)
        length = min(point_count, max(length + 1, int(round(length * 1.45))))
    return sorted(lengths)


def _evaluate_region(
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

    start = start % contour_point_count
    region = extended_points[start : start + length]
    fitted = _fit_circle(region)
    if fitted is None:
        return None

    cx, cy, radius, mean_error = fitted
    if radius < min_radius or radius > max_radius:
        return None

    relative_error = mean_error / radius
    if relative_error > max_relative_error:
        return None

    coverage = _angular_coverage(region, cx, cy)
    if coverage < min_coverage:
        return None

    support_fraction = length / contour_point_count
    score = (
        coverage**1.5
        * math.sqrt(float(length))
        * (0.75 + 0.25 * math.sqrt(support_fraction))
        / (relative_error + 0.002)
    )

    return {
        "center": (cx, cy),
        "radius": radius,
        "relative_error": relative_error,
        "coverage": coverage,
        "points": region.copy(),
        "region_start": start,
        "region_length": length,
        "score": score,
    }


def _refine_region(
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
        start = best["region_start"]
        length = best["region_length"]
        for start_delta, length_delta in moves:
            new_length = length + length_delta
            if not min_region_points <= new_length <= contour_point_count:
                continue
            trial = _evaluate_region(
                extended_points,
                contour_point_count,
                start + start_delta,
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


def _same_circle(a, b, center_fraction=0.12, radius_fraction=0.12):
    ax, ay = a["center"]
    bx, by = b["center"]
    ar = a["radius"]
    br = b["radius"]
    scale = max(ar, br, 1.0)
    return (
        math.hypot(ax - bx, ay - by) < center_fraction * scale
        and abs(ar - br) < radius_fraction * scale
    )


def _find_regions(points, min_radius, max_radius, max_relative_error, min_coverage, min_region_points=12):
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    if point_count < max(3, min_region_points):
        return []

    min_region_points = min(min_region_points, point_count)
    extended = np.vstack((points, points))
    coarse = []

    for length in _candidate_window_lengths(point_count, min_region_points):
        step = max(1, length // 5)
        for start in range(0, point_count, step):
            candidate = _evaluate_region(
                extended,
                point_count,
                start,
                length,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
            )
            if candidate is not None:
                coarse.append(candidate)

    if not coarse:
        return []

    coarse.sort(key=lambda item: item["score"], reverse=True)
    seeds = []
    for candidate in coarse:
        if any(_same_circle(candidate, existing, 0.10, 0.10) for existing in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= max(8, MAX_REGIONS_PER_CONTOUR * 4):
            break

    refined = []
    for seed in seeds:
        candidate = _refine_region(
            seed,
            extended,
            point_count,
            min_region_points,
            min_radius,
            max_radius,
            max_relative_error,
            min_coverage,
        )
        if not any(_same_circle(candidate, existing) for existing in refined):
            refined.append(candidate)

    refined.sort(key=lambda item: item["score"], reverse=True)
    return refined[:MAX_REGIONS_PER_CONTOUR]


def _downsample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices]


def _find_candidates(
    binary,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    max_contours,
    max_search_points,
    min_contour_points=12,
):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_region_points = max(8, min_contour_points)
    min_perimeter = max(12.0, min_radius * min_coverage * math.pi)

    usable = []
    for contour in contours:
        if len(contour) < min_region_points:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter >= min_perimeter:
            usable.append((perimeter, contour))

    usable.sort(key=lambda item: item[0], reverse=True)
    if max_contours > 0:
        usable = usable[:max_contours]

    candidates = []
    for _, contour in usable:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        points = _downsample_points(points, max_search_points)
        candidates.extend(
            _find_regions(
                points,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
                min_region_points,
            )
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _interior_match_fraction(class_mask, candidate, excluded_circle=None):
    height, width = class_mask.shape
    cx, cy = candidate["center"]
    radius = candidate["radius"]

    x0 = max(0, int(math.floor(cx - radius)))
    y0 = max(0, int(math.floor(cy - radius)))
    x1 = min(width, int(math.ceil(cx + radius)) + 1)
    y1 = min(height, int(math.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    valid = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius

    if excluded_circle is not None:
        ex, ey = excluded_circle["center"]
        er = excluded_circle["radius"]
        valid &= (xx - ex) ** 2 + (yy - ey) ** 2 > er * er

    visible_pixels = int(np.count_nonzero(valid))
    if visible_pixels == 0:
        return 0.0, 0

    roi = class_mask[y0:y1, x0:x1]
    matching_pixels = int(np.count_nonzero(roi[valid]))
    return matching_pixels / visible_pixels, visible_pixels


def _select_class_circle(candidates, class_mask, label, excluded_circle=None):
    best = None
    best_score = -math.inf

    for candidate in candidates[:MAX_CLASS_CANDIDATES]:
        match_fraction, visible_pixels = _interior_match_fraction(class_mask, candidate, excluded_circle)
        if visible_pixels < 16 or match_fraction < MIN_INTERIOR_FRACTION:
            continue

        score = candidate["score"] * match_fraction**2
        if score > best_score:
            best = candidate.copy()
            best["class"] = label
            best["interior_fraction"] = match_fraction
            best_score = score

    return best


def detect_dark_bright_circles(
    gray,
    threshold_value,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    morphology=False,
    max_contours=100,
    max_search_points=500,
):
    threshold_value = int(threshold_value)
    _, dark_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    dark_detection_mask = dark_mask
    bright_detection_mask = bright_mask
    if morphology:
        dark_detection_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        bright_detection_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, MORPH_KERNEL)

    common = dict(
        min_radius=min_radius,
        max_radius=max_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
        max_contours=max_contours,
        max_search_points=max_search_points,
    )

    dark_candidates = _find_candidates(dark_detection_mask, **common)
    bright_candidates = _find_candidates(bright_detection_mask, **common)

    dark_circle = _select_class_circle(dark_candidates, dark_mask, label="below threshold")
    bright_circle = _select_class_circle(
        bright_candidates,
        bright_mask,
        label="above threshold",
        excluded_circle=dark_circle,
    )

    detections = []
    if dark_circle is not None:
        detections.append(dark_circle)
    if bright_circle is not None:
        detections.append(bright_circle)

    return bright_mask, detections


def process_image(
    image,
    threshold_value,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    morphology,
    max_contours,
    max_search_points,
    gray=None,
):
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_preview, detections = detect_dark_bright_circles(
        gray,
        threshold_value,
        min_radius=min_radius,
        max_radius=max_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
        morphology=morphology,
        max_contours=max_contours,
        max_search_points=max_search_points,
    )

    output = image.copy()
    for detection in detections:
        cx, cy = detection["center"]
        radius = detection["radius"]
        center = (int(round(cx)), int(round(cy)))
        radius_int = int(round(radius))

        support_points = np.rint(detection["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support_points) >= 2:
            cv2.polylines(output, [support_points], False, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.circle(output, center, radius_int, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

        short_class = "DARK <= T" if detection["class"] == "below threshold" else "BRIGHT > T"
        coverage_degrees = detection["coverage"] * 360.0
        label = f"{short_class}  r={radius:.1f}  arc={coverage_degrees:.0f}deg  err={detection['relative_error']:.3f}"
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

    return threshold_preview, output, detections


def build_threshold_palette(image, gray, max_colors=12, min_shade_gap=15):
    if max_colors <= 0:
        return []

    pixel_count = gray.size
    sample_limit = 200_000
    stride = max(1, int(math.ceil(math.sqrt(pixel_count / sample_limit))))
    sample_gray = gray[::stride, ::stride].reshape(-1)
    sample_bgr = image[::stride, ::stride].reshape(-1, 3)

    hist = np.bincount(sample_gray, minlength=256).astype(np.float64)
    smooth_radius = max(2, min_shade_gap // 3)
    density = np.convolve(hist, np.ones(2 * smooth_radius + 1, dtype=np.float64), mode="same")

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
            threshold = int(round(float(np.average(sample_gray[mask]))))
            bgr = tuple(int(round(value)) for value in np.median(sample_bgr[mask], axis=0))
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


def _print_detections(detections):
    if not detections:
        print("No accepted circles.")
        return
    for detection in detections:
        cx, cy = detection["center"]
        print(
            f"{detection['class']}: center=({cx:.2f}, {cy:.2f}), "
            f"radius={detection['radius']:.2f}, coverage={detection['coverage'] * 360.0:.1f} deg, "
            f"relative_error={detection['relative_error']:.4f}, interior_match={detection['interior_fraction']:.1%}"
        )


class DetectorApp:
    def __init__(self, root, image, gray, palette, args):
        self.root = root
        self.image = image
        self.gray = gray
        self.palette = palette
        self.args = args

        self.current_output_path = args.output
        self.last_threshold_preview = None
        self.last_result_image = None
        self.last_detections = []
        self.threshold_photo = None
        self.result_photo = None
        self.resize_job = None

        self.threshold_var = tk.IntVar(value=args.threshold)
        self.min_radius_var = tk.IntVar(value=int(round(args.min_radius)))
        self.max_radius_var = tk.IntVar(value=int(round(args.max_radius)))
        self.max_error_var = tk.DoubleVar(value=args.max_error * 100.0)
        self.min_coverage_var = tk.IntVar(value=int(round(args.min_coverage * 100.0)))
        self.morphology_var = tk.BooleanVar(value=args.morphology)
        self.status_var = tk.StringVar(value="Adjust settings, then click Apply.")
        self.save_path_var = tk.StringVar(value=self.current_output_path)

        self.root.title("Circle / Arc Detector")
        self.root.minsize(980, 680)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_controls()
        self._build_previews()
        self._draw_placeholder(self.threshold_canvas, "Threshold preview")
        self._draw_placeholder(self.result_canvas, "Detected circles")

        self.root.bind("<Return>", lambda _event: self.apply())
        self.root.bind("<Control-s>", lambda _event: self.save_result())
        self.root.bind("<Escape>", lambda _event: self.on_close())

    def _build_controls(self):
        controls = tk.Frame(self.root, padx=10, pady=8)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        max_slider_radius = max(100, int(round(self.args.max_radius * 2)))
        rows = [
            ("Brightness threshold (0=black, 255=white)", self.threshold_var, 0, 255, 1, lambda v: f"{int(v)}"),
            ("Minimum fitted circle radius (px)", self.min_radius_var, 1, max_slider_radius, 1, lambda v: f"{int(v)} px"),
            ("Maximum fitted circle radius (px)", self.max_radius_var, 1, max_slider_radius, 1, lambda v: f"{int(v)} px"),
            ("Maximum average radial error (% of radius)", self.max_error_var, 0.5, 50.0, 0.1, lambda v: f"{float(v):.1f}%"),
            ("Minimum visible circle arc (%)", self.min_coverage_var, 0, 100, 1, lambda v: f"{int(v)}% (~{int(v) * 3.6:.0f}°)"),
        ]
        for row, values in enumerate(rows):
            self._add_scale_row(controls, row, *values)

        morphology = tk.Checkbutton(
            controls,
            text="Use 3x3 morphology cleanup before contour detection",
            variable=self.morphology_var,
            anchor="w",
        )
        morphology.grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 8))
        self.morphology_var.trace_add("write", self._mark_pending)

        tk.Label(controls, text="Pick threshold from image colors:").grid(row=6, column=0, sticky="nw")
        palette_frame = tk.Frame(controls)
        palette_frame.grid(row=6, column=1, columnspan=2, sticky="w", pady=(0, 8))
        self._build_palette(palette_frame)

        buttons = tk.Frame(controls)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(4, weight=1)
        tk.Button(buttons, text="Apply", width=12, command=self.apply).grid(row=0, column=0, padx=(0, 8))
        tk.Button(buttons, text="Save", width=12, command=self.save_result).grid(row=0, column=1, padx=(0, 8))
        tk.Button(buttons, text="Save As...", width=12, command=self.choose_save_path).grid(row=0, column=2, padx=(0, 8))
        tk.Label(buttons, text="Output file:").grid(row=0, column=3, sticky="e")
        tk.Entry(buttons, textvariable=self.save_path_var).grid(row=0, column=4, sticky="ew", padx=(6, 0))

        tk.Label(controls, textvariable=self.status_var, anchor="w", justify="left", wraplength=1100).grid(
            row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0)
        )

    def _build_previews(self):
        previews = tk.Frame(self.root, padx=10, pady=(0, 10))
        previews.grid(row=1, column=0, sticky="nsew")
        previews.rowconfigure(1, weight=1)
        previews.columnconfigure(0, weight=1, uniform="preview")
        previews.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(previews, text="Threshold preview").grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(previews, text="Detected circles").grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.threshold_canvas = tk.Canvas(previews, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.result_canvas = tk.Canvas(previews, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.threshold_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.result_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        self.threshold_canvas.bind("<Configure>", self._schedule_preview_redraw)
        self.result_canvas.bind("<Configure>", self._schedule_preview_redraw)

    def _add_scale_row(self, parent, row, label_text, variable, minimum, maximum, resolution, formatter):
        tk.Label(parent, text=label_text, width=38, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        scale = tk.Scale(
            parent,
            from_=minimum,
            to=maximum,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            showvalue=False,
            length=420,
            highlightthickness=0,
        )
        scale.grid(row=row, column=1, sticky="ew", pady=2)
        value_label = tk.Label(parent, anchor="e", width=18)
        value_label.grid(row=row, column=2, sticky="e", pady=2)

        def refresh_value(*_args):
            value_label.config(text=formatter(variable.get()))
            self._mark_pending()

        variable.trace_add("write", refresh_value)
        refresh_value()

    def _build_palette(self, parent):
        for index, item in enumerate(self.palette):
            tk.Button(
                parent,
                text=str(item["threshold"]),
                width=4,
                background=_bgr_to_hex(item["bgr"]),
                foreground=_contrasting_text_color(item["bgr"]),
                command=lambda value=item["threshold"]: self._set_threshold_from_palette(value),
            ).grid(row=index // 8, column=index % 8, padx=2, pady=2)

    def _set_threshold_from_palette(self, threshold):
        self.threshold_var.set(int(threshold))
        self.status_var.set(f"Pending threshold set to {threshold} from the color palette. Click Apply to recompute.")

    def _mark_pending(self, *_args):
        if self.last_result_image is not None:
            self.status_var.set("Settings changed. Click Apply to recompute the result.")

    def _current_settings(self):
        threshold = int(self.threshold_var.get())
        min_radius = max(1.0, float(self.min_radius_var.get()))
        max_radius = max(1.0, float(self.max_radius_var.get()))
        if max_radius < min_radius:
            max_radius = min_radius
            self.max_radius_var.set(int(round(max_radius)))
        max_relative_error = max(0.001, float(self.max_error_var.get()) / 100.0)
        min_coverage = float(np.clip(self.min_coverage_var.get() / 100.0, 0.0, 1.0))
        return threshold, min_radius, max_radius, max_relative_error, min_coverage

    def apply(self):
        threshold, min_radius, max_radius, max_relative_error, min_coverage = self._current_settings()
        started = time.perf_counter()
        threshold_preview, result_image, detections = process_image(
            self.image,
            threshold,
            min_radius=min_radius,
            max_radius=max_radius,
            max_relative_error=max_relative_error,
            min_coverage=min_coverage,
            morphology=self.morphology_var.get(),
            max_contours=self.args.max_contours,
            max_search_points=self.args.max_search_points,
            gray=self.gray,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self.last_threshold_preview = threshold_preview
        self.last_result_image = result_image
        self.last_detections = detections
        self._redraw_previews()

        self.status_var.set(
            f"Applied: threshold={threshold}; radius={min_radius:.0f}-{max_radius:.0f}px; "
            f"max avg radial error={max_relative_error:.1%}; minimum visible arc={min_coverage:.0%}; "
            f"morphology={'on' if self.morphology_var.get() else 'off'}; {len(detections)} circle(s); {elapsed_ms:.1f} ms."
        )
        print(
            f"Applied: threshold={threshold}, min_radius={min_radius:.0f}, max_radius={max_radius:.0f}, "
            f"max_relative_error={max_relative_error:.3f}, min_coverage={min_coverage:.2f}, "
            f"morphology={self.morphology_var.get()}: {len(detections)} circle(s), {elapsed_ms:.1f} ms"
        )
        _print_detections(detections)

    def _schedule_preview_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self._redraw_previews)

    def _redraw_previews(self):
        self.resize_job = None
        if self.last_threshold_preview is not None:
            self.threshold_photo = self._show_image(self.threshold_canvas, self.last_threshold_preview)
        if self.last_result_image is not None:
            self.result_photo = self._show_image(self.result_canvas, self.last_result_image)

    def _show_image(self, canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = min(canvas_width / image_width, canvas_height / image_height)
        scale = max(scale, 1e-6)
        target_width = max(1, int(round(image_width * scale)))
        target_height = max(1, int(round(image_height * scale)))

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        fitted = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None

        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(canvas_width // 2 + 1, canvas_height // 2 + 1, image=photo, anchor="center")
        return photo

    def _draw_placeholder(self, canvas, text):
        canvas.delete("all")
        canvas.create_text(
            max(1, canvas.winfo_width()) // 2,
            max(1, canvas.winfo_height()) // 2,
            text=text + "\nClick Apply",
            fill="#cccccc",
            justify="center",
        )

    def choose_save_path(self):
        filename = filedialog.asksaveasfilename(
            title="Save result",
            defaultextension=".png",
            initialfile=self.current_output_path,
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not filename:
            return
        self.current_output_path = filename
        self.save_path_var.set(filename)
        self.status_var.set(f"Output path set to: {filename}")

    def save_result(self):
        if self.last_result_image is None:
            self.status_var.set("Nothing to save yet. Click Apply first.")
            return

        output_path = self.save_path_var.get().strip() or self.current_output_path
        if not output_path:
            self.choose_save_path()
            output_path = self.save_path_var.get().strip()
            if not output_path:
                return

        if not cv2.imwrite(output_path, self.last_result_image):
            messagebox.showerror("Save failed", f"Could not write output image:\n{output_path}")
            return

        self.current_output_path = output_path
        self.save_path_var.set(output_path)
        self.status_var.set(f"Saved current result to: {output_path}")
        print(f"Saved: {output_path}")

    def on_close(self):
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image and detect at most two inferred circles: one with an interior at/below "
            "the threshold and one above it."
        )
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--threshold", type=int, default=128, help="Initial threshold, 0-255")
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
    parser.add_argument("--morphology", action="store_true", help="Start with a 3x3 morphological opening enabled")
    parser.add_argument(
        "--max-contours",
        type=int,
        default=100,
        help="Maximum largest contours searched per threshold class; 0 means unlimited",
    )
    parser.add_argument(
        "--max-search-points",
        type=int,
        default=500,
        help="Maximum ordered points searched per contour; 0 means unlimited",
    )
    parser.add_argument("--palette-size", type=int, default=12, help="Maximum number of threshold color choices")
    parser.add_argument(
        "--palette-min-gap",
        type=int,
        default=15,
        help="Minimum grayscale difference between palette entries, 1-255",
    )
    parser.add_argument("--output", default="detected_circles.png", help="Default output image path for Save")
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
    image_max_dim = max(gray.shape)
    if args.max_radius == 0:
        args.max_radius = float(image_max_dim)
    args.max_radius = max(args.max_radius, args.min_radius)

    palette = build_threshold_palette(image, gray, max_colors=args.palette_size, min_shade_gap=args.palette_min_gap)

    root = tk.Tk()
    DetectorApp(root, image, gray, palette, args)
    root.mainloop()


if __name__ == "__main__":
    main()
