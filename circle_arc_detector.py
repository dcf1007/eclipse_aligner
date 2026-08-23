import argparse
import base64
import math
import time
import tkinter as tk

import cv2
import numpy as np

MIN_INTERIOR_FRACTION = 0.50
MAX_CLASS_CANDIDATES = 30
MAX_REGIONS_PER_CONTOUR = 4


def fit_circle(points):
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        return None

    x, y = p[:, 0], p[:, 1]
    xm, ym = float(x.mean()), float(y.mean())
    u, v = x - xm, y - ym
    z = u * u + v * v

    suu, svv, suv = np.dot(u, u), np.dot(v, v), np.dot(u, v)
    suz, svz = np.dot(u, z), np.dot(v, z)
    det = suu * svv - suv * suv
    if abs(det) <= 1e-12 * (suu * svv + 1.0):
        return None

    uc = 0.5 * (suz * svv - svz * suv) / det
    vc = 0.5 * (svz * suu - suz * suv) / det
    cx, cy = xm + uc, ym + vc

    r2 = float(z.mean()) + uc * uc + vc * vc
    if r2 <= 0 or not np.isfinite(r2):
        return None

    radius = math.sqrt(r2)
    mean_error = float(np.mean(np.abs(np.hypot(x - cx, y - cy) - radius)))
    values = (cx, cy, radius, mean_error)
    if not np.all(np.isfinite(values)):
        return None
    return float(cx), float(cy), radius, mean_error


def coverage(points, cx, cy):
    if len(points) < 2:
        return 0.0
    angles = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    return float(np.clip(np.ptp(angles) / (2 * np.pi), 0, 1))


def window_lengths(point_count, minimum):
    minimum = max(3, min(minimum, point_count))
    values = {minimum, point_count}
    length = minimum
    while length < point_count:
        values.add(length)
        length = min(point_count, max(length + 1, int(round(length * 1.45))))
    return sorted(values)


def eval_region(extended, point_count, start, length, min_radius, max_radius, max_error, min_coverage):
    if length < 3 or length > point_count:
        return None

    start %= point_count
    region = extended[start : start + length]
    fitted = fit_circle(region)
    if fitted is None:
        return None

    cx, cy, radius, mean_error = fitted
    if not min_radius <= radius <= max_radius:
        return None

    relative_error = mean_error / radius
    if relative_error > max_error:
        return None

    arc_coverage = coverage(region, cx, cy)
    if arc_coverage < min_coverage:
        return None

    support_fraction = length / point_count
    score = (
        arc_coverage**1.5
        * math.sqrt(length)
        * (0.75 + 0.25 * math.sqrt(support_fraction))
        / (relative_error + 0.002)
    )
    return {
        "center": (cx, cy),
        "radius": radius,
        "relative_error": relative_error,
        "coverage": arc_coverage,
        "points": region.copy(),
        "start": start,
        "length": length,
        "score": score,
    }


def same_circle(a, b, center_fraction=0.12, radius_fraction=0.12):
    ax, ay = a["center"]
    bx, by = b["center"]
    scale = max(a["radius"], b["radius"], 1.0)
    return (
        math.hypot(ax - bx, ay - by) < center_fraction * scale
        and abs(a["radius"] - b["radius"]) < radius_fraction * scale
    )


def refine(candidate, extended, point_count, minimum, min_radius, max_radius, max_error, min_coverage):
    best = candidate
    moves = ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0))

    for _ in range(40):
        improved = False
        for start_delta, length_delta in moves:
            length = best["length"] + length_delta
            if not minimum <= length <= point_count:
                continue

            trial = eval_region(
                extended,
                point_count,
                best["start"] + start_delta,
                length,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1 + 1e-9):
                best = trial
                improved = True

        if not improved:
            break

    return best


def find_regions(points, min_radius, max_radius, max_error, min_coverage, minimum=12):
    points = np.asarray(points, np.float64)
    point_count = len(points)
    if point_count < max(3, minimum):
        return []

    minimum = min(minimum, point_count)
    extended = np.vstack((points, points))
    coarse = []

    for length in window_lengths(point_count, minimum):
        for start in range(0, point_count, max(1, length // 5)):
            candidate = eval_region(
                extended,
                point_count,
                start,
                length,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if candidate is not None:
                coarse.append(candidate)

    coarse.sort(key=lambda item: item["score"], reverse=True)

    seeds = []
    for candidate in coarse:
        if any(same_circle(candidate, old, 0.10, 0.10) for old in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= max(8, MAX_REGIONS_PER_CONTOUR * 4):
            break

    result = []
    for seed in seeds:
        candidate = refine(
            seed,
            extended,
            point_count,
            minimum,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
        )
        if not any(same_circle(candidate, old) for old in result):
            result.append(candidate)

    result.sort(key=lambda item: item["score"], reverse=True)
    return result[:MAX_REGIONS_PER_CONTOUR]


def find_candidates(binary, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum = 12
    min_perimeter = max(12.0, min_radius * min_coverage * math.pi)

    usable = [(cv2.arcLength(contour, True), contour) for contour in contours if len(contour) >= minimum]
    usable = [(perimeter, contour) for perimeter, contour in usable if perimeter >= min_perimeter]
    usable.sort(key=lambda item: item[0], reverse=True)
    if max_contours > 0:
        usable = usable[:max_contours]

    candidates = []
    for _, contour in usable:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        if max_points > 0 and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
            points = points[indices]
        candidates.extend(find_regions(points, min_radius, max_radius, max_error, min_coverage, minimum))

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def interior_fraction(mask, candidate, exclude=None):
    height, width = mask.shape
    cx, cy = candidate["center"]
    radius = candidate["radius"]

    x0, y0 = max(0, int(cx - radius)), max(0, int(cy - radius))
    x1 = min(width, int(math.ceil(cx + radius)) + 1)
    y1 = min(height, int(math.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    valid = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius

    if exclude is not None:
        ex, ey = exclude["center"]
        er = exclude["radius"]
        valid &= (xx - ex) ** 2 + (yy - ey) ** 2 > er * er

    visible_pixels = int(np.count_nonzero(valid))
    if not visible_pixels:
        return 0.0, 0

    matching_pixels = int(np.count_nonzero(mask[y0:y1, x0:x1][valid]))
    return matching_pixels / visible_pixels, visible_pixels


def select_circle(candidates, mask, label, exclude=None):
    best = None
    best_score = -math.inf

    for candidate in candidates[:MAX_CLASS_CANDIDATES]:
        fraction, visible_pixels = interior_fraction(mask, candidate, exclude)
        if visible_pixels < 16 or fraction < MIN_INTERIOR_FRACTION:
            continue

        score = candidate["score"] * fraction**2
        if score > best_score:
            best = candidate.copy()
            best.update({"class": label, "interior_fraction": fraction})
            best_score = score

    return best


def detect(gray, threshold, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
    _, dark_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    _, bright_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    candidate_args = (min_radius, max_radius, max_error, min_coverage, max_contours, max_points)
    dark_circle = select_circle(find_candidates(dark_mask, *candidate_args), dark_mask, "below threshold")
    bright_circle = select_circle(
        find_candidates(bright_mask, *candidate_args),
        bright_mask,
        "above threshold",
        dark_circle,
    )

    return bright_mask, [circle for circle in (dark_circle, bright_circle) if circle is not None]


def process_image(image, gray, threshold, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
    binary, circles = detect(
        gray,
        threshold,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
    )

    output = image.copy()
    for circle in circles:
        cx, cy = circle["center"]
        center = (int(round(cx)), int(round(cy)))
        radius = int(round(circle["radius"]))

        support = np.rint(circle["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support) >= 2:
            cv2.polylines(output, [support], False, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.circle(output, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

        kind = "DARK <= T" if circle["class"] == "below threshold" else "BRIGHT > T"
        text = (
            f"{kind}  r={circle['radius']:.1f}  "
            f"arc={circle['coverage'] * 360:.0f}deg  "
            f"err={circle['relative_error']:.3f}"
        )
        cv2.putText(
            output,
            text,
            (center[0] + 10, max(18, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return binary, output, circles


def build_palette(image, gray, max_colors=12, min_gap=15):
    stride = max(1, int(math.ceil(math.sqrt(gray.size / 200_000))))
    sample_gray = gray[::stride, ::stride].reshape(-1)
    sample_bgr = image[::stride, ::stride].reshape(-1, 3)

    hist = np.bincount(sample_gray, minlength=256).astype(float)
    smooth_radius = max(2, min_gap // 3)
    density = np.convolve(hist, np.ones(2 * smooth_radius + 1), mode="same")

    shades = []
    for shade in np.argsort(density)[::-1]:
        shade = int(shade)
        if density[shade] <= 0:
            break
        if any(abs(shade - old) < min_gap for old in shades):
            continue
        shades.append(shade)
        if len(shades) >= max_colors:
            break

    result = []
    sample_gray_i16 = sample_gray.astype(np.int16, copy=False)
    color_radius = max(3, min_gap // 3)
    for shade in shades:
        mask = np.abs(sample_gray_i16 - shade) <= color_radius
        if np.any(mask):
            threshold = int(round(float(np.average(sample_gray[mask]))))
            bgr = tuple(int(round(value)) for value in np.median(sample_bgr[mask], axis=0))
        else:
            threshold = shade
            bgr = (shade,) * 3
        result.append({"threshold": int(np.clip(threshold, 0, 255)), "bgr": bgr})

    result.sort(key=lambda item: item["threshold"])
    return result


def bgr_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def text_color(bgr):
    b, g, r = bgr
    return "#111111" if 0.114 * b + 0.587 * g + 0.299 * r >= 150 else "#f7f7f7"


class DetectorApp:
    def __init__(self, root, image, gray, palette, args):
        self.root = root
        self.image = image
        self.gray = gray
        self.palette = palette
        self.args = args

        self.last_binary = None
        self.last_result = None
        self.binary_photo = None
        self.result_photo = None
        self.resize_job = None

        self.threshold = tk.IntVar(value=args.threshold)
        self.min_radius = tk.IntVar(value=round(args.min_radius))
        self.max_radius = tk.IntVar(value=round(args.max_radius))
        self.max_error = tk.DoubleVar(value=args.max_error * 100)
        self.min_coverage = tk.IntVar(value=round(args.min_coverage * 100))
        self.status = tk.StringVar(value="Adjust settings, then click Apply.")

        root.title("Circle / Arc Detector")
        root.minsize(980, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.build_controls()
        self.build_previews()

        root.bind("<Return>", lambda _event: self.apply())
        root.bind("<Escape>", lambda _event: root.destroy())

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        radius_limit = max(100, round(self.args.max_radius * 2))
        rows = [
            ("Brightness threshold (0=black, 255=white)", self.threshold, 0, 255, 1, lambda value: str(int(value))),
            ("Minimum fitted circle radius (px)", self.min_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum fitted circle radius (px)", self.max_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum average radial error (% of radius)", self.max_error, 0.5, 50, 0.1, lambda value: f"{float(value):.1f}%"),
            ("Minimum visible circle arc (%)", self.min_coverage, 0, 100, 1, lambda value: f"{int(value)}% (~{int(value) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)

        tk.Label(frame, text="Pick threshold from image colors:").grid(row=5, column=0, sticky="nw")
        palette_frame = tk.Frame(frame)
        palette_frame.grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 8))
        for index, item in enumerate(self.palette):
            tk.Button(
                palette_frame,
                text=str(item["threshold"]),
                width=4,
                bg=bgr_hex(item["bgr"]),
                fg=text_color(item["bgr"]),
                command=lambda value=item["threshold"]: self.pick(value),
            ).grid(row=index // 8, column=index % 8, padx=2, pady=2)

        tk.Button(frame, text="Apply", width=12, command=self.apply).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(2, 0),
        )

        tk.Label(
            frame,
            textvariable=self.status,
            anchor="w",
            justify="left",
            wraplength=1100,
        ).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=38, anchor="w").grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
        )
        tk.Scale(
            parent,
            from_=low,
            to=high,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            showvalue=False,
            length=420,
            highlightthickness=0,
        ).grid(row=row, column=1, sticky="ew", pady=2)

        value = tk.Label(parent, width=18, anchor="e")
        value.grid(row=row, column=2, pady=2)

        def update(*_args):
            value.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update)
        update()

    def build_previews(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(frame, text="Threshold preview").grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(frame, text="Detected circles").grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.binary_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.result_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.binary_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.result_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        self.binary_canvas.bind("<Configure>", self.schedule_redraw)
        self.result_canvas.bind("<Configure>", self.schedule_redraw)
        self.placeholder(self.binary_canvas, "Threshold preview")
        self.placeholder(self.result_canvas, "Detected circles")

    def pick(self, value):
        self.threshold.set(value)
        self.status.set(f"Threshold set to {value}. Click Apply to recompute.")

    def pending(self, *_args):
        if self.last_result is not None:
            self.status.set("Settings changed. Click Apply to recompute the result.")

    def settings(self):
        min_radius = max(1.0, float(self.min_radius.get()))
        max_radius = max(1.0, float(self.max_radius.get()))
        if max_radius < min_radius:
            max_radius = min_radius
            self.max_radius.set(round(max_radius))

        return (
            int(self.threshold.get()),
            min_radius,
            max_radius,
            max(0.001, self.max_error.get() / 100),
            float(np.clip(self.min_coverage.get() / 100, 0, 1)),
        )

    def apply(self):
        threshold, min_radius, max_radius, max_error, min_coverage = self.settings()
        started = time.perf_counter()
        binary, result, circles = process_image(
            self.image,
            self.gray,
            threshold,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
            self.args.max_contours,
            self.args.max_search_points,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.last_binary = binary
        self.last_result = result
        self.redraw()
        self.status.set(
            f"Applied: T={threshold}; radius={min_radius:.0f}-{max_radius:.0f}px; "
            f"error={max_error:.1%}; arc={min_coverage:.0%}; "
            f"{len(circles)} circle(s); {elapsed_ms:.1f} ms."
        )

        for circle in circles:
            print(
                f"{circle['class']}: "
                f"center=({circle['center'][0]:.2f}, {circle['center'][1]:.2f}), "
                f"radius={circle['radius']:.2f}, "
                f"arc={circle['coverage'] * 360:.1f}°, "
                f"error={circle['relative_error']:.4f}, "
                f"interior={circle['interior_fraction']:.1%}"
            )

    def schedule_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        self.resize_job = None
        if self.last_binary is not None:
            self.binary_photo = self.show_image(self.binary_canvas, self.last_binary)
        if self.last_result is not None:
            self.result_photo = self.show_image(self.result_canvas, self.last_result)

    @staticmethod
    def show_image(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]

        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        target_size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        fitted = cv2.resize(
            image,
            target_size,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None

        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2 + 1,
            canvas_height // 2 + 1,
            image=photo,
            anchor="center",
        )
        return photo

    @staticmethod
    def placeholder(canvas, text):
        canvas.create_text(
            160,
            120,
            text=text + "\nClick Apply",
            fill="#cccccc",
            justify="center",
        )


def main():
    parser = argparse.ArgumentParser(description="Detect up to two inferred circles from thresholded image arcs.")
    parser.add_argument("image")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--min-radius", type=float, default=1000.0)
    parser.add_argument("--max-radius", type=float, default=1500.0, help="0 = largest image dimension")
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.12)
    parser.add_argument("--max-contours", type=int, default=100)
    parser.add_argument("--max-search-points", type=int, default=500)
    parser.add_argument("--palette-size", type=int, default=12)
    parser.add_argument("--palette-min-gap", type=int, default=15)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be 0..255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be > 0")
    if args.max_radius < 0:
        parser.error("--max-radius must be >= 0")
    if args.max_error <= 0:
        parser.error("--max-error must be > 0")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be 0..1")
    if args.max_contours < 0 or args.max_search_points < 0:
        parser.error("search limits must be >= 0")
    if args.palette_size < 1 or not 1 <= args.palette_min_gap <= 255:
        parser.error("invalid palette settings")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if args.max_radius == 0:
        args.max_radius = float(max(gray.shape))
    args.max_radius = max(args.max_radius, args.min_radius)

    palette = build_palette(image, gray, args.palette_size, args.palette_min_gap)
    root = tk.Tk()
    DetectorApp(root, image, gray, palette, args)
    root.mainloop()


if __name__ == "__main__":
    main()
