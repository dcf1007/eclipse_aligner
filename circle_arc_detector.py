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

    r = math.sqrt(r2)
    err = float(np.mean(np.abs(np.hypot(x - cx, y - cy) - r)))
    return (float(cx), float(cy), r, err) if np.all(np.isfinite((cx, cy, r, err))) else None


def coverage(points, cx, cy):
    if len(points) < 2:
        return 0.0
    angles = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    return float(np.clip(np.ptp(angles) / (2 * np.pi), 0.0, 1.0))


def window_lengths(n, minimum):
    minimum = max(3, min(minimum, n))
    lengths = {minimum, n}
    length = minimum
    while length < n:
        lengths.add(length)
        length = min(n, max(length + 1, int(round(length * 1.45))))
    return sorted(lengths)


def eval_region(extended, n, start, length, min_radius, max_radius, max_error, min_coverage):
    if length < 3 or length > n:
        return None

    start %= n
    region = extended[start : start + length]
    fit = fit_circle(region)
    if fit is None:
        return None

    cx, cy, radius, error = fit
    if not min_radius <= radius <= max_radius:
        return None

    relative_error = error / radius
    if relative_error > max_error:
        return None

    cov = coverage(region, cx, cy)
    if cov < min_coverage:
        return None

    support = length / n
    score = cov**1.5 * math.sqrt(length) * (0.75 + 0.25 * math.sqrt(support)) / (relative_error + 0.002)
    return {
        "center": (cx, cy),
        "radius": radius,
        "relative_error": relative_error,
        "coverage": cov,
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


def refine(candidate, extended, n, minimum, min_radius, max_radius, max_error, min_coverage):
    best = candidate
    for _ in range(40):
        improved = False
        for ds, dl in ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0)):
            new_length = best["length"] + dl
            if not minimum <= new_length <= n:
                continue
            trial = eval_region(
                extended,
                n,
                best["start"] + ds,
                new_length,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1 + 1e-9):
                best, improved = trial, True
        if not improved:
            break
    return best


def find_regions(points, min_radius, max_radius, max_error, min_coverage, minimum=12):
    p = np.asarray(points, np.float64)
    n = len(p)
    if n < max(3, minimum):
        return []

    minimum = min(minimum, n)
    extended = np.vstack((p, p))
    coarse = []
    for length in window_lengths(n, minimum):
        step = max(1, length // 5)
        for start in range(0, n, step):
            candidate = eval_region(extended, n, start, length, min_radius, max_radius, max_error, min_coverage)
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
        candidate = refine(seed, extended, n, minimum, min_radius, max_radius, max_error, min_coverage)
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

    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
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
        text = f"{kind}  r={circle['radius']:.1f}  arc={circle['coverage'] * 360:.0f}deg  err={circle['relative_error']:.3f}"
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


def build_palette(gray, max_colors=20, min_gap=10):
    grayscale_values = gray.reshape(-1)
    hist = np.bincount(grayscale_values, minlength=256).astype(float)
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

    return [int(shade) for shade in sorted(shades)]


def gray_hex(value):
    value = int(np.clip(value, 0, 255))
    return f"#{value:02x}{value:02x}{value:02x}"


def text_color(gray_value):
    return "#111111" if gray_value >= 150 else "#f7f7f7"


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

        radius_limit = max(1600, round(self.args.max_radius * 1.5))
        rows = [
            ("Brightness threshold (0=black, 255=white)", self.threshold, 0, 255, 1, lambda value: str(int(value))),
            ("Minimum fitted circle radius (px)", self.min_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum fitted circle radius (px)", self.max_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum average radial error (% of radius)", self.max_error, 0.5, 50, 0.1, lambda value: f"{float(value):.1f}%"),
            ("Minimum visible circle arc (%)", self.min_coverage, 0, 100, 1, lambda value: f"{int(value)}% (~{int(value) * 3.6:.0f} deg)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)

        tk.Label(frame, text="Pick threshold from grayscale tones:").grid(row=5, column=0, sticky="nw")
        palette_frame = tk.Frame(frame)
        palette_frame.grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 8))
        for index, shade in enumerate(self.palette):
            tk.Button(
                palette_frame,
                text=str(shade),
                width=4,
                bg=gray_hex(shade),
                fg=text_color(shade),
                command=lambda value=shade: self.pick(value),
            ).grid(row=index // 10, column=index % 10, padx=2, pady=2)

        tk.Button(frame, text="Apply", width=12, command=self.apply).grid(row=6, column=0, sticky="w", pady=(2, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=1100).grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 0),
        )

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
        tk.Label(frame, text="Detected circles on grayscale image").grid(row=0, column=1, sticky="w", pady=(0, 4))

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
        elapsed = (time.perf_counter() - started) * 1000

        self.last_binary = binary
        self.last_result = result
        self.redraw()

        self.status.set(
            f"Applied: T={threshold}; radius={min_radius:.0f}-{max_radius:.0f}px; "
            f"error={max_error:.1%}; arc={min_coverage:.0%}; {len(circles)} circle(s); {elapsed:.1f} ms."
        )
        for circle in circles:
            print(
                f"{circle['class']}: center=({circle['center'][0]:.2f}, {circle['center'][1]:.2f}), "
                f"radius={circle['radius']:.2f}, arc={circle['coverage'] * 360:.1f} deg, "
                f"error={circle['relative_error']:.4f}, interior={circle['interior_fraction']:.1%}"
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
        fitted_size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))
        fitted = cv2.resize(
            image,
            fitted_size,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None

        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(canvas_width // 2 + 1, canvas_height // 2 + 1, image=photo, anchor="center")
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
    parser.add_argument("--palette-size", type=int, default=20)
    parser.add_argument("--palette-min-gap", type=int, default=10)
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

    palette = build_palette(gray, args.palette_size, args.palette_min_gap)
    root = tk.Tk()
    DetectorApp(root, image, gray, palette, args)
    root.mainloop()


if __name__ == "__main__":
    main()
