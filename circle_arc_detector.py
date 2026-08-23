import argparse
import base64
import math
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np

MORPH_KERNEL = np.ones((3, 3), np.uint8)
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
    a = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    return float(np.clip(np.ptp(a) / (2 * np.pi), 0, 1))


def window_lengths(n, minimum):
    minimum = max(3, min(minimum, n))
    values = {minimum, n}
    length = minimum
    while length < n:
        values.add(length)
        length = min(n, max(length + 1, int(round(length * 1.45))))
    return sorted(values)


def eval_region(extended, n, start, length, min_r, max_r, max_err, min_cov):
    if length < 3 or length > n:
        return None
    start %= n
    region = extended[start : start + length]
    fit = fit_circle(region)
    if fit is None:
        return None
    cx, cy, r, err = fit
    if not min_r <= r <= max_r:
        return None
    rel_err = err / r
    if rel_err > max_err:
        return None
    cov = coverage(region, cx, cy)
    if cov < min_cov:
        return None
    support = length / n
    score = cov**1.5 * math.sqrt(length) * (0.75 + 0.25 * math.sqrt(support)) / (rel_err + 0.002)
    return {
        "center": (cx, cy), "radius": r, "relative_error": rel_err,
        "coverage": cov, "points": region.copy(), "start": start,
        "length": length, "score": score,
    }


def same_circle(a, b, center_frac=0.12, radius_frac=0.12):
    ax, ay = a["center"]
    bx, by = b["center"]
    scale = max(a["radius"], b["radius"], 1.0)
    return math.hypot(ax - bx, ay - by) < center_frac * scale and abs(a["radius"] - b["radius"]) < radius_frac * scale


def refine(candidate, extended, n, minimum, min_r, max_r, max_err, min_cov):
    best = candidate
    for _ in range(40):
        improved = False
        for ds, dl in ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0)):
            length = best["length"] + dl
            if not minimum <= length <= n:
                continue
            trial = eval_region(extended, n, best["start"] + ds, length, min_r, max_r, max_err, min_cov)
            if trial is not None and trial["score"] > best["score"] * (1 + 1e-9):
                best, improved = trial, True
        if not improved:
            break
    return best


def find_regions(points, min_r, max_r, max_err, min_cov, minimum=12):
    p = np.asarray(points, np.float64)
    n = len(p)
    if n < max(3, minimum):
        return []
    minimum = min(minimum, n)
    extended = np.vstack((p, p))
    coarse = []
    for length in window_lengths(n, minimum):
        for start in range(0, n, max(1, length // 5)):
            item = eval_region(extended, n, start, length, min_r, max_r, max_err, min_cov)
            if item is not None:
                coarse.append(item)
    coarse.sort(key=lambda x: x["score"], reverse=True)
    seeds = []
    for item in coarse:
        if any(same_circle(item, old, 0.10, 0.10) for old in seeds):
            continue
        seeds.append(item)
        if len(seeds) >= max(8, MAX_REGIONS_PER_CONTOUR * 4):
            break
    result = []
    for seed in seeds:
        item = refine(seed, extended, n, minimum, min_r, max_r, max_err, min_cov)
        if not any(same_circle(item, old) for old in result):
            result.append(item)
    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:MAX_REGIONS_PER_CONTOUR]


def find_candidates(binary, min_r, max_r, max_err, min_cov, max_contours, max_points):
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum = 12
    min_perimeter = max(12.0, min_r * min_cov * math.pi)
    usable = [(cv2.arcLength(c, True), c) for c in contours if len(c) >= minimum]
    usable = [(perim, c) for perim, c in usable if perim >= min_perimeter]
    usable.sort(key=lambda x: x[0], reverse=True)
    if max_contours > 0:
        usable = usable[:max_contours]
    out = []
    for _, contour in usable:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        if max_points > 0 and len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points, dtype=np.int32)]
        out.extend(find_regions(points, min_r, max_r, max_err, min_cov, minimum))
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def interior_fraction(mask, candidate, exclude=None):
    h, w = mask.shape
    cx, cy = candidate["center"]
    r = candidate["radius"]
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(w, int(math.ceil(cx + r)) + 1), min(h, int(math.ceil(cy + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    valid = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if exclude is not None:
        ex, ey = exclude["center"]
        er = exclude["radius"]
        valid &= (xx - ex) ** 2 + (yy - ey) ** 2 > er * er
    count = int(np.count_nonzero(valid))
    if not count:
        return 0.0, 0
    return int(np.count_nonzero(mask[y0:y1, x0:x1][valid])) / count, count


def select_circle(candidates, mask, label, exclude=None):
    best = None
    best_score = -math.inf
    for item in candidates[:MAX_CLASS_CANDIDATES]:
        fraction, visible = interior_fraction(mask, item, exclude)
        if visible < 16 or fraction < MIN_INTERIOR_FRACTION:
            continue
        score = item["score"] * fraction**2
        if score > best_score:
            best = item.copy()
            best.update({"class": label, "interior_fraction": fraction})
            best_score = score
    return best


def detect(gray, threshold, min_r, max_r, max_err, min_cov, morphology, max_contours, max_points):
    _, dark = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    _, bright = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    dark_search, bright_search = dark, bright
    if morphology:
        dark_search = cv2.morphologyEx(dark, cv2.MORPH_OPEN, MORPH_KERNEL)
        bright_search = cv2.morphologyEx(bright, cv2.MORPH_OPEN, MORPH_KERNEL)
    args = (min_r, max_r, max_err, min_cov, max_contours, max_points)
    dark_circle = select_circle(find_candidates(dark_search, *args), dark, "below threshold")
    bright_circle = select_circle(find_candidates(bright_search, *args), bright, "above threshold", dark_circle)
    return bright, [c for c in (dark_circle, bright_circle) if c is not None]


def process_image(image, gray, threshold, min_r, max_r, max_err, min_cov, morphology, max_contours, max_points):
    binary, circles = detect(gray, threshold, min_r, max_r, max_err, min_cov, morphology, max_contours, max_points)
    output = image.copy()
    for c in circles:
        cx, cy = c["center"]
        center = (int(round(cx)), int(round(cy)))
        radius = int(round(c["radius"]))
        support = np.rint(c["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support) >= 2:
            cv2.polylines(output, [support], False, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(output, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)
        kind = "DARK <= T" if c["class"] == "below threshold" else "BRIGHT > T"
        text = f"{kind}  r={c['radius']:.1f}  arc={c['coverage']*360:.0f}deg  err={c['relative_error']:.3f}"
        cv2.putText(output, text, (center[0] + 10, max(18, center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return binary, output, circles


def build_palette(image, gray, max_colors=12, min_gap=15):
    stride = max(1, int(math.ceil(math.sqrt(gray.size / 200_000))))
    gs = gray[::stride, ::stride].reshape(-1)
    bgrs = image[::stride, ::stride].reshape(-1, 3)
    hist = np.bincount(gs, minlength=256).astype(float)
    radius = max(2, min_gap // 3)
    density = np.convolve(hist, np.ones(2 * radius + 1), mode="same")
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
    gs16 = gs.astype(np.int16, copy=False)
    radius = max(3, min_gap // 3)
    for shade in shades:
        mask = np.abs(gs16 - shade) <= radius
        threshold = int(round(float(np.average(gs[mask])))) if np.any(mask) else shade
        bgr = tuple(int(round(v)) for v in np.median(bgrs[mask], axis=0)) if np.any(mask) else (shade,) * 3
        result.append({"threshold": int(np.clip(threshold, 0, 255)), "bgr": bgr})
    result.sort(key=lambda x: x["threshold"])
    return result


def bgr_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def text_color(bgr):
    b, g, r = bgr
    return "#111111" if 0.114 * b + 0.587 * g + 0.299 * r >= 150 else "#f7f7f7"


class DetectorApp:
    def __init__(self, root, image, gray, palette, args):
        self.root, self.image, self.gray, self.palette, self.args = root, image, gray, palette, args
        self.last_binary = self.last_result = None
        self.binary_photo = self.result_photo = None
        self.resize_job = None
        self.output_path = args.output

        self.threshold = tk.IntVar(value=args.threshold)
        self.min_radius = tk.IntVar(value=round(args.min_radius))
        self.max_radius = tk.IntVar(value=round(args.max_radius))
        self.max_error = tk.DoubleVar(value=args.max_error * 100)
        self.min_coverage = tk.IntVar(value=round(args.min_coverage * 100))
        self.morphology = tk.BooleanVar(value=args.morphology)
        self.output_var = tk.StringVar(value=args.output)
        self.status = tk.StringVar(value="Adjust settings, then click Apply.")

        root.title("Circle / Arc Detector")
        root.minsize(980, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        self.build_controls()
        self.build_previews()
        root.bind("<Return>", lambda _e: self.apply())
        root.bind("<Control-s>", lambda _e: self.save())
        root.bind("<Escape>", lambda _e: root.destroy())

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)
        radius_limit = max(100, round(self.args.max_radius * 2))
        rows = [
            ("Brightness threshold (0=black, 255=white)", self.threshold, 0, 255, 1, lambda x: str(int(x))),
            ("Minimum fitted circle radius (px)", self.min_radius, 1, radius_limit, 1, lambda x: f"{int(x)} px"),
            ("Maximum fitted circle radius (px)", self.max_radius, 1, radius_limit, 1, lambda x: f"{int(x)} px"),
            ("Maximum average radial error (% of radius)", self.max_error, 0.5, 50, 0.1, lambda x: f"{float(x):.1f}%"),
            ("Minimum visible circle arc (%)", self.min_coverage, 0, 100, 1, lambda x: f"{int(x)}% (~{int(x)*3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)
        tk.Checkbutton(frame, text="Use 3x3 morphology cleanup", variable=self.morphology).grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 8))
        self.morphology.trace_add("write", self.pending)
        tk.Label(frame, text="Pick threshold from image colors:").grid(row=6, column=0, sticky="nw")
        palette_frame = tk.Frame(frame)
        palette_frame.grid(row=6, column=1, columnspan=2, sticky="w", pady=(0, 8))
        for i, item in enumerate(self.palette):
            tk.Button(palette_frame, text=str(item["threshold"]), width=4, bg=bgr_hex(item["bgr"]), fg=text_color(item["bgr"]), command=lambda v=item["threshold"]: self.pick(v)).grid(row=i // 8, column=i % 8, padx=2, pady=2)
        buttons = tk.Frame(frame)
        buttons.grid(row=7, column=0, columnspan=3, sticky="ew")
        buttons.columnconfigure(4, weight=1)
        tk.Button(buttons, text="Apply", width=12, command=self.apply).grid(row=0, column=0, padx=(0, 8))
        tk.Button(buttons, text="Save", width=12, command=self.save).grid(row=0, column=1, padx=(0, 8))
        tk.Button(buttons, text="Save As...", width=12, command=self.save_as).grid(row=0, column=2, padx=(0, 8))
        tk.Label(buttons, text="Output file:").grid(row=0, column=3)
        tk.Entry(buttons, textvariable=self.output_var).grid(row=0, column=4, sticky="ew", padx=(6, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=1100).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=38, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Scale(parent, from_=low, to=high, orient=tk.HORIZONTAL, resolution=resolution, variable=variable, showvalue=False, length=420, highlightthickness=0).grid(row=row, column=1, sticky="ew", pady=2)
        value = tk.Label(parent, width=18, anchor="e")
        value.grid(row=row, column=2, pady=2)
        def update(*_):
            value.config(text=formatter(variable.get()))
            self.pending()
        variable.trace_add("write", update)
        update()

    def build_previews(self):
        # Tuple padding belongs to grid()/pack(), not a Tk widget constructor.
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

    def pending(self, *_):
        if self.last_result is not None:
            self.status.set("Settings changed. Click Apply to recompute the result.")

    def settings(self):
        min_r = max(1.0, float(self.min_radius.get()))
        max_r = max(1.0, float(self.max_radius.get()))
        if max_r < min_r:
            max_r = min_r
            self.max_radius.set(round(max_r))
        return int(self.threshold.get()), min_r, max_r, max(0.001, self.max_error.get() / 100), float(np.clip(self.min_coverage.get() / 100, 0, 1))

    def apply(self):
        threshold, min_r, max_r, max_err, min_cov = self.settings()
        started = time.perf_counter()
        binary, result, circles = process_image(self.image, self.gray, threshold, min_r, max_r, max_err, min_cov, self.morphology.get(), self.args.max_contours, self.args.max_search_points)
        elapsed = (time.perf_counter() - started) * 1000
        self.last_binary, self.last_result = binary, result
        self.redraw()
        self.status.set(f"Applied: T={threshold}; radius={min_r:.0f}-{max_r:.0f}px; error={max_err:.1%}; arc={min_cov:.0%}; {len(circles)} circle(s); {elapsed:.1f} ms.")
        for c in circles:
            print(f"{c['class']}: center=({c['center'][0]:.2f}, {c['center'][1]:.2f}), radius={c['radius']:.2f}, arc={c['coverage']*360:.1f}°, error={c['relative_error']:.4f}, interior={c['interior_fraction']:.1%}")

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
        cw, ch = max(2, canvas.winfo_width() - 2), max(2, canvas.winfo_height() - 2)
        h, w = image.shape[:2]
        scale = max(min(cw / w, ch / h), 1e-6)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        fitted = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(cw // 2 + 1, ch // 2 + 1, image=photo, anchor="center")
        return photo

    @staticmethod
    def placeholder(canvas, text):
        canvas.create_text(160, 120, text=text + "\nClick Apply", fill="#cccccc", justify="center")

    def save_as(self):
        path = filedialog.asksaveasfilename(title="Save result", defaultextension=".png", initialfile=self.output_var.get(), filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")])
        if path:
            self.output_path = path
            self.output_var.set(path)

    def save(self):
        if self.last_result is None:
            self.status.set("Nothing to save yet. Click Apply first.")
            return
        path = self.output_var.get().strip() or self.output_path
        if not path:
            self.save_as()
            path = self.output_var.get().strip()
        if not path:
            return
        if not cv2.imwrite(path, self.last_result):
            messagebox.showerror("Save failed", f"Could not write output image:\n{path}")
            return
        self.output_path = path
        self.output_var.set(path)
        self.status.set(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Detect up to two inferred circles from thresholded image arcs.")
    p.add_argument("image")
    p.add_argument("--threshold", type=int, default=128)
    p.add_argument("--min-radius", type=float, default=10.0)
    p.add_argument("--max-radius", type=float, default=0.0, help="0 = largest image dimension")
    p.add_argument("--max-error", type=float, default=0.08)
    p.add_argument("--min-coverage", type=float, default=0.12)
    p.add_argument("--morphology", action="store_true")
    p.add_argument("--max-contours", type=int, default=100)
    p.add_argument("--max-search-points", type=int, default=500)
    p.add_argument("--palette-size", type=int, default=12)
    p.add_argument("--palette-min-gap", type=int, default=15)
    p.add_argument("--output", default="detected_circles.png")
    args = p.parse_args()
    if not 0 <= args.threshold <= 255: p.error("--threshold must be 0..255")
    if args.min_radius <= 0: p.error("--min-radius must be > 0")
    if args.max_radius < 0: p.error("--max-radius must be >= 0")
    if args.max_error <= 0: p.error("--max-error must be > 0")
    if not 0 <= args.min_coverage <= 1: p.error("--min-coverage must be 0..1")
    if args.max_contours < 0 or args.max_search_points < 0: p.error("search limits must be >= 0")
    if args.palette_size < 1 or not 1 <= args.palette_min_gap <= 255: p.error("invalid palette settings")
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
