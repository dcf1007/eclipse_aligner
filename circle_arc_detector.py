import argparse
import math
import time

import cv2
import numpy as np


MORPH_KERNEL = np.ones((3, 3), dtype=np.uint8)

CONTROL_BAR_HEIGHT = 154
APPLY_RECT = (10, 9, 130, 36)
SAVE_RECT = (150, 9, 130, 36)
PALETTE_LABEL_X = 300
PALETTE_Y = 55
PALETTE_START_X = 410
SWATCH_WIDTH = 42
SWATCH_HEIGHT = 30
SWATCH_GAP = 7

MIN_INTERIOR_FRACTION = 0.50
MAX_CLASS_CANDIDATES = 30
MAX_REGIONS_PER_CONTOUR = 4

TRACK_THRESHOLD = "Threshold brightness (0-255)"
TRACK_MIN_RADIUS = "Minimum circle radius (px)"
TRACK_MAX_RADIUS = "Maximum circle radius (px)"
TRACK_MAX_ERROR = "Maximum avg radial error (%)"
TRACK_MIN_COVERAGE = "Minimum visible arc (%)"


def _fit_circle(points):
    """Fast centered algebraic circle fit for Nx2 points."""
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
    """Fraction of a full circle spanned by an ordered contour region."""
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

    start %= contour_point_count
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


def _find_regions(
    points,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    min_region_points=12,
):
    """Find distinct circular arcs that may share one composite contour."""
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
    seed_limit = max(8, MAX_REGIONS_PER_CONTOUR * 4)
    for candidate in coarse:
        if any(_same_circle(candidate, existing, 0.10, 0.10) for existing in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= seed_limit:
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
    """Fraction of the candidate interior belonging to this threshold class."""
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
        match_fraction, visible_pixels = _interior_match_fraction(
            class_mask,
            candidate,
            excluded_circle,
        )
        if visible_pixels < 16 or match_fraction < MIN_INTERIOR_FRACTION:
            continue

        score = candidate["score"] * match_fraction**2
        if score > best_score:
            best = candidate.copy()
            best["class"] = label
            best["interior_fraction"] = match_fraction
            best_score = score

    return best


def detect_circles(
    gray,
    threshold,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    morphology=False,
    max_contours=100,
    max_search_points=500,
):
    """
    Return at most two circles: one <= threshold and one > threshold.

    The <= threshold circle is selected first and owns geometric overlap.
    """
    threshold = int(threshold)
    _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if morphology:
        dark_detection = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        bright_detection = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    else:
        dark_detection = dark_mask
        bright_detection = bright_mask

    candidate_args = (
        min_radius,
        max_radius,
        max_relative_error,
        min_coverage,
        max_contours,
        max_search_points,
    )
    dark_candidates = _find_candidates(dark_detection, *candidate_args)
    bright_candidates = _find_candidates(bright_detection, *candidate_args)

    dark_circle = _select_class_circle(dark_candidates, dark_mask, "below threshold")
    bright_circle = _select_class_circle(
        bright_candidates,
        bright_mask,
        "above threshold",
        excluded_circle=dark_circle,
    )

    detections = [circle for circle in (dark_circle, bright_circle) if circle is not None]
    return bright_mask, detections


def process_image(
    image,
    gray,
    threshold,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    morphology,
    max_contours,
    max_search_points,
):
    binary, detections = detect_circles(
        gray,
        threshold,
        min_radius,
        max_radius,
        max_relative_error,
        min_coverage,
        morphology,
        max_contours,
        max_search_points,
    )

    output = image.copy()
    for detection in detections:
        cx, cy = detection["center"]
        radius = detection["radius"]
        center = (int(round(cx)), int(round(cy)))
        radius_int = int(round(radius))

        support = np.rint(detection["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support) >= 2:
            cv2.polylines(output, [support], False, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.circle(output, center, radius_int, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

        class_label = "DARK <= T" if detection["class"] == "below threshold" else "BRIGHT > T"
        label = (
            f"{class_label}  r={radius:.1f}  "
            f"arc={detection['coverage'] * 360.0:.0f}deg  "
            f"err={detection['relative_error']:.3f}"
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

    return binary, output, detections


def build_threshold_palette(image, gray, max_colors=12, min_shade_gap=15):
    """Dominant source colors separated by meaningful grayscale brightness."""
    sample_limit = 200_000
    stride = max(1, int(math.ceil(math.sqrt(gray.size / sample_limit))))
    sample_gray = gray[::stride, ::stride].reshape(-1)
    sample_bgr = image[::stride, ::stride].reshape(-1, 3)

    hist = np.bincount(sample_gray, minlength=256).astype(np.float64)
    smooth_radius = max(2, min_shade_gap // 3)
    density = np.convolve(
        hist,
        np.ones(2 * smooth_radius + 1, dtype=np.float64),
        mode="same",
    )

    shades = []
    for shade in np.argsort(density)[::-1]:
        shade = int(shade)
        if density[shade] <= 0:
            break
        if any(abs(shade - existing) < min_shade_gap for existing in shades):
            continue
        shades.append(shade)
        if len(shades) >= max_colors:
            break

    palette = []
    sample_gray_i16 = sample_gray.astype(np.int16, copy=False)
    color_radius = max(3, min_shade_gap // 3)

    for shade in shades:
        mask = np.abs(sample_gray_i16 - shade) <= color_radius
        matching_gray = sample_gray[mask]
        matching_bgr = sample_bgr[mask]

        threshold = int(round(float(matching_gray.mean()))) if len(matching_gray) else shade
        bgr = (
            tuple(int(round(value)) for value in np.median(matching_bgr, axis=0))
            if len(matching_bgr)
            else (shade, shade, shade)
        )
        palette.append(
            {
                "threshold": int(np.clip(threshold, 0, 255)),
                "bgr": bgr,
                "strength": float(density[shade]),
            }
        )

    palette.sort(key=lambda item: item["threshold"])
    deduped = []
    for item in palette:
        if not deduped or item["threshold"] - deduped[-1]["threshold"] >= min_shade_gap:
            deduped.append(item)
        elif item["strength"] > deduped[-1]["strength"]:
            deduped[-1] = item

    for item in deduped:
        del item["strength"]

    return deduped


def _point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def _palette_rect(index):
    return (
        PALETTE_START_X + index * (SWATCH_WIDTH + SWATCH_GAP),
        PALETTE_Y,
        SWATCH_WIDTH,
        SWATCH_HEIGHT,
    )


def _draw_button(bar, rect, label):
    x, y, w, h = rect
    cv2.rectangle(bar, (x, y), (x + w, y + h), (215, 215, 215), -1)
    cv2.rectangle(bar, (x, y), (x + w, y + h), (245, 245, 245), 1)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    cv2.putText(
        bar,
        label,
        (x + (w - text_size[0]) // 2, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _make_display(binary, result, palette, applied, elapsed_ms):
    preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    body = np.hstack((preview, result))

    scale = min(1.0, 1600 / body.shape[1], 820 / body.shape[0])
    if scale < 1.0:
        body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    palette_width = PALETTE_START_X + len(palette) * (SWATCH_WIDTH + SWATCH_GAP) + 10
    min_width = max(1200, palette_width)
    if body.shape[1] < min_width:
        body = cv2.copyMakeBorder(
            body,
            0,
            0,
            0,
            min_width - body.shape[1],
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    bar = np.full((CONTROL_BAR_HEIGHT, body.shape[1], 3), 42, dtype=np.uint8)
    _draw_button(bar, APPLY_RECT, "Apply")
    _draw_button(bar, SAVE_RECT, "Save")

    threshold, min_radius, max_radius, max_error, min_coverage = applied
    status = (
        f"Applied: threshold={threshold}   radius={min_radius:.0f}-{max_radius:.0f}px   "
        f"max avg radial error={max_error:.0%}   min visible arc={min_coverage:.0%}   "
        f"{elapsed_ms:.1f} ms"
    )
    cv2.putText(
        bar,
        status,
        (10, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        bar,
        "Detection: <= threshold = dark circle; > threshold = bright circle; "
        "dark circle owns overlap. Max 2 total.",
        (10, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        bar,
        "Pick threshold color:",
        (PALETTE_LABEL_X, PALETTE_Y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    for index, item in enumerate(palette):
        x, y, w, h = _palette_rect(index)
        cv2.rectangle(bar, (x, y), (x + w, y + h), item["bgr"], -1)

        selected = item["threshold"] == threshold
        border = (255, 255, 255) if selected else (120, 120, 120)
        cv2.rectangle(bar, (x, y), (x + w, y + h), border, 2 if selected else 1)

        text_color = (20, 20, 20) if item["threshold"] >= 150 else (245, 245, 245)
        label = str(item["threshold"])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        cv2.putText(
            bar,
            label,
            (x + (w - text_size[0]) // 2, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return np.vstack((bar, body))


def _print_detections(detections):
    if not detections:
        print("No accepted circles.")
        return

    for detection in detections:
        cx, cy = detection["center"]
        print(
            f"{detection['class']}: "
            f"center=({cx:.2f}, {cy:.2f}), "
            f"radius={detection['radius']:.2f}, "
            f"coverage={detection['coverage'] * 360.0:.1f} deg, "
            f"relative_error={detection['relative_error']:.4f}, "
            f"interior_match={detection['interior_fraction']:.1%}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detect at most two inferred circles: one with an interior at/below "
            "the threshold and one above it. The dark circle owns overlap."
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
        help="Initial minimum visible circle fraction, 0..1",
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
        help="Maximum largest contours searched per threshold class; 0 means unlimited",
    )
    parser.add_argument(
        "--max-search-points",
        type=int,
        default=500,
        help="Maximum ordered contour points searched; 0 means unlimited",
    )
    parser.add_argument(
        "--palette-size",
        type=int,
        default=12,
        help="Maximum threshold color choices (default: 12)",
    )
    parser.add_argument(
        "--palette-min-gap",
        type=int,
        default=15,
        help="Minimum grayscale difference between color choices (default: 15)",
    )
    parser.add_argument(
        "--output",
        default="detected_circles.png",
        help="Output image written by Save or the S key",
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

    return args


def main():
    args = _parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_max_dim = max(gray.shape)
    initial_max_radius = max(
        args.min_radius,
        args.max_radius if args.max_radius > 0 else float(image_max_dim),
    )
    radius_slider_limit = max(
        100,
        int(math.ceil(image_max_dim * 2.0)),
        int(math.ceil(args.min_radius)),
        int(math.ceil(initial_max_radius)),
    )
    error_slider_limit = max(50, int(math.ceil(args.max_error * 100.0)))

    palette = build_threshold_palette(
        image,
        gray,
        args.palette_size,
        args.palette_min_gap,
    )

    window_name = "Circle / Arc Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar(TRACK_THRESHOLD, window_name, args.threshold, 255, lambda _: None)
    cv2.createTrackbar(
        TRACK_MIN_RADIUS,
        window_name,
        int(round(args.min_radius)),
        radius_slider_limit,
        lambda _: None,
    )
    cv2.createTrackbar(
        TRACK_MAX_RADIUS,
        window_name,
        int(round(initial_max_radius)),
        radius_slider_limit,
        lambda _: None,
    )
    cv2.createTrackbar(
        TRACK_MAX_ERROR,
        window_name,
        max(1, int(round(args.max_error * 100.0))),
        error_slider_limit,
        lambda _: None,
    )
    cv2.createTrackbar(
        TRACK_MIN_COVERAGE,
        window_name,
        int(round(args.min_coverage * 100.0)),
        100,
        lambda _: None,
    )

    state = {
        "apply_requested": True,
        "save_requested": False,
        "result": None,
    }

    def on_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONUP:
            return

        if _point_in_rect(x, y, APPLY_RECT):
            state["apply_requested"] = True
            return

        if _point_in_rect(x, y, SAVE_RECT):
            state["save_requested"] = True
            return

        for index, item in enumerate(palette):
            if _point_in_rect(x, y, _palette_rect(index)):
                cv2.setTrackbarPos(TRACK_THRESHOLD, window_name, item["threshold"])
                print(
                    f"Selected threshold {item['threshold']} from color palette; "
                    "click Apply to process."
                )
                return

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state["apply_requested"]:
            state["apply_requested"] = False

            threshold = cv2.getTrackbarPos(TRACK_THRESHOLD, window_name)
            min_radius = max(1.0, float(cv2.getTrackbarPos(TRACK_MIN_RADIUS, window_name)))
            max_radius = max(1.0, float(cv2.getTrackbarPos(TRACK_MAX_RADIUS, window_name)))
            if max_radius < min_radius:
                max_radius = min_radius
                cv2.setTrackbarPos(TRACK_MAX_RADIUS, window_name, int(round(max_radius)))

            max_error = max(
                0.01,
                cv2.getTrackbarPos(TRACK_MAX_ERROR, window_name) / 100.0,
            )
            min_coverage = cv2.getTrackbarPos(TRACK_MIN_COVERAGE, window_name) / 100.0

            started = time.perf_counter()
            binary, result, detections = process_image(
                image,
                gray,
                threshold,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
                args.morphology,
                args.max_contours,
                args.max_search_points,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            state["result"] = result
            applied = (threshold, min_radius, max_radius, max_error, min_coverage)
            cv2.imshow(window_name, _make_display(binary, result, palette, applied, elapsed_ms))

            print(
                f"Applied: threshold={threshold}, "
                f"min_radius={min_radius:.0f}, max_radius={max_radius:.0f}, "
                f"max_relative_error={max_error:.3f}, "
                f"min_coverage={min_coverage:.2f}: "
                f"{len(detections)} circle(s), {elapsed_ms:.1f} ms"
            )
            _print_detections(detections)

        if state["save_requested"]:
            state["save_requested"] = False
            if state["result"] is not None:
                if not cv2.imwrite(args.output, state["result"]):
                    raise RuntimeError(f"Could not write output image: {args.output}")
                print(f"Saved: {args.output}")

        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (13, 32):
            state["apply_requested"] = True
        elif key == ord("s"):
            state["save_requested"] = True

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
