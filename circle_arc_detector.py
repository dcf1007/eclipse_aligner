import argparse
import math
import time

import cv2
import numpy as np

from circle_arc_detector_core import (
    MORPH_KERNEL,
    _candidate_window_lengths,
    _evaluate_circular_region,
    _hill_climb_region,
    _downsample_contour_points,
    build_threshold_palette,
)


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


def _same_circle_geometry(a, b, center_fraction=0.12, radius_fraction=0.12):
    ax, ay = a["center"]
    bx, by = b["center"]
    ar = a["radius"]
    br = b["radius"]
    scale = max(ar, br, 1.0)
    return (
        math.hypot(ax - bx, ay - by) < center_fraction * scale
        and abs(ar - br) < radius_fraction * scale
    )


def find_circular_regions(
    points,
    min_radius,
    max_radius,
    max_relative_error,
    min_coverage,
    min_region_points=12,
    max_regions=4,
):
    """Return several geometrically distinct circular arcs from one contour."""
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)
    if point_count < max(3, min_region_points):
        return []

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
        return []

    coarse_candidates.sort(key=lambda item: item["score"], reverse=True)

    seeds = []
    for candidate in coarse_candidates:
        if any(_same_circle_geometry(candidate, existing, 0.10, 0.10) for existing in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= max(8, max_regions * 4):
            break

    refined_candidates = []
    for seed in seeds:
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
        if any(_same_circle_geometry(refined, existing) for existing in refined_candidates):
            continue
        refined_candidates.append(refined)

    refined_candidates.sort(key=lambda item: item["score"], reverse=True)
    return refined_candidates[:max_regions]


def find_circular_candidates(
    binary,
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
        candidates.extend(
            find_circular_regions(
                points,
                min_radius=min_radius,
                max_radius=max_radius,
                max_relative_error=max_relative_error,
                min_coverage=min_coverage,
                min_region_points=min_region_points,
                max_regions=4,
            )
        )

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def _interior_fraction(class_mask, candidate, excluded_circle=None):
    height, width = class_mask.shape[:2]
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

    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0, 0

    roi = class_mask[y0:y1, x0:x1]
    class_count = int(np.count_nonzero(roi[valid]))
    return class_count / valid_count, valid_count


def _choose_class_candidate(candidates, class_mask, label, excluded_circle=None):
    best = None
    best_class_score = -math.inf

    for candidate in candidates[:30]:
        fraction, visible_pixels = _interior_fraction(
            class_mask,
            candidate,
            excluded_circle=excluded_circle,
        )
        if visible_pixels < 16 or fraction < MIN_INTERIOR_FRACTION:
            continue

        class_score = candidate["score"] * (fraction**2)
        if class_score > best_class_score:
            best = candidate.copy()
            best["class"] = label
            best["interior_fraction"] = fraction
            best_class_score = class_score

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
        min_contour_points=12,
        min_radius=min_radius,
        max_radius=max_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
        max_contours=max_contours,
        max_search_points=max_search_points,
    )

    dark_candidates = find_circular_candidates(dark_detection_mask, **common)
    bright_candidates = find_circular_candidates(bright_detection_mask, **common)

    dark_circle = _choose_class_candidate(
        dark_candidates,
        dark_mask,
        label="below threshold",
    )
    bright_circle = _choose_class_candidate(
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

    binary, detections = detect_dark_bright_circles(
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

        coverage_degrees = detection["coverage"] * 360.0
        short_class = "DARK <= T" if detection["class"] == "below threshold" else "BRIGHT > T"
        label = (
            f"{short_class}  r={radius:.1f}  "
            f"arc={coverage_degrees:.0f}deg  "
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


def _make_display(binary, result, palette, applied_settings, elapsed_ms):
    threshold_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    body = np.hstack((threshold_preview, result))

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

    threshold, min_radius, max_radius, max_error, min_coverage = applied_settings
    status = (
        f"Applied: threshold={threshold}   radius={min_radius:.0f}-{max_radius:.0f}px   "
        f"max avg radial error={max_error:.0%}   min visible arc={min_coverage:.0%}   "
        f"{elapsed_ms:.1f} ms"
    )
    cv2.putText(bar, status, (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA)

    help_text = (
        "Detection: <= threshold = dark-interior circle; > threshold = bright-interior circle; "
        "dark circle owns overlap. Max 1 of each (2 total)."
    )
    cv2.putText(bar, help_text, (10, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 205, 205), 1, cv2.LINE_AA)

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image and detect at most two inferred circles: one with "
            "an interior at/below the threshold and one above it."
        )
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--threshold", type=int, default=128, help="Initial threshold, 0-255")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Deprecated compatibility option; both threshold sides are now detected explicitly",
    )
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
        help="Maximum largest contours searched per threshold class; 0 means unlimited",
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

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    if args.invert:
        print("Note: --invert is no longer needed; both threshold classes are detected explicitly.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_max_dim = max(gray.shape)
    initial_max_radius = args.max_radius if args.max_radius > 0 else float(image_max_dim)
    initial_max_radius = max(initial_max_radius, args.min_radius)
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
        max_colors=args.palette_size,
        min_shade_gap=args.palette_min_gap,
    )

    window_name = "Circle / Arc Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar(
        "Threshold brightness (0 black - 255 white)",
        window_name,
        args.threshold,
        255,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Minimum fitted circle radius (px)",
        window_name,
        int(round(args.min_radius)),
        radius_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Maximum fitted circle radius (px)",
        window_name,
        int(round(initial_max_radius)),
        radius_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Maximum average radial error (%)",
        window_name,
        max(1, int(round(args.max_error * 100.0))),
        error_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Minimum visible circle arc (%)",
        window_name,
        int(round(args.min_coverage * 100.0)),
        100,
        lambda _value: None,
    )

    state = {
        "apply_requested": True,
        "save_requested": False,
        "result": None,
        "detections": [],
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
                cv2.setTrackbarPos(
                    "Threshold brightness (0 black - 255 white)",
                    window_name,
                    item["threshold"],
                )
                print(
                    f"Selected threshold {item['threshold']} from color palette; "
                    "click Apply to process."
                )
                return

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state["apply_requested"]:
            state["apply_requested"] = False

            threshold_value = cv2.getTrackbarPos(
                "Threshold brightness (0 black - 255 white)",
                window_name,
            )
            min_radius = max(
                1.0,
                float(cv2.getTrackbarPos("Minimum fitted circle radius (px)", window_name)),
            )
            max_radius = max(
                1.0,
                float(cv2.getTrackbarPos("Maximum fitted circle radius (px)", window_name)),
            )
            if max_radius < min_radius:
                max_radius = min_radius
                cv2.setTrackbarPos(
                    "Maximum fitted circle radius (px)",
                    window_name,
                    int(round(max_radius)),
                )

            max_relative_error = max(
                0.01,
                cv2.getTrackbarPos("Maximum average radial error (%)", window_name) / 100.0,
            )
            min_coverage = (
                cv2.getTrackbarPos("Minimum visible circle arc (%)", window_name) / 100.0
            )

            started = time.perf_counter()
            binary, result, detections = process_image(
                image,
                threshold_value,
                min_radius=min_radius,
                max_radius=max_radius,
                max_relative_error=max_relative_error,
                min_coverage=min_coverage,
                morphology=args.morphology,
                max_contours=args.max_contours,
                max_search_points=args.max_search_points,
                gray=gray,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            state["result"] = result
            state["detections"] = detections
            applied_settings = (
                threshold_value,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
            )
            display = _make_display(binary, result, palette, applied_settings, elapsed_ms)
            cv2.imshow(window_name, display)

            print(
                f"Applied: threshold={threshold_value}, "
                f"min_radius={min_radius:.0f}, max_radius={max_radius:.0f}, "
                f"max_relative_error={max_relative_error:.3f}, "
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
