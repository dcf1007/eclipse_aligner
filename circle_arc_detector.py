import argparse
import math
import time

import cv2
import numpy as np


CONTROL_BAR_HEIGHT = 104
APPLY_RECT = (10, 9, 120, 35)
SAVE_RECT = (140, 9, 120, 35)
PALETTE_LABEL_X = 10
PALETTE_Y = 58
PALETTE_START_X = 124
SWATCH_WIDTH = 42
SWATCH_HEIGHT = 30
SWATCH_GAP = 8
ERROR_SCALE = 1000
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
    cv2.putText(
        bar,
        label,
        (x + 28, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def _make_display(
    binary_for_detection,
    result,
    applied_threshold,
    elapsed_ms,
    palette,
    applied_min_radius,
    applied_max_radius,
    applied_max_error,
    applied_min_coverage,
):
    """Build the preview only after Apply; pending slider changes do not redraw it."""
    threshold_preview = cv2.cvtColor(binary_for_detection, cv2.COLOR_GRAY2BGR)
    body = np.hstack((threshold_preview, result))

    palette_width = PALETTE_START_X + len(palette) * (SWATCH_WIDTH + SWATCH_GAP) + 10
    min_width = max(760, palette_width)
    max_width = 1600
    max_body_height = 820
    scale = min(1.0, max_width / body.shape[1], max_body_height / body.shape[0])
    if scale < 1.0:
        body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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

    status = (
        f"Applied: T={applied_threshold}  "
        f"R={applied_min_radius:.0f}-{applied_max_radius:.0f}  "
        f"err={applied_max_error:.3f}  "
        f"cov={applied_min_coverage:.0%}  "
        f"{elapsed_ms:.1f} ms"
    )
    cv2.putText(
        bar,
        status,
        (280, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        bar,
        "Pick color:",
        (PALETTE_LABEL_X, PALETTE_Y + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )

    for index, item in enumerate(palette):
        x, y, w, h = _palette_rect(index)
        cv2.rectangle(bar, (x, y), (x + w, y + h), item["bgr"], -1)
        selected = item["threshold"] == applied_threshold
        border = (255, 255, 255) if selected else (120, 120, 120)
        cv2.rectangle(bar, (x, y), (x + w, y + h), border, 2 if selected else 1)

        text_color = (20, 20, 20) if item["threshold"] >= 150 else (245, 245, 245)
        label = str(item["threshold"])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        cv2.putText(
            bar,
            label,
            (x + (w - text_size[0]) // 2, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return np.vstack((bar, body))


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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_max_dim = max(gray.shape)
    initial_max_radius = args.max_radius if args.max_radius > 0 else float(image_max_dim)
    if initial_max_radius < args.min_radius:
        initial_max_radius = args.min_radius

    radius_slider_limit = max(
        100,
        int(math.ceil(image_max_dim * 2.0)),
        int(math.ceil(args.min_radius)),
        int(math.ceil(initial_max_radius)),
    )
    error_slider_limit = max(500, int(math.ceil(args.max_error * ERROR_SCALE)))

    palette = build_threshold_palette(
        image,
        gray,
        max_colors=args.palette_size,
        min_shade_gap=args.palette_min_gap,
    )

    window_name = "Circle / Arc Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # These trackbars only store pending values. No processing callback is attached.
    cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda _value: None)
    cv2.createTrackbar(
        "Min radius",
        window_name,
        int(round(args.min_radius)),
        radius_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Max radius",
        window_name,
        int(round(initial_max_radius)),
        radius_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Max error x1000",
        window_name,
        max(1, int(round(args.max_error * ERROR_SCALE))),
        error_slider_limit,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Min coverage %",
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
                cv2.setTrackbarPos("Threshold", window_name, item["threshold"])
                print(
                    f"Selected threshold {item['threshold']} from color palette; "
                    "click Apply to process."
                )
                return

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state["apply_requested"]:
            state["apply_requested"] = False

            threshold_value = cv2.getTrackbarPos("Threshold", window_name)
            min_radius = max(1.0, float(cv2.getTrackbarPos("Min radius", window_name)))
            max_radius = max(1.0, float(cv2.getTrackbarPos("Max radius", window_name)))
            if max_radius < min_radius:
                max_radius = min_radius
                cv2.setTrackbarPos("Max radius", window_name, int(round(max_radius)))

            max_relative_error = max(
                1.0 / ERROR_SCALE,
                cv2.getTrackbarPos("Max error x1000", window_name) / ERROR_SCALE,
            )
            min_coverage = cv2.getTrackbarPos("Min coverage %", window_name) / 100.0

            started = time.perf_counter()
            _, binary_for_detection, result, detections = process_image(
                image,
                threshold_value,
                invert=args.invert,
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
            display = _make_display(
                binary_for_detection,
                result,
                threshold_value,
                elapsed_ms,
                palette,
                min_radius,
                max_radius,
                max_relative_error,
                min_coverage,
            )
            cv2.imshow(window_name, display)

            print(
                f"Applied: threshold={threshold_value}, "
                f"min_radius={min_radius:.0f}, max_radius={max_radius:.0f}, "
                f"max_relative_error={max_relative_error:.3f}, "
                f"min_coverage={min_coverage:.2f}: "
                f"{len(detections)} object(s), {elapsed_ms:.1f} ms"
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
