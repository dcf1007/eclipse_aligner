import argparse
import math
import time

import cv2
import numpy as np


CONTROL_BAR_HEIGHT = 52
APPLY_RECT = (10, 9, 120, 35)
SAVE_RECT = (140, 9, 120, 35)
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
    scale = suu * svv + 1.0
    if abs(det) <= 1e-12 * scale:
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
    """Estimate angular coverage of a contiguous contour region without sorting."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    unwrapped = np.unwrap(angles)
    span = float(np.ptp(unwrapped))
    return float(np.clip(span / (2.0 * np.pi), 0.0, 1.0))


def _candidate_window_lengths(point_count, min_region_points):
    """Generate coarse contiguous-region sizes from short arcs to full contour."""
    min_region_points = max(3, min(min_region_points, point_count))
    lengths = {min_region_points, point_count}

    length = min_region_points
    while length < point_count:
        lengths.add(length)
        next_length = max(length + 1, int(round(length * 1.45)))
        length = min(point_count, next_length)

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
    """Fit and score one contiguous section of a closed contour."""
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

    error_floor = 0.002
    support_fraction = length / contour_point_count
    score = (
        (coverage**1.5)
        * math.sqrt(float(length))
        * (0.75 + 0.25 * math.sqrt(support_fraction))
        / (relative_error + error_floor)
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
    """Refine coarse arc boundaries locally."""
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
    """Find the contiguous part of a closed contour that best follows a circle."""
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
    """Keep ordered contour geometry while bounding circular-region search cost."""
    point_count = len(points)
    if max_search_points <= 0 or point_count <= max_search_points:
        return points

    indices = np.linspace(
        0,
        point_count - 1,
        max_search_points,
        dtype=np.int32,
    )
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
    """Detect up to max_objects circles from circular regions of contours."""
    height, width = binary.shape[:2]
    if max_radius is None:
        max_radius = float(max(width, height))

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

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
    max_relative_error=0.08,
    min_coverage=0.12,
    morphology=False,
    max_contours=100,
    max_search_points=500,
    gray=None,
):
    """Threshold an image, find circular regions, and overlay inferred circles."""
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, int(threshold_value), 255, threshold_mode)

    binary_for_detection = binary
    if morphology:
        binary_for_detection = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            MORPH_KERNEL,
        )

    detections = detect_circular_objects(
        binary_for_detection,
        max_objects=2,
        min_contour_points=12,
        min_radius=min_radius,
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
            cv2.polylines(
                output,
                [support_points],
                False,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

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


def _point_in_rect(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def _make_display(binary_for_detection, result, applied_threshold, elapsed_ms):
    """Build a bounded-size preview; called only after Apply."""
    threshold_preview = cv2.cvtColor(binary_for_detection, cv2.COLOR_GRAY2BGR)
    body = np.hstack((threshold_preview, result))

    max_width = 1600
    max_body_height = 820
    scale = min(1.0, max_width / body.shape[1], max_body_height / body.shape[0])
    if scale < 1.0:
        body = cv2.resize(
            body,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    min_width = 560
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

    for rect, label in ((APPLY_RECT, "Apply"), (SAVE_RECT, "Save")):
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

    status = f"Applied threshold: {applied_threshold}   processing: {elapsed_ms:.1f} ms"
    cv2.putText(
        bar,
        status,
        (280, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
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
            f"support_points={detection['region_length']}/"
            f"{detection['contour_points']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image and detect up to two circles by finding the "
            "contiguous region of each contour that follows a circle best."
        )
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Initial grayscale threshold, 0-255 (default: 128)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Use inverted binary thresholding",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=10.0,
        help="Minimum fitted circle radius in pixels (default: 10)",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.08,
        help="Maximum mean radial error divided by radius (default: 0.08)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.12,
        help="Minimum angular coverage of the selected circular region, 0..1 (default: 0.12)",
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
        help="Maximum largest contours searched; 0 means unlimited (default: 100)",
    )
    parser.add_argument(
        "--max-search-points",
        type=int,
        default=500,
        help="Maximum ordered points searched per contour; 0 means unlimited (default: 500)",
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
    if args.max_error <= 0:
        parser.error("--max-error must be greater than zero")
    if not 0.0 <= args.min_coverage <= 1.0:
        parser.error("--min-coverage must be between 0 and 1")
    if args.max_contours < 0:
        parser.error("--max-contours must be zero or greater")
    if args.max_search_points < 0:
        parser.error("--max-search-points must be zero or greater")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    window_name = "Circle / Arc Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar(
        "Threshold",
        window_name,
        args.threshold,
        255,
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
        elif _point_in_rect(x, y, SAVE_RECT):
            state["save_requested"] = True

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if state["apply_requested"]:
            state["apply_requested"] = False
            threshold_value = cv2.getTrackbarPos("Threshold", window_name)

            started = time.perf_counter()
            _, binary_for_detection, result, detections = process_image(
                image,
                threshold_value,
                invert=args.invert,
                min_radius=args.min_radius,
                max_relative_error=args.max_error,
                min_coverage=args.min_coverage,
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
            )
            cv2.imshow(window_name, display)

            print(
                f"Applied threshold {threshold_value}: "
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
