import argparse
import math

import cv2
import numpy as np


def fit_circle_least_squares(points):
    """Fit a circle to Nx2 contour points using linear least squares."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # x^2 + y^2 = 2*cx*x + 2*cy*y + c
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x**2 + y**2

    try:
        solution, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        return None

    cx, cy, c = solution
    radius_squared = c + cx**2 + cy**2
    if radius_squared <= 0 or not np.isfinite(radius_squared):
        return None

    radius = math.sqrt(radius_squared)
    distances = np.hypot(x - cx, y - cy)
    mean_error = float(np.mean(np.abs(distances - radius)))

    if not all(np.isfinite(value) for value in (cx, cy, radius, mean_error)):
        return None

    return float(cx), float(cy), float(radius), mean_error


def angular_coverage(points, cx, cy):
    """Return the fraction (0..1) of a fitted circle covered by the points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    angles = np.mod(
        np.arctan2(points[:, 1] - cy, points[:, 0] - cx),
        2.0 * np.pi,
    )
    angles = np.sort(angles)

    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
    largest_gap = float(np.max(np.append(gaps, wrap_gap)))

    return float(np.clip(1.0 - largest_gap / (2.0 * np.pi), 0.0, 1.0))


def _candidate_window_lengths(point_count, min_region_points):
    """Generate coarse contiguous-region sizes from short arcs to full contour."""
    min_region_points = max(3, min(min_region_points, point_count))
    lengths = {min_region_points, point_count}

    length = min_region_points
    while length < point_count:
        lengths.add(length)
        next_length = max(length + 1, int(round(length * 1.35)))
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
    """
    Fit one contiguous section of a closed contour and score how circular it is.

    The region, not the whole contour, is used to estimate the circle.
    """
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

    # A tiny region can have an artificially tiny residual, so the score also
    # rewards angular coverage and the amount of contiguous contour support.
    # This lets a clean, meaningful arc beat a few coincidental points.
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
    """Refine the two boundaries of a coarse circular region one point at a time."""
    best = candidate

    # Boundary moves:
    #   start - 1, length + 1  -> grow at the beginning
    #   start,     length + 1  -> grow at the end
    #   start + 1, length - 1  -> shrink at the beginning
    #   start,     length - 1  -> shrink at the end
    moves = ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0))

    for _ in range(80):
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
    min_region_points=20,
):
    """
    Find the contiguous part of a closed contour that best follows a circle.

    A multi-scale sliding-window search produces candidate arcs. The strongest
    candidates are then refined by moving their boundaries point by point.
    """
    points = np.asarray(points, dtype=np.float64)
    point_count = len(points)

    if point_count < max(3, min_region_points):
        return None

    min_region_points = min(min_region_points, point_count)

    # Duplicate the contour once so a region may cross the contour's 0/N seam.
    extended_points = np.vstack((points, points))
    coarse_candidates = []

    for length in _candidate_window_lengths(point_count, min_region_points):
        # Search more densely for short regions and still sample long regions
        # sufficiently well. The final boundary refinement removes quantization.
        step = max(1, length // 8)

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

    # Refining several coarse seeds protects against the highest coarse score
    # landing near, but not exactly on, the best arc boundaries.
    best = None
    for seed in coarse_candidates[:8]:
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


def detect_circular_objects(
    binary,
    max_objects=2,
    min_contour_points=20,
    min_radius=10.0,
    max_radius=None,
    max_relative_error=0.08,
    min_coverage=0.12,
):
    """
    Detect up to max_objects circles from the best circular region of contours.

    The full contour is never required to be circular. For each contour the
    detector searches for the strongest contiguous circular arc and fits only
    that region.
    """
    height, width = binary.shape[:2]
    if max_radius is None:
        max_radius = float(max(width, height))

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    candidates = []
    min_region_points = max(12, min_contour_points)

    for contour in contours:
        if len(contour) < min_region_points:
            continue

        points = contour.reshape(-1, 2).astype(np.float64)

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

    # RETR_LIST can expose both sides of the same edge. Remove near-identical
    # circle fits so one physical circle does not consume both result slots.
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
):
    """Threshold an image, find circular regions, and overlay inferred circles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    _, binary = cv2.threshold(
        gray,
        int(threshold_value),
        255,
        threshold_mode,
    )

    binary_for_detection = binary
    if morphology:
        kernel = np.ones((3, 3), np.uint8)
        binary_for_detection = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

    detections = detect_circular_objects(
        binary_for_detection,
        max_objects=2,
        min_contour_points=20,
        min_radius=min_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
    )

    output = image.copy()

    for index, detection in enumerate(detections, start=1):
        cx, cy = detection["center"]
        radius = detection["radius"]
        center = (int(round(cx)), int(round(cy)))
        radius_int = int(round(radius))

        # Show the exact contour region that was used for the fit in green.
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

        # Draw the complete circle inferred from that region in red.
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
        "--output",
        default="detected_circles.png",
        help="Output image written when S is pressed",
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

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    window_name = "Circle / Arc Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(
        "Threshold",
        window_name,
        args.threshold,
        255,
        lambda _value: None,
    )

    while True:
        threshold_value = cv2.getTrackbarPos("Threshold", window_name)

        _, binary_for_detection, result, detections = process_image(
            image,
            threshold_value,
            invert=args.invert,
            min_radius=args.min_radius,
            max_relative_error=args.max_error,
            min_coverage=args.min_coverage,
            morphology=args.morphology,
        )

        threshold_preview = cv2.cvtColor(
            binary_for_detection,
            cv2.COLOR_GRAY2BGR,
        )
        display = np.hstack((threshold_preview, result))
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

        if key == ord("s"):
            if not cv2.imwrite(args.output, result):
                raise RuntimeError(f"Could not write output image: {args.output}")

            print(f"Saved: {args.output}")
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

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
