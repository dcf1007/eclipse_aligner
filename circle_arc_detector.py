import argparse
import math

import cv2
import numpy as np


def smooth_contour(points, passes):
    """Smooth a closed contour while preserving its point count."""
    smoothed = np.asarray(points, dtype=np.float64).copy()
    for _ in range(max(0, int(passes))):
        previous = np.roll(smoothed, 1, axis=0)
        following = np.roll(smoothed, -1, axis=0)
        smoothed = (previous + 2.0 * smoothed + following) / 4.0
    return smoothed


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
        solution, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    cx, cy, c = solution
    radius_squared = c + cx**2 + cy**2
    if radius_squared <= 0:
        return None

    radius = math.sqrt(radius_squared)
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mean_error = float(np.mean(np.abs(distances - radius)))

    return float(cx), float(cy), float(radius), mean_error


def circle_angular_coverage(points, cx, cy):
    """Return the fraction (0..1) of a fitted circle covered by a contour."""
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

    return 1.0 - largest_gap / (2.0 * np.pi)


def fit_ellipse_with_error(points):
    """
    Fit an ellipse and return a dimensionless mean geometric residual.

    OpenCV returns full axis lengths. The residual compares each point's
    normalized radius in ellipse coordinates with the ideal value 1.0.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 5:
        return None

    contour = points.astype(np.float32).reshape(-1, 1, 2)

    try:
        (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
    except cv2.error:
        return None

    if axis_a <= 0 or axis_b <= 0:
        return None

    semi_a = axis_a / 2.0
    semi_b = axis_b / 2.0

    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy

    # Rotate into the ellipse coordinate frame.
    local_x = cos_theta * dx + sin_theta * dy
    local_y = -sin_theta * dx + cos_theta * dy

    normalized_radius = np.sqrt(
        (local_x / semi_a) ** 2 + (local_y / semi_b) ** 2
    )
    relative_error = float(np.mean(np.abs(normalized_radius - 1.0)))

    return (
        float(cx),
        float(cy),
        float(axis_a),
        float(axis_b),
        float(angle),
        relative_error,
    )


def ellipse_angular_coverage(points, cx, cy, axis_a, axis_b, angle):
    """Return the approximate fraction (0..1) of a fitted ellipse covered."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or axis_a <= 0 or axis_b <= 0:
        return 0.0

    semi_a = axis_a / 2.0
    semi_b = axis_b / 2.0

    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    local_x = cos_theta * dx + sin_theta * dy
    local_y = -sin_theta * dx + cos_theta * dy

    # Parameter angle for an ideal ellipse:
    # x = a*cos(t), y = b*sin(t)
    angles = np.mod(
        np.arctan2(local_y / semi_b, local_x / semi_a),
        2.0 * np.pi,
    )
    angles = np.sort(angles)

    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
    largest_gap = float(np.max(np.append(gaps, wrap_gap)))

    return 1.0 - largest_gap / (2.0 * np.pi)


def try_circle_with_smoothing(
    points,
    max_relative_error,
    min_radius,
    max_radius,
    min_coverage,
    max_smoothing_passes,
):
    """Progressively smooth a contour until it meets the circle error limit."""
    for smoothing_passes in range(max_smoothing_passes + 1):
        smoothed = smooth_contour(points, smoothing_passes)
        fitted = fit_circle_least_squares(smoothed)
        if fitted is None:
            continue

        cx, cy, radius, mean_error = fitted
        if radius < min_radius or radius > max_radius:
            continue

        relative_error = mean_error / radius
        coverage = circle_angular_coverage(smoothed, cx, cy)

        if relative_error <= max_relative_error and coverage >= min_coverage:
            return {
                "type": "circle",
                "center": (cx, cy),
                "radius": radius,
                "error": mean_error,
                "relative_error": relative_error,
                "coverage": coverage,
                "points": smoothed,
                "smoothing_passes": smoothing_passes,
            }

    return None


def try_ellipse_with_smoothing(
    points,
    max_relative_error,
    min_radius,
    max_radius,
    min_coverage,
    max_smoothing_passes,
):
    """Try an ellipse only after circle fitting has failed."""
    if len(points) < 5:
        return None

    for smoothing_passes in range(max_smoothing_passes + 1):
        smoothed = smooth_contour(points, smoothing_passes)
        fitted = fit_ellipse_with_error(smoothed)
        if fitted is None:
            continue

        cx, cy, axis_a, axis_b, angle, relative_error = fitted
        semi_major = max(axis_a, axis_b) / 2.0
        semi_minor = min(axis_a, axis_b) / 2.0

        # Apply the radius bounds to ellipse semi-axes. This rejects tiny
        # artifacts and fits implausibly larger than the image.
        if semi_minor < min_radius or semi_major > max_radius:
            continue

        coverage = ellipse_angular_coverage(
            smoothed,
            cx,
            cy,
            axis_a,
            axis_b,
            angle,
        )

        if relative_error <= max_relative_error and coverage >= min_coverage:
            return {
                "type": "ellipse",
                "center": (cx, cy),
                "axes": (axis_a, axis_b),
                "angle": angle,
                "relative_error": relative_error,
                "coverage": coverage,
                "points": smoothed,
                "smoothing_passes": smoothing_passes,
            }

    return None


def detections_are_duplicates(first, second):
    """Heuristic duplicate rejection for nested/parallel contour edges."""
    fx, fy = first["center"]
    sx, sy = second["center"]

    if first["type"] == "circle":
        first_scale = first["radius"]
    else:
        first_scale = max(first["axes"]) / 2.0

    if second["type"] == "circle":
        second_scale = second["radius"]
    else:
        second_scale = max(second["axes"]) / 2.0

    scale = max(first_scale, second_scale)
    if scale <= 0:
        return False

    if math.hypot(fx - sx, fy - sy) >= 0.15 * scale:
        return False

    if first["type"] == "circle" and second["type"] == "circle":
        return abs(first["radius"] - second["radius"]) < 0.15 * scale

    if first["type"] == "ellipse" and second["type"] == "ellipse":
        first_axes = sorted(first["axes"])
        second_axes = sorted(second["axes"])
        return (
            abs(first_axes[0] - second_axes[0])
            < 0.20 * max(first_axes[0], second_axes[0])
            and abs(first_axes[1] - second_axes[1])
            < 0.20 * max(first_axes[1], second_axes[1])
        )

    # A near-circular ellipse can duplicate a circle from the other side of
    # the same physical edge. Compare their effective radii.
    if first["type"] == "circle":
        circle = first
        ellipse = second
    else:
        circle = second
        ellipse = first

    axis_a, axis_b = ellipse["axes"]
    ellipse_radius = (axis_a + axis_b) / 4.0
    circularity = abs(axis_a - axis_b) / max(axis_a, axis_b)
    return (
        circularity < 0.20
        and abs(circle["radius"] - ellipse_radius) < 0.15 * scale
    )


def detect_round_objects(
    binary,
    max_objects=2,
    min_contour_points=20,
    min_radius=10.0,
    max_radius=None,
    max_relative_error=0.08,
    min_coverage=0.12,
    max_smoothing_passes=12,
):
    """
    Detect circles/arcs first, then fall back to ellipses.

    Each contour is progressively smoothed. The first smoothing level whose
    circle residual satisfies max_relative_error is accepted. If no circle
    fit succeeds, ellipse fitting is attempted with the same smoothing
    progression.
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

    for contour in contours:
        if len(contour) < min_contour_points:
            continue

        points = contour.reshape(-1, 2)

        candidate = try_circle_with_smoothing(
            points,
            max_relative_error=max_relative_error,
            min_radius=min_radius,
            max_radius=max_radius,
            min_coverage=min_coverage,
            max_smoothing_passes=max_smoothing_passes,
        )

        if candidate is None:
            candidate = try_ellipse_with_smoothing(
                points,
                max_relative_error=max_relative_error,
                min_radius=min_radius,
                max_radius=max_radius,
                min_coverage=min_coverage,
                max_smoothing_passes=max_smoothing_passes,
            )

        if candidate is None:
            continue

        candidate["score"] = (
            candidate["coverage"]
            * len(candidate["points"])
            / (candidate["relative_error"] + 0.001)
        )
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

    selected = []
    for candidate in candidates:
        duplicate = any(
            detections_are_duplicates(candidate, existing)
            for existing in selected
        )
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
    max_smoothing_passes=12,
):
    """Threshold an image, detect round objects, and overlay full fits."""
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

    detections = detect_round_objects(
        binary_for_detection,
        max_objects=2,
        min_contour_points=20,
        min_radius=min_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
        max_smoothing_passes=max_smoothing_passes,
    )

    output = image.copy()

    for index, detection in enumerate(detections, start=1):
        cx, cy = detection["center"]
        center = (int(round(cx)), int(round(cy)))

        # OpenCV uses BGR; (0, 0, 255) is red.
        if detection["type"] == "circle":
            radius = detection["radius"]
            cv2.circle(
                output,
                center,
                int(round(radius)),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            geometry = f"circle r={radius:.1f}"
        else:
            axis_a, axis_b = detection["axes"]
            ellipse = (
                center,
                (max(1, int(round(axis_a))), max(1, int(round(axis_b)))),
                detection["angle"],
            )
            cv2.ellipse(output, ellipse, (0, 0, 255), 2, cv2.LINE_AA)
            geometry = f"ellipse {axis_a:.1f}x{axis_b:.1f}"

        cv2.circle(output, center, 3, (0, 0, 255), -1)

        coverage_degrees = detection["coverage"] * 360.0
        label = (
            f"{index}: {geometry} "
            f"arc={coverage_degrees:.0f}deg "
            f"err={detection['relative_error']:.3f} "
            f"smooth={detection['smoothing_passes']}"
        )
        cv2.putText(
            output,
            label,
            (center[0] + 10, center[1] - 10),
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
            "Threshold an image and detect up to two round objects. "
            "Contours are progressively smoothed and fitted as circles first; "
            "failed circle fits fall back to ellipses."
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
        help="Minimum fitted circle radius / ellipse semi-axis in pixels (default: 10)",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=0.08,
        help="Initial maximum relative fitting error (default: 0.08)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.12,
        help="Minimum visible circle/ellipse fraction, 0..1 (default: 0.12)",
    )
    parser.add_argument(
        "--max-smoothing-passes",
        type=int,
        default=12,
        help="Maximum contour smoothing passes before rejecting a fit (default: 12)",
    )
    parser.add_argument(
        "--morphology",
        action="store_true",
        help="Apply a 3x3 morphological opening before contour detection",
    )
    parser.add_argument(
        "--output",
        default="detected_round_objects.png",
        help="Output image written when S is pressed",
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be greater than zero")
    if not 0.001 <= args.max_error <= 0.5:
        parser.error("--max-error must be between 0.001 and 0.5")
    if not 0.0 <= args.min_coverage <= 1.0:
        parser.error("--min-coverage must be between 0 and 1")
    if args.max_smoothing_passes < 0:
        parser.error("--max-smoothing-passes cannot be negative")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    window_name = "Circle / Ellipse Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(
        "Threshold",
        window_name,
        args.threshold,
        255,
        lambda _value: None,
    )

    # OpenCV trackbars are integer-valued. Represent relative error in
    # thousandths so 80 means 0.080, 25 means 0.025, etc.
    initial_error_slider = max(1, min(500, int(round(args.max_error * 1000.0))))
    cv2.createTrackbar(
        "Max error x1000",
        window_name,
        initial_error_slider,
        500,
        lambda _value: None,
    )

    while True:
        threshold_value = cv2.getTrackbarPos("Threshold", window_name)
        error_slider = cv2.getTrackbarPos("Max error x1000", window_name)
        max_relative_error = max(1, error_slider) / 1000.0

        _, binary_for_detection, result, detections = process_image(
            image,
            threshold_value,
            invert=args.invert,
            min_radius=args.min_radius,
            max_relative_error=max_relative_error,
            min_coverage=args.min_coverage,
            morphology=args.morphology,
            max_smoothing_passes=args.max_smoothing_passes,
        )

        threshold_preview = cv2.cvtColor(
            binary_for_detection,
            cv2.COLOR_GRAY2BGR,
        )

        status = (
            f"threshold={threshold_value}  "
            f"max_relative_error={max_relative_error:.3f}"
        )
        cv2.putText(
            threshold_preview,
            status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
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
            print(
                f"threshold={threshold_value}, "
                f"max_relative_error={max_relative_error:.3f}"
            )

            for index, detection in enumerate(detections, start=1):
                cx, cy = detection["center"]
                if detection["type"] == "circle":
                    geometry = f"radius={detection['radius']:.2f}"
                else:
                    axis_a, axis_b = detection["axes"]
                    geometry = (
                        f"axes=({axis_a:.2f}, {axis_b:.2f}), "
                        f"angle={detection['angle']:.2f}"
                    )

                print(
                    f"Object {index}: "
                    f"type={detection['type']}, "
                    f"center=({cx:.2f}, {cy:.2f}), "
                    f"{geometry}, "
                    f"coverage={detection['coverage'] * 360.0:.1f} deg, "
                    f"relative_error={detection['relative_error']:.4f}, "
                    f"smoothing_passes={detection['smoothing_passes']}"
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
