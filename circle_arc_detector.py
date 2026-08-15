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


def angular_coverage(points, cx, cy):
    """Return the fraction (0..1) of the fitted circle covered by a contour."""
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


def detect_circular_objects(
    binary,
    max_objects=2,
    min_contour_points=20,
    min_radius=10.0,
    max_radius=None,
    max_relative_error=0.08,
    min_coverage=0.12,
):
    """Detect up to max_objects contours compatible with circles or arcs."""
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
        fitted = fit_circle_least_squares(points)
        if fitted is None:
            continue

        cx, cy, radius, mean_error = fitted
        if radius < min_radius or radius > max_radius:
            continue

        relative_error = mean_error / radius
        if relative_error > max_relative_error:
            continue

        coverage = angular_coverage(points, cx, cy)
        if coverage < min_coverage:
            continue

        # Prefer long, well-covered contours with low radial fitting error.
        score = coverage * len(points) / (relative_error + 0.001)

        candidates.append(
            {
                "center": (cx, cy),
                "radius": radius,
                "error": mean_error,
                "relative_error": relative_error,
                "coverage": coverage,
                "points": points,
                "score": score,
            }
        )

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
    """Threshold an image, detect circular objects, and overlay full circles."""
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

        # OpenCV uses BGR; (0, 0, 255) is red.
        cv2.circle(output, center, radius_int, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1)

        coverage_degrees = detection["coverage"] * 360.0
        label = (
            f"{index}: r={radius:.1f} "
            f"arc={coverage_degrees:.0f}deg "
            f"err={detection['relative_error']:.3f}"
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
            "Threshold an image and detect up to two circular objects from "
            "full circles, partial circles, or arcs."
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
        help="Minimum visible circle fraction, 0..1 (default: 0.12)",
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
                    f"relative_error={detection['relative_error']:.4f}"
                )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
