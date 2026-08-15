#!/usr/bin/env python3
"""Threshold an image and reconstruct up to two circular objects from contours.

The detector accepts complete circles, partial circles, and circular arcs. It
fits a circle to each contour, ranks plausible circular contours, suppresses
duplicate detections, and draws the complete inferred circle in red.

Dependencies:
    pip install opencv-python numpy

Examples:
    python eclipse_aligner.py image.jpg
    python eclipse_aligner.py image.jpg --threshold 150 --invert
    python eclipse_aligner.py image.jpg --no-gui --output result.png

GUI controls:
    Threshold slider  Adjust the grayscale threshold (0-255)
    I                 Toggle threshold inversion
    S                 Save the current annotated result
    Q / Esc           Quit
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CircleDetection:
    center_x: float
    center_y: float
    radius: float
    mean_error: float
    relative_error: float
    coverage: float
    contour_points: int
    score: float

    @property
    def coverage_degrees(self) -> float:
        return self.coverage * 360.0


def fit_circle_least_squares(points: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """Fit a circle to Nx2 points using algebraic least squares.

    Returns (center_x, center_y, radius, mean_absolute_radial_error), or None
    when a stable circle cannot be fitted.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]

    # x^2 + y^2 = 2*cx*x + 2*cy*y + c
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y

    try:
        solution, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        return None

    cx, cy, c = solution
    radius_squared = c + cx * cx + cy * cy
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        return None

    radius = math.sqrt(radius_squared)
    distances = np.hypot(x - cx, y - cy)
    mean_error = float(np.mean(np.abs(distances - radius)))

    if not all(np.isfinite(v) for v in (cx, cy, radius, mean_error)):
        return None

    return float(cx), float(cy), float(radius), mean_error


def refine_circle_fit(
    points: np.ndarray,
    iterations: int = 3,
    sigma: float = 2.5,
) -> Optional[tuple[float, float, float, float, np.ndarray]]:
    """Iteratively reject radial outliers and refit the circle.

    This makes fitting less sensitive to small contour defects while retaining
    the original contour-based approach.
    """
    source = np.asarray(points, dtype=np.float64)
    working = source

    for _ in range(max(1, iterations)):
        fit = fit_circle_least_squares(working)
        if fit is None:
            return None

        cx, cy, radius, _ = fit
        residuals = np.abs(np.hypot(source[:, 0] - cx, source[:, 1] - cy) - radius)

        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        robust_sigma = 1.4826 * mad

        if robust_sigma < 1e-9:
            break

        limit = median + sigma * robust_sigma
        inliers = source[residuals <= limit]
        if len(inliers) < 3 or len(inliers) == len(working):
            working = inliers if len(inliers) >= 3 else working
            break
        working = inliers

    fit = fit_circle_least_squares(working)
    if fit is None:
        return None

    cx, cy, radius, _ = fit
    inlier_errors = np.abs(np.hypot(working[:, 0] - cx, working[:, 1] - cy) - radius)
    mean_error = float(np.mean(inlier_errors))
    return cx, cy, radius, mean_error, working


def angular_coverage(points: np.ndarray, center_x: float, center_y: float) -> float:
    """Estimate what fraction of the fitted circle is represented by points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    angles = np.mod(
        np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x),
        2.0 * np.pi,
    )
    angles.sort()

    gaps = np.diff(angles)
    wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
    largest_gap = float(np.max(np.append(gaps, wrap_gap)))
    coverage = 1.0 - largest_gap / (2.0 * np.pi)
    return float(np.clip(coverage, 0.0, 1.0))


def detect_circular_objects(
    binary: np.ndarray,
    max_objects: int = 2,
    min_contour_points: int = 20,
    min_radius: float = 10.0,
    max_radius: Optional[float] = None,
    max_relative_error: float = 0.08,
    min_coverage: float = 0.12,
) -> list[CircleDetection]:
    """Detect and rank up to ``max_objects`` circular contours/arcs."""
    height, width = binary.shape[:2]
    if max_radius is None:
        max_radius = float(math.hypot(width, height))

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[CircleDetection] = []

    for contour in contours:
        if len(contour) < min_contour_points:
            continue

        points = contour.reshape(-1, 2).astype(np.float64)
        refined = refine_circle_fit(points)
        if refined is None:
            continue

        cx, cy, radius, mean_error, inliers = refined
        if radius < min_radius or radius > max_radius:
            continue

        relative_error = mean_error / radius
        if relative_error > max_relative_error:
            continue

        coverage = angular_coverage(inliers, cx, cy)
        if coverage < min_coverage:
            continue

        # Require enough inliers so a few coincidental points do not dominate.
        inlier_fraction = len(inliers) / len(points)
        if inlier_fraction < 0.60:
            continue

        # Coverage strongly rewards meaningful arcs/full circles. Contour size
        # provides a secondary preference, while radial error penalizes shapes
        # that only approximately resemble circles.
        score = (
            (coverage**1.5)
            * math.sqrt(float(len(inliers)))
            * inlier_fraction
            / (relative_error + 0.002)
        )

        candidates.append(
            CircleDetection(
                center_x=cx,
                center_y=cy,
                radius=radius,
                mean_error=mean_error,
                relative_error=relative_error,
                coverage=coverage,
                contour_points=len(points),
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)

    selected: list[CircleDetection] = []
    for candidate in candidates:
        duplicate = False
        for existing in selected:
            center_distance = math.hypot(
                candidate.center_x - existing.center_x,
                candidate.center_y - existing.center_y,
            )
            scale = max(candidate.radius, existing.radius)
            if (
                center_distance < 0.15 * scale
                and abs(candidate.radius - existing.radius) < 0.15 * scale
            ):
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(candidate)
        if len(selected) >= max_objects:
            break

    return selected


def threshold_image(image: np.ndarray, threshold_value: int, invert: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, threshold_value, 255, mode)
    return binary


def process_image(
    image: np.ndarray,
    threshold_value: int,
    invert: bool = False,
    min_radius: float = 10.0,
    max_radius: Optional[float] = None,
    max_relative_error: float = 0.08,
    min_coverage: float = 0.12,
    min_contour_points: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[CircleDetection]]:
    """Threshold, detect up to two circles/arcs, and draw full red circles."""
    binary = threshold_image(image, threshold_value, invert)

    detections = detect_circular_objects(
        binary,
        max_objects=2,
        min_contour_points=min_contour_points,
        min_radius=min_radius,
        max_radius=max_radius,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
    )

    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        center = (
            int(round(detection.center_x)),
            int(round(detection.center_y)),
        )
        radius = int(round(detection.radius))

        # OpenCV uses BGR ordering: (0, 0, 255) is red.
        cv2.circle(output, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 3, (0, 0, 255), -1, cv2.LINE_AA)

        label = (
            f"{index}: r={detection.radius:.1f} "
            f"arc={detection.coverage_degrees:.0f}deg "
            f"err={detection.relative_error:.3f}"
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


def print_detections(detections: list[CircleDetection]) -> None:
    if not detections:
        print("No circular objects detected.")
        return

    for index, detection in enumerate(detections, start=1):
        print(
            f"Object {index}: "
            f"center=({detection.center_x:.2f}, {detection.center_y:.2f}), "
            f"radius={detection.radius:.2f}, "
            f"coverage={detection.coverage_degrees:.1f} deg, "
            f"relative_error={detection.relative_error:.4f}"
        )


def save_result(path: Path, output: np.ndarray, detections: list[CircleDetection]) -> None:
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Could not save output image: {path}")
    print(f"Saved: {path}")
    print_detections(detections)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image, detect up to two circular contours/arcs, "
            "and reconstruct each complete circle in red."
        )
    )
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Initial grayscale threshold, 0-255 (default: 128)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert threshold polarity (useful for dark objects on light backgrounds)",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=10.0,
        help="Minimum accepted fitted radius in pixels (default: 10)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="Maximum accepted fitted radius in pixels",
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
        help="Minimum visible fraction of the fitted circle, 0-1 (default: 0.12)",
    )
    parser.add_argument(
        "--min-contour-points",
        type=int,
        default=20,
        help="Minimum contour point count (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("detected_circles.png"),
        help="Annotated output path (default: detected_circles.png)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Process once with --threshold, save, and exit without opening a window",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be greater than 0")
    if args.max_radius is not None and args.max_radius <= args.min_radius:
        parser.error("--max-radius must be greater than --min-radius")
    if args.max_error <= 0:
        parser.error("--max-error must be greater than 0")
    if not 0.0 < args.min_coverage <= 1.0:
        parser.error("--min-coverage must be in the interval (0, 1]")
    if args.min_contour_points < 3:
        parser.error("--min-contour-points must be at least 3")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"could not load image: {args.image}")

    def run(threshold_value: int, invert: bool):
        return process_image(
            image,
            threshold_value,
            invert=invert,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            max_relative_error=args.max_error,
            min_coverage=args.min_coverage,
            min_contour_points=args.min_contour_points,
        )

    if args.no_gui:
        _, output, detections = run(args.threshold, args.invert)
        save_result(args.output, output, detections)
        return 0

    window_name = "Eclipse Aligner - threshold | reconstructed circles"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window_name, args.threshold, 255, lambda _: None)
    invert = args.invert

    while True:
        threshold_value = cv2.getTrackbarPos("Threshold", window_name)
        binary, output, detections = run(threshold_value, invert)

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        display = np.hstack((binary_bgr, output))

        polarity = "INVERTED" if invert else "NORMAL"
        cv2.putText(
            display,
            f"threshold={threshold_value} polarity={polarity} | I=invert S=save Q/Esc=quit",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("i"), ord("I")):
            invert = not invert
        if key in (ord("s"), ord("S")):
            save_result(args.output, output, detections)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
