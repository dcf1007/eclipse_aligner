#!/usr/bin/env python3
"""Threshold an image and reconstruct up to two circular objects from contours.

The output is the thresholded black/white image converted to BGR so the inferred
complete circles can be drawn in red.

Examples:
    python eclipse_aligner.py input.jpg --threshold 120
    python eclipse_aligner.py input.jpg --threshold 120 --invert
    python eclipse_aligner.py input.jpg --interactive
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class CircleCandidate:
    center_x: float
    center_y: float
    radius: float
    rms_error: float
    relative_error: float
    angular_coverage: float
    contour_length: float
    score: float


def fit_circle_least_squares(points: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit a circle to Nx2 points and return (cx, cy, radius, RMS radial error)."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None

    # Centering the coordinates improves numerical conditioning on large images.
    origin = points.mean(axis=0)
    q = points - origin
    x = q[:, 0]
    y = q[:, 1]

    # x^2 + y^2 = 2*cx*x + 2*cy*y + c
    a = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
    b = x * x + y * y

    try:
        solution, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if rank < 3:
        return None

    local_cx, local_cy, c = solution
    radius_sq = c + local_cx * local_cx + local_cy * local_cy
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        return None

    radius = float(math.sqrt(radius_sq))
    cx = float(local_cx + origin[0])
    cy = float(local_cy + origin[1])

    distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    rms_error = float(np.sqrt(np.mean((distances - radius) ** 2)))

    if not all(np.isfinite(v) for v in (cx, cy, radius, rms_error)):
        return None

    return cx, cy, radius, rms_error


def angular_coverage(points: np.ndarray, cx: float, cy: float) -> float:
    """Estimate the fraction of the 360-degree circle represented by the contour."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return 0.0

    angles = np.mod(
        np.arctan2(points[:, 1] - cy, points[:, 0] - cx),
        2.0 * np.pi,
    )
    angles.sort()

    gaps = np.diff(angles)
    wrap_gap = angles[0] + 2.0 * np.pi - angles[-1]
    largest_gap = float(max(np.max(gaps, initial=0.0), wrap_gap))
    return max(0.0, min(1.0, 1.0 - largest_gap / (2.0 * np.pi)))


def is_duplicate(a: CircleCandidate, b: CircleCandidate) -> bool:
    """Return True when two candidates are effectively the same fitted circle."""
    scale = max(a.radius, b.radius, 1.0)
    center_distance = math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)
    radius_difference = abs(a.radius - b.radius)
    return center_distance <= 0.08 * scale and radius_difference <= 0.08 * scale


def detect_circular_contours(
    binary: np.ndarray,
    *,
    max_objects: int = 2,
    min_radius: float = 8.0,
    max_radius: float | None = None,
    min_contour_length: float = 30.0,
    max_relative_error: float = 0.08,
    min_coverage: float = 0.10,
) -> list[CircleCandidate]:
    """Fit circles to contours and return the best one or two circular candidates."""
    height, width = binary.shape[:2]
    if max_radius is None:
        max_radius = math.hypot(width, height)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[CircleCandidate] = []

    for contour in contours:
        contour_length = float(cv2.arcLength(contour, closed=True))
        if contour_length < min_contour_length or len(contour) < 6:
            continue

        points = contour.reshape(-1, 2).astype(np.float64)
        fitted = fit_circle_least_squares(points)
        if fitted is None:
            continue

        cx, cy, radius, rms_error = fitted
        if radius < min_radius or radius > max_radius:
            continue

        relative_error = rms_error / max(radius, 1e-9)
        if relative_error > max_relative_error:
            continue

        coverage = angular_coverage(points, cx, cy)
        if coverage < min_coverage:
            continue

        # Prefer long, well-covered contours with a low normalized radial error.
        score = contour_length * (0.25 + coverage) / (0.005 + relative_error)
        candidates.append(
            CircleCandidate(
                center_x=cx,
                center_y=cy,
                radius=radius,
                rms_error=rms_error,
                relative_error=relative_error,
                angular_coverage=coverage,
                contour_length=contour_length,
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    selected: list[CircleCandidate] = []
    for candidate in candidates:
        if any(is_duplicate(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max(1, min(max_objects, 2)):
            break

    return selected


def threshold_image(gray: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, int(threshold), 255, mode)
    return binary


def render_result(binary: np.ndarray, circles: Iterable[CircleCandidate]) -> np.ndarray:
    """Render the B/W image with complete inferred circles in red."""
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for circle in circles:
        center = (int(round(circle.center_x)), int(round(circle.center_y)))
        radius = int(round(circle.radius))
        cv2.circle(output, center, radius, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(output, center, 2, (0, 0, 255), -1, cv2.LINE_AA)

    return output


def process(
    image: np.ndarray,
    *,
    threshold: int,
    invert: bool,
    min_radius: float,
    max_radius: float | None,
    min_contour_length: float,
    max_relative_error: float,
    min_coverage: float,
) -> tuple[np.ndarray, list[CircleCandidate]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = threshold_image(gray, threshold, invert)
    circles = detect_circular_contours(
        binary,
        max_objects=2,
        min_radius=min_radius,
        max_radius=max_radius,
        min_contour_length=min_contour_length,
        max_relative_error=max_relative_error,
        min_coverage=min_coverage,
    )
    return render_result(binary, circles), circles


def print_detections(circles: list[CircleCandidate], threshold: int) -> None:
    print(f"threshold={threshold}; detected={len(circles)}")
    for index, circle in enumerate(circles, start=1):
        print(
            f"circle {index}: "
            f"center=({circle.center_x:.2f}, {circle.center_y:.2f}), "
            f"radius={circle.radius:.2f}, "
            f"coverage={circle.angular_coverage * 360.0:.1f} deg, "
            f"relative_error={circle.relative_error:.4f}"
        )


def run_interactive(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[CircleCandidate], int]:
    window = "eclipse_aligner"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Threshold", window, args.threshold, 255, lambda _value: None)

    last_result: np.ndarray | None = None
    last_circles: list[CircleCandidate] = []
    last_threshold = args.threshold

    while True:
        threshold = cv2.getTrackbarPos("Threshold", window)
        result, circles = process(
            image,
            threshold=threshold,
            invert=args.invert,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_contour_length=args.min_contour_length,
            max_relative_error=args.max_relative_error,
            min_coverage=args.min_coverage,
        )
        cv2.imshow(window, result)

        last_result = result
        last_circles = circles
        last_threshold = threshold

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            cv2.imwrite(str(args.output), result)
            print_detections(circles, threshold)
            print(f"saved: {args.output}")

    cv2.destroyAllWindows()
    assert last_result is not None
    return last_result, last_circles, last_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold an image to black/white, detect up to two circles or circular "
            "arcs from contours, and draw each inferred full circle in red."
        )
    )
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("-o", "--output", type=Path, default=Path("aligned.png"))
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=128,
        help="black/white threshold, 0-255 (default: 128)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="use THRESH_BINARY_INV; useful for dark objects on a bright background",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="open a threshold slider; press S to save, Q/Esc to quit",
    )
    parser.add_argument("--min-radius", type=float, default=8.0)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--min-contour-length", type=float, default=30.0)
    parser.add_argument(
        "--max-relative-error",
        type=float,
        default=0.08,
        help="maximum RMS radial fit error divided by radius (default: 0.08)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.10,
        help="minimum visible fraction of the inferred circle, 0-1 (default: 0.10)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be > 0")
    if args.max_radius is not None and args.max_radius <= args.min_radius:
        parser.error("--max-radius must be greater than --min-radius")
    if args.max_relative_error <= 0:
        parser.error("--max-relative-error must be > 0")
    if not 0 < args.min_coverage <= 1:
        parser.error("--min-coverage must be in (0, 1]")

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"could not read image: {args.image}")

    if args.interactive:
        result, circles, threshold = run_interactive(image, args)
        # Save the final state on exit as well, so interactive mode is deterministic.
        cv2.imwrite(str(args.output), result)
    else:
        result, circles = process(
            image,
            threshold=args.threshold,
            invert=args.invert,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
            min_contour_length=args.min_contour_length,
            max_relative_error=args.max_relative_error,
            min_coverage=args.min_coverage,
        )
        threshold = args.threshold
        if not cv2.imwrite(str(args.output), result):
            raise RuntimeError(f"failed to write output image: {args.output}")

    print_detections(circles, threshold)
    print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
