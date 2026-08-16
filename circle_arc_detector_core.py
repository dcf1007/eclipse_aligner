import math

import cv2
import numpy as np


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


def _downsample_contour_points(points, max_search_points):
    point_count = len(points)
    if max_search_points <= 0 or point_count <= max_search_points:
        return points

    indices = np.linspace(0, point_count - 1, max_search_points, dtype=np.int32)
    return points[indices]


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
