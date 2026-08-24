"""Interactive eclipse limb detector with fast preview and full-resolution modes.

The detector is intentionally self-contained and uses only Tkinter, OpenCV and
NumPy.  It keeps the threshold semantics explicit and deterministic:

    dark mask  = grayscale <= threshold
    light mask = grayscale > threshold

The GUI supports an ordered image list, per-image settings, a fast reduced-size
preview pass, and an authoritative full-resolution pass.  Each image receives an
automatically selected initial threshold the first time it is encountered; that
threshold is then stored explicitly in the per-image dictionary so subsequent
navigation never silently re-estimates it.

Detection overview
------------------
1. Threshold the image into complementary dark/light masks.
2. Optionally clean a *guidance copy* of each mask with morphology.  The original
   masks remain authoritative for polarity, edge location, horizon validation,
   arc support, and final ellipse acceptance.
3. Search contour windows with multiple OpenCV ellipse fitters plus a circular
   stabilizing prior.  Intermediate fits may temporarily exceed the user's final
   semi-axis limits so short arcs are not rejected before they can be grown.
4. Optionally add outer-limb seeds using actual contour points that lie on the
   convex envelope.  Artificial convex-hull chords are never used as fit points.
5. Detect a horizon whenever a strong straight light/dark boundary cuts the
   provisional light/Sun ellipse.  Horizon orientation is unrestricted.  A
   horizon is accepted only when one half-plane is predominantly dark and the
   opposite half-plane contains the visible solar region.
6. Grow each ellipse around all 360 degrees and retain *multiple disconnected
   real support segments*.  Gaps remain gaps and do not count toward coverage.
   When a horizon exists, every retained arc must lie on the visible side; the
   dark/occluded half-plane contributes no ellipse support.
7. Refit to all recovered support points and enforce the exact final semi-axis
   limits, normalized residual limit, boundary polarity, and class logic.
8. Render the threshold preview with magenta support arcs, blue dark ellipse,
   golden light ellipse, and a green horizon line when one was validated.
9. Render a full-color image translated by integer pixels so the detected light
   ellipse is centered.  No interpolation is used for that centering operation.

Dependencies:
    pip install opencv-python numpy
"""

import argparse
import base64
import math
import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detector tuning constants
# ---------------------------------------------------------------------------
MIN_BOUNDARY_POLARITY = 0.42
MAX_CLASS_CANDIDATES = 30
MAX_REGIONS_PER_CONTOUR = 4
EDGE_SEARCH_FRACTION = 0.15
ARC_GROW_SAMPLES = 720
DARK_DOMINANT_COVERAGE = 0.55

# Intermediate ellipse hypotheses are deliberately allowed outside the strict
# final user range.  Final candidates must still pass the exact selected limits.
INTERMEDIATE_MIN_FACTOR = 0.65
INTERMEDIATE_MAX_FACTOR = 1.35

# Reduced-resolution working sizes.  Auto-threshold selection uses an even
# smaller image because it may evaluate several candidate thresholds.
PREVIEW_MAX_DIM = 2400
AUTO_THRESHOLD_MAX_DIM = 600

# Horizon detection is intentionally conservative.  Values are relative to the
# provisional solar ellipse so the test remains meaningful across image scales.
HORIZON_MIN_SPAN_RADIUS = 0.55
HORIZON_MAX_RESIDUAL_RADIUS = 0.008
HORIZON_MAX_CENTER_DISTANCE_RADIUS = 0.95
HORIZON_DARK_SIDE_MAX_LIGHT = 0.08
HORIZON_VISIBLE_SIDE_MIN_LIGHT = 0.12
HORIZON_MIN_SIDE_CONTRAST = 0.10
HORIZON_EXCLUSION_RADIUS = 0.012

# Small morphology is guidance-only.  It can help contour discovery but can
# never redefine the final measured limb.
MORPH_KERNEL_SIZE = 5

# Coarse contour-window refinement moves: expand/shrink either side or slide.
REFINE_MOVES = (
    (-1, 1),
    (0, 1),
    (1, -1),
    (0, -1),
    (-1, 0),
    (1, 0),
)

# OpenCV BGR colors used only for preview annotation.
ARC_COLOR = (255, 0, 255)              # magenta
DARK_ELLIPSE_COLOR = (255, 0, 0)      # blue
LIGHT_ELLIPSE_COLOR = (0, 190, 255)   # golden/orange yellow
HORIZON_COLOR = (0, 255, 0)           # green
ELLIPSE_LINE_THICKNESS = 3
ARC_LINE_THICKNESS = 2
HORIZON_LINE_THICKNESS = 3

# Threshold is special: once an image is initialized it is ALWAYS present in the
# per-image dictionary.  The remaining settings are stored only when they differ
# from their session defaults.
SETTING_NAMES = (
    "threshold",
    "min_radius",
    "max_radius",
    "max_error",
    "min_coverage",
    "morphology",
    "outer_limb_assistance",
)
NON_THRESHOLD_SETTING_NAMES = SETTING_NAMES[1:]

IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)


# ---------------------------------------------------------------------------
# Basic ellipse geometry and fitting
# ---------------------------------------------------------------------------
def normalize_ellipse(raw):
    """Normalize OpenCV's ellipse tuple to center, semi-axes and major angle."""
    try:
        (cx, cy), (width, height), angle = raw
    except (TypeError, ValueError):
        return None

    values = (cx, cy, width, height, angle)
    if not np.all(np.isfinite(values)) or width <= 0 or height <= 0:
        return None

    semi_x = width * 0.5
    semi_y = height * 0.5
    if semi_x >= semi_y:
        major, minor, major_angle = semi_x, semi_y, angle
    else:
        major, minor, major_angle = semi_y, semi_x, angle + 90.0

    major_angle %= 180.0
    equivalent_radius = math.sqrt(major * minor)
    if equivalent_radius <= 0 or not np.isfinite(equivalent_radius):
        return None

    return {
        "center": (float(cx), float(cy)),
        "major": float(major),
        "minor": float(minor),
        "angle": float(major_angle),
        "equivalent_radius": float(equivalent_radius),
    }


def axes_in_range(ellipse, min_radius, max_radius):
    """Strict FINAL constraint: both semi-axis radii must be inside the range."""
    return (
        min_radius <= ellipse["major"] <= max_radius
        and min_radius <= ellipse["minor"] <= max_radius
    )


def axes_in_intermediate_range(ellipse, min_radius, max_radius):
    """Loose search envelope used before multi-segment support is recovered."""
    search_min = max(1.0, min_radius * INTERMEDIATE_MIN_FACTOR)
    search_max = max(max_radius, min_radius) * INTERMEDIATE_MAX_FACTOR
    return (
        search_min <= ellipse["major"] <= search_max
        and search_min <= ellipse["minor"] <= search_max
    )


def fit_circle_prior(points):
    """Least-squares circle exposed as a zero-eccentricity ellipse candidate."""
    points = np.asarray(points, np.float64)
    if len(points) < 3:
        return None

    x = points[:, 0]
    y = points[:, 1]
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    u = x - x_mean
    v = y - y_mean
    squared_distance = u * u + v * v

    suu = np.dot(u, u)
    svv = np.dot(v, v)
    suv = np.dot(u, v)
    suz = np.dot(u, squared_distance)
    svz = np.dot(v, squared_distance)

    determinant = suu * svv - suv * suv
    if abs(determinant) <= 1e-12 * (suu * svv + 1.0):
        return None

    u_center = 0.5 * (suz * svv - svz * suv) / determinant
    v_center = 0.5 * (svz * suu - suz * suv) / determinant
    cx = x_mean + u_center
    cy = y_mean + v_center
    radius_sq = float(squared_distance.mean()) + u_center**2 + v_center**2
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return None

    radius = math.sqrt(radius_sq)
    return {
        "center": (float(cx), float(cy)),
        "major": radius,
        "minor": radius,
        "angle": 0.0,
        "equivalent_radius": radius,
    }


def fit_ellipse_options(points):
    """Try Direct, AMS, standard OpenCV ellipse fits plus a circle prior."""
    points = np.asarray(points, np.float32)
    if len(points) < 5:
        return []

    shaped = points.reshape(-1, 1, 2)
    options = []
    for fitter in (cv2.fitEllipseDirect, cv2.fitEllipseAMS, cv2.fitEllipse):
        try:
            ellipse = normalize_ellipse(fitter(shaped))
        except cv2.error:
            ellipse = None
        if ellipse is not None:
            options.append(ellipse)

    circle = fit_circle_prior(points)
    if circle is not None:
        options.append(circle)
    return options


def ellipse_coordinates(points, ellipse):
    """Convert image points into normalized coordinates of a rotated ellipse."""
    points = np.asarray(points, np.float64)
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return local_x / ellipse["major"], local_y / ellipse["minor"]


def ellipse_error_and_coverage(points, ellipse):
    """Return mean normalized radial residual and simple angular span coverage."""
    if len(points) < 2:
        return math.inf, 0.0

    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    relative_error = float(np.mean(np.abs(np.hypot(x_norm, y_norm) - 1.0)))
    angles = np.unwrap(np.arctan2(y_norm, x_norm))
    coverage = float(np.clip(np.ptp(angles) / (2.0 * np.pi), 0.0, 1.0))
    return relative_error, coverage


def boundary_polarity(mask, ellipse):
    """Measure whether support points cross from mask-inside to mask-outside."""
    points = np.asarray(ellipse["points"], np.float64)
    if len(points) < 5:
        return 0.0, 0.0, 0

    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    theta = np.arctan2(y_norm, x_norm)
    radius = ellipse["equivalent_radius"]
    sample_distance = max(3.0, radius * 0.008)
    delta = sample_distance / max(radius, 1.0)

    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = ellipse["center"]

    def sample(scale):
        local_x = ellipse["major"] * scale * np.cos(theta)
        local_y = ellipse["minor"] * scale * np.sin(theta)
        x = np.rint(cx + cos_a * local_x - sin_a * local_y).astype(np.int32)
        y = np.rint(cy + sin_a * local_x + cos_a * local_y).astype(np.int32)
        return x, y

    xi, yi = sample(1.0 - delta)
    xo, yo = sample(1.0 + delta)
    height, width = mask.shape
    valid = (
        (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        & (xo >= 0) & (xo < width) & (yo >= 0) & (yo < height)
    )
    count = int(np.count_nonzero(valid))
    if count == 0:
        return 0.0, 0.0, 0

    inside = mask[yi[valid], xi[valid]] != 0
    outside = mask[yo[valid], xo[valid]] != 0
    return (
        float(np.mean(inside & ~outside)),
        float(np.mean(inside)),
        count,
    )


def angle_difference_180(a, b):
    difference = abs((a - b) % 180.0)
    return min(difference, 180.0 - difference)


def same_ellipse(first, second, center_fraction=0.12, axis_fraction=0.12,
                 angle_tolerance=15.0):
    """Scale-relative duplicate test for two ellipse hypotheses."""
    fx, fy = first["center"]
    sx, sy = second["center"]
    scale = max(first["equivalent_radius"], second["equivalent_radius"], 1.0)

    if math.hypot(fx - sx, fy - sy) >= center_fraction * scale:
        return False
    if abs(first["major"] - second["major"]) >= axis_fraction * scale:
        return False
    if abs(first["minor"] - second["minor"]) >= axis_fraction * scale:
        return False

    ecc1 = (first["major"] - first["minor"]) / max(first["major"], 1.0)
    ecc2 = (second["major"] - second["minor"]) / max(second["major"], 1.0)
    if max(ecc1, ecc2) < 0.05:
        return True
    return angle_difference_180(first["angle"], second["angle"]) < angle_tolerance


# ---------------------------------------------------------------------------
# Contour-window candidate discovery
# ---------------------------------------------------------------------------
def window_lengths(point_count, minimum):
    minimum = max(5, min(minimum, point_count))
    lengths = {minimum, point_count}
    length = minimum
    while length < point_count:
        lengths.add(length)
        length = min(point_count, max(length + 1, int(round(length * 1.45))))
    return sorted(lengths)


def evaluate_region(extended_points, point_count, start, length, original_mask,
                    min_radius, max_radius, max_error, min_coverage):
    """Fit one cyclic contour window using relaxed intermediate axis limits."""
    if length < 5 or length > point_count:
        return None

    start %= point_count
    region = extended_points[start:start + length]
    best = None

    for ellipse in fit_ellipse_options(region):
        if not axes_in_intermediate_range(ellipse, min_radius, max_radius):
            continue

        relative_error, coverage = ellipse_error_and_coverage(region, ellipse)
        # Intermediate residual can be somewhat looser; final growth/refit will
        # reapply the exact selected error threshold.
        if relative_error > max(max_error * 1.5, max_error + 0.01):
            continue
        if coverage < max(0.02, min_coverage * 0.65):
            continue

        candidate = {
            **ellipse,
            "relative_error": relative_error,
            "coverage": coverage,
            "points": region.copy(),
            "arc_segments": [region.copy()],
            "total_supported_coverage": coverage,
            "largest_segment_coverage": coverage,
            "segment_count": 1,
            "start": start,
            "length": length,
        }

        polarity, inside_fraction, sample_count = boundary_polarity(
            original_mask, candidate
        )
        if (
            sample_count < 5
            or polarity < MIN_BOUNDARY_POLARITY
            or inside_fraction < 0.5
        ):
            continue

        support = length / point_count
        axis_ratio = candidate["major"] / max(candidate["minor"], 1.0)
        shape_penalty = 1.0 + 0.06 * max(0.0, axis_ratio - 1.0)
        geometry_score = (
            coverage**1.5
            * math.sqrt(length)
            * (0.75 + 0.25 * math.sqrt(support))
            / ((relative_error + 0.002) * shape_penalty)
        )
        candidate["boundary_polarity"] = polarity
        candidate["interior_fraction"] = inside_fraction
        candidate["score"] = geometry_score * polarity**2

        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def refine_region(candidate, extended_points, point_count, minimum_length,
                  original_mask, min_radius, max_radius, max_error, min_coverage):
    best = candidate
    for _ in range(40):
        improved = False
        for start_delta, length_delta in REFINE_MOVES:
            new_length = best["length"] + length_delta
            if not minimum_length <= new_length <= point_count:
                continue
            trial = evaluate_region(
                extended_points,
                point_count,
                best["start"] + start_delta,
                new_length,
                original_mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1 + 1e-9):
                best = trial
                improved = True
        if not improved:
            break
    return best


def find_regions(points, original_mask, min_radius, max_radius, max_error,
                 min_coverage, minimum=12):
    points = np.asarray(points, np.float64)
    point_count = len(points)
    if point_count < max(5, minimum):
        return []

    minimum = min(minimum, point_count)
    extended = np.vstack((points, points))
    coarse = []

    for length in window_lengths(point_count, minimum):
        step = max(1, length // 5)
        for start in range(0, point_count, step):
            candidate = evaluate_region(
                extended,
                point_count,
                start,
                length,
                original_mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if candidate is not None:
                coarse.append(candidate)

    coarse.sort(key=lambda item: item["score"], reverse=True)
    seeds = []
    for candidate in coarse:
        if any(same_ellipse(candidate, old, 0.10, 0.10) for old in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= max(10, MAX_REGIONS_PER_CONTOUR * 5):
            break

    refined = []
    for seed in seeds:
        candidate = refine_region(
            seed,
            extended,
            point_count,
            minimum,
            original_mask,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
        )
        if not any(same_ellipse(candidate, old) for old in refined):
            refined.append(candidate)

    refined.sort(key=lambda item: item["score"], reverse=True)
    return refined[:MAX_REGIONS_PER_CONTOUR]


def morphology_guidance(mask):
    """Return a cleaned contour-discovery copy; never used for final validation."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def outer_limb_seed_from_contour(contour, original_mask, min_radius, max_radius,
                                max_error, min_coverage):
    """Fit actual convex-envelope contour vertices as optional outer-limb seeds."""
    if len(contour) < 8:
        return []
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 8:
        return []

    indices = np.unique(hull_indices.reshape(-1))
    points = contour.reshape(-1, 2).astype(np.float64, copy=False)[indices]
    if len(points) < 8:
        return []

    candidates = []
    for ellipse in fit_ellipse_options(points):
        if not axes_in_intermediate_range(ellipse, min_radius, max_radius):
            continue
        relative_error, coverage = ellipse_error_and_coverage(points, ellipse)
        if relative_error > max(max_error * 1.5, max_error + 0.01):
            continue

        candidate = {
            **ellipse,
            "relative_error": relative_error,
            "coverage": coverage,
            "points": points.copy(),
            "arc_segments": [points.copy()],
            "total_supported_coverage": coverage,
            "largest_segment_coverage": coverage,
            "segment_count": 1,
            "start": 0,
            "length": len(points),
        }
        polarity, inside_fraction, sample_count = boundary_polarity(
            original_mask, candidate
        )
        if sample_count < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue
        candidate["boundary_polarity"] = polarity
        candidate["interior_fraction"] = inside_fraction
        candidate["score"] = (
            max(coverage, min_coverage * 0.5)
            * math.sqrt(len(points))
            * polarity**2
            / (relative_error + 0.003)
        )
        candidates.append(candidate)
    return candidates


def find_candidates(original_mask, min_radius, max_radius, max_error,
                    min_coverage, max_contours, max_points,
                    morphology=False, outer_limb_assistance=False):
    """Discover ellipse seeds from contours while validating on original mask."""
    guidance_mask = morphology_guidance(original_mask) if morphology else original_mask
    contours, _ = cv2.findContours(
        guidance_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_points = 12
    min_perimeter = max(12.0, min_radius * min_coverage * math.pi * 0.6)
    usable = []
    for contour in contours:
        if len(contour) < minimum_points:
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
        if max_points > 0 and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
            points = points[indices]

        candidates.extend(
            find_regions(
                points,
                original_mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
                minimum_points,
            )
        )

        if outer_limb_assistance:
            candidates.extend(
                outer_limb_seed_from_contour(
                    contour,
                    original_mask,
                    min_radius,
                    max_radius,
                    max_error,
                    min_coverage,
                )
            )

    candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Ellipse interior helpers
# ---------------------------------------------------------------------------
def ellipse_inside(x_coordinates, y_coordinates, ellipse):
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x_coordinates - cx
    dy = y_coordinates - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (
        (local_x / ellipse["major"])**2
        + (local_y / ellipse["minor"])**2
        <= 1.0
    )


def ellipse_bounds(ellipse, width, height):
    cx, cy = ellipse["center"]
    major = ellipse["major"]
    minor = ellipse["minor"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    extent_x = math.sqrt((major * cos_a)**2 + (minor * sin_a)**2)
    extent_y = math.sqrt((major * sin_a)**2 + (minor * cos_a)**2)
    return (
        max(0, int(math.floor(cx - extent_x))),
        max(0, int(math.floor(cy - extent_y))),
        min(width, int(math.ceil(cx + extent_x)) + 1),
        min(height, int(math.ceil(cy + extent_y)) + 1),
    )


def visible_fraction(mask, candidate, exclude=None):
    height, width = mask.shape
    x0, y0, x1, y1 = ellipse_bounds(candidate, width, height)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    valid = ellipse_inside(xx, yy, candidate)
    if exclude is not None:
        valid &= ~ellipse_inside(xx, yy, exclude)

    visible_pixels = int(np.count_nonzero(valid))
    if visible_pixels == 0:
        return 0.0, 0
    matching = int(np.count_nonzero(mask[y0:y1, x0:x1][valid]))
    return matching / visible_pixels, visible_pixels


# ---------------------------------------------------------------------------
# Horizon detection and half-plane constraints
# ---------------------------------------------------------------------------
def line_signed_distance(points, horizon):
    """Signed perpendicular distance to normalized line n.x + c = 0."""
    points = np.asarray(points, np.float64)
    return (
        points[..., 0] * horizon["normal"][0]
        + points[..., 1] * horizon["normal"][1]
        + horizon["c"]
    )


def points_on_visible_side(points, horizon, margin=0.0):
    if horizon is None:
        return np.ones(len(points), dtype=bool)
    distances = line_signed_distance(points, horizon)
    return distances * horizon["visible_sign"] >= margin


def line_segment_across_image(horizon, width, height):
    """Clip an infinite normalized line to the image rectangle for rendering."""
    nx, ny = horizon["normal"]
    c = horizon["c"]
    points = []

    if abs(ny) > 1e-9:
        for x in (0.0, width - 1.0):
            y = -(nx * x + c) / ny
            if 0 <= y <= height - 1:
                points.append((x, y))
    if abs(nx) > 1e-9:
        for y in (0.0, height - 1.0):
            x = -(ny * y + c) / nx
            if 0 <= x <= width - 1:
                points.append((x, y))

    if len(points) < 2:
        return None

    # Choose the farthest pair if a corner intersection produced duplicates.
    best = None
    best_distance = -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            if d > best_distance:
                best_distance = d
                best = (points[i], points[j])
    return best


def _fit_line_to_points(points):
    points = np.asarray(points, np.float32)
    if len(points) < 2:
        return None
    vx, vy, x0, y0 = cv2.fitLine(
        points.reshape(-1, 1, 2),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    norm = math.hypot(float(vx), float(vy))
    if norm <= 1e-12:
        return None
    vx, vy = float(vx) / norm, float(vy) / norm
    # Unit normal is perpendicular to direction vector.
    nx, ny = -vy, vx
    c = -(nx * float(x0) + ny * float(y0))
    return (vx, vy), (nx, ny), float(c)


def detect_horizon(light_mask, provisional_light):
    """Detect an orientation-independent straight clipping edge through the Sun.

    The search works on ordered *actual threshold-contour points* rather than on
    synthetic Hough chords.  It looks for long low-residual straight runs that
    pass through the interior of the provisional solar ellipse.  Local sampling
    must show a strong light/dark transition across the run, and the inferred
    occluded half-plane must remain essentially dark inside the solar ellipse.

    A curved solar limb can be locally almost straight, especially at large
    radii, so candidate runs close to the ellipse perimeter are rejected: a
    horizon is a chord through the solar disk, not a tangent to its limb.
    """
    if provisional_light is None:
        return None

    height, width = light_mask.shape
    radius = max(provisional_light["equivalent_radius"], 1.0)
    cx, cy = provisional_light["center"]
    x0, y0, x1, y1 = ellipse_bounds(provisional_light, width, height)
    if x0 >= x1 or y0 >= y1:
        return None

    # Work only inside the provisional solar bounding box.  This keeps horizon
    # detection cheap even when the threshold creates enormous sky/land contours
    # elsewhere in the frame.
    roi_mask = light_mask[y0:y1, x0:x1]
    contours, _ = cv2.findContours(
        roi_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    contours = list(contours)
    contours.sort(key=lambda contour: cv2.arcLength(contour, True), reverse=True)
    contours = contours[:12]

    # Ordered contour windows are expressed in approximately one-pixel contour
    # samples because CHAIN_APPROX_NONE is used.  Search several physical spans
    # so short late-sunset horizon chords and longer early contacts are both seen.
    target_lengths = sorted(set(
        max(12, int(round(radius * fraction)))
        for fraction in (0.08, 0.12, 0.16, 0.22, 0.30, 0.42)
    ))

    best = None
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        points = points + np.array([x0, y0], dtype=np.float64)
        count = len(points)
        if count < 12:
            continue

        extended = np.vstack((points, points))
        for length in target_lengths:
            if length > count:
                continue
            step = max(1, length // 6)
            for start_index in range(0, count, step):
                run = extended[start_index:start_index + length]

                # Candidate points must actually traverse the interior of the
                # provisional ellipse.  Mean normalized radius below ~0.93 keeps
                # tangential solar-limb stretches from being called a horizon.
                xn, yn = ellipse_coordinates(run, provisional_light)
                normalized_radius = np.hypot(xn, yn)
                if float(np.mean(normalized_radius)) > 0.93:
                    continue
                if float(np.min(normalized_radius)) > 0.88:
                    continue

                fitted = _fit_line_to_points(run)
                if fitted is None:
                    continue
                (vx, vy), (nx, ny), c = fitted
                signed_run = run[:, 0] * nx + run[:, 1] * ny + c
                residual = float(np.sqrt(np.mean(signed_run**2)))
                max_residual = max(1.25, HORIZON_MAX_RESIDUAL_RADIUS * radius)
                if residual > max_residual:
                    continue

                projection = run[:, 0] * vx + run[:, 1] * vy
                span = float(np.ptp(projection))
                if span < 0.12 * radius:
                    continue

                center_distance = abs(cx * nx + cy * ny + c)
                if center_distance > 0.90 * radius:
                    continue

                # Directly sample a few pixels to either side of the fitted run.
                # This is a local edge-polarity test independent of image rotation.
                sample_offset = max(2.0, 0.012 * radius)
                sample_a = np.rint(
                    run + sample_offset * np.array([nx, ny], dtype=np.float64)
                ).astype(np.int32)
                sample_b = np.rint(
                    run - sample_offset * np.array([nx, ny], dtype=np.float64)
                ).astype(np.int32)
                valid = (
                    (sample_a[:, 0] >= 0) & (sample_a[:, 0] < width)
                    & (sample_a[:, 1] >= 0) & (sample_a[:, 1] < height)
                    & (sample_b[:, 0] >= 0) & (sample_b[:, 0] < width)
                    & (sample_b[:, 1] >= 0) & (sample_b[:, 1] < height)
                )
                if np.count_nonzero(valid) < max(8, len(run) // 2):
                    continue
                a_light = float(np.mean(
                    light_mask[sample_a[valid, 1], sample_a[valid, 0]] != 0
                ))
                b_light = float(np.mean(
                    light_mask[sample_b[valid, 1], sample_b[valid, 0]] != 0
                ))
                local_contrast = abs(a_light - b_light)
                if local_contrast < 0.35:
                    continue

                # Determine the candidate visible side from the local transition.
                visible_sign = 1.0 if a_light > b_light else -1.0

                # Validate the physical half-plane inside the solar ellipse.  The
                # occluded side should be nearly all dark.  The visible side may be
                # a very small crescent late in sunset, so only a small nonzero
                # light fraction is required there; local contrast above carries
                # most of the confidence.
                bx0, by0, bx1, by1 = ellipse_bounds(
                    provisional_light, width, height
                )
                yy, xx = np.ogrid[by0:by1, bx0:bx1]
                inside = ellipse_inside(xx, yy, provisional_light)
                signed_grid = xx * nx + yy * ny + c
                mask_roi = light_mask[by0:by1, bx0:bx1] != 0
                margin = max(1.0, 0.004 * radius)

                visible_region = inside & (signed_grid * visible_sign > margin)
                dark_region = inside & (signed_grid * visible_sign < -margin)
                visible_count = int(np.count_nonzero(visible_region))
                dark_count = int(np.count_nonzero(dark_region))
                if visible_count < 32 or dark_count < 32:
                    continue

                visible_fraction_value = float(np.mean(mask_roi[visible_region]))
                dark_fraction = float(np.mean(mask_roi[dark_region]))
                if dark_fraction > 0.025:
                    continue
                if visible_fraction_value < 0.004:
                    continue
                if visible_fraction_value - dark_fraction < 0.003:
                    continue

                score = (
                    (span / radius)
                    * local_contrast
                    * (1.0 - dark_fraction)
                    * (1.0 + min(0.25, visible_fraction_value))
                    / (1.0 + residual / max(radius, 1.0) * 100.0)
                )
                candidate = {
                    "normal": (float(nx), float(ny)),
                    "c": float(c),
                    "visible_sign": float(visible_sign),
                    "dark_light_fraction": float(dark_fraction),
                    "visible_light_fraction": float(visible_fraction_value),
                    "local_light_contrast": float(local_contrast),
                    "residual": residual,
                    "span": span,
                    "score": score,
                }
                candidate["segment"] = line_segment_across_image(
                    candidate, width, height
                )
                if best is None or candidate["score"] > best["score"]:
                    best = candidate

    return best


# ---------------------------------------------------------------------------
# Multi-segment arc recovery
# ---------------------------------------------------------------------------
def ellipse_points_and_normals(ellipse, theta):
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    local_x = ellipse["major"] * cos_t
    local_y = ellipse["minor"] * sin_t
    x = cx + cos_a * local_x - sin_a * local_y
    y = cy + sin_a * local_x + cos_a * local_y

    nx_local = cos_t / max(ellipse["major"], 1e-9)
    ny_local = sin_t / max(ellipse["minor"], 1e-9)
    nx = cos_a * nx_local - sin_a * ny_local
    ny = sin_a * nx_local + cos_a * ny_local
    norm = np.hypot(nx, ny)
    return x, y, nx / norm, ny / norm


def circular_components(mask):
    count = len(mask)
    if count == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [np.arange(count)]

    start = (int(np.flatnonzero(~mask)[0]) + 1) % count
    order = (start + np.arange(count)) % count
    values = mask[order]
    components = []
    index = 0
    while index < count:
        if not values[index]:
            index += 1
            continue
        end = index
        while end < count and values[end]:
            end += 1
        components.append(order[index:end])
        index = end
    return components


def bridge_small_gaps(supported, max_gap):
    if max_gap <= 0 or not np.any(supported):
        return supported
    bridged = supported.copy()
    # Fill a false bin only when nearby support exists on both sides.  Repeating
    # over small shifts bridges tiny interruptions but does not invent long arcs.
    for shift in range(1, max_gap + 1):
        bridged |= np.roll(supported, shift) & np.roll(supported, -shift)
    return bridged


def grow_arc_support(mask, ellipse, horizon=None):
    """Recover every coherent supported arc segment around a predicted ellipse."""
    theta = np.linspace(0.0, 2.0 * np.pi, ARC_GROW_SAMPLES, endpoint=False)
    px, py, nx, ny = ellipse_points_and_normals(ellipse, theta)

    search_distance = EDGE_SEARCH_FRACTION * min(ellipse["major"], ellipse["minor"])
    step = max(1.0, search_distance / 90.0)
    offsets = np.arange(
        -search_distance,
        search_distance + 0.5 * step,
        step,
        dtype=np.float32,
    )

    sample_x = np.rint(px[:, None] + nx[:, None] * offsets[None, :]).astype(np.int32)
    sample_y = np.rint(py[:, None] + ny[:, None] * offsets[None, :]).astype(np.int32)
    height, width = mask.shape
    valid = (
        (sample_x >= 0) & (sample_x < width)
        & (sample_y >= 0) & (sample_y < height)
    )

    values = np.zeros(sample_x.shape, np.uint8)
    values[valid] = (mask[sample_y[valid], sample_x[valid]] != 0).astype(np.uint8)
    transitions = (
        (values[:, :-1] == 1)
        & (values[:, 1:] == 0)
        & valid[:, :-1]
        & valid[:, 1:]
    )
    midpoint_offsets = 0.5 * (offsets[:-1] + offsets[1:])
    costs = np.where(transitions, np.abs(midpoint_offsets)[None, :], np.inf)
    best_index = np.argmin(costs, axis=1)
    best_cost = costs[np.arange(ARC_GROW_SAMPLES), best_index]
    supported = np.isfinite(best_cost)
    if not np.any(supported):
        return None

    found_offset = midpoint_offsets[best_index]
    found_x = px + nx * found_offset
    found_y = py + ny * found_offset

    # If a horizon exists, the physical constraint is strict: support must be on
    # the visible half-plane, and a small strip around the horizon itself is also
    # removed so the straight horizon edge cannot masquerade as solar curvature.
    if horizon is not None:
        points = np.column_stack((found_x, found_y))
        margin = max(2.0, HORIZON_EXCLUSION_RADIUS * ellipse["equivalent_radius"])
        allowed = points_on_visible_side(points, horizon, margin)
        supported &= allowed
        if not np.any(supported):
            return None

    max_gap = max(1, round(ARC_GROW_SAMPLES * 0.008))
    connected = bridge_small_gaps(supported, max_gap)
    components = circular_components(connected)
    if not components:
        return None

    # Seed bins identify the physical boundary that started the growth.  At least
    # one retained component must overlap the seed; other disconnected components
    # may join the same ellipse when they independently have enough support.
    seed_x, seed_y = ellipse_coordinates(ellipse["points"], ellipse)
    seed_theta = np.mod(np.arctan2(seed_y, seed_x), 2.0 * np.pi)
    seed_bins = set(
        np.mod(
            np.rint(seed_theta / (2.0 * np.pi) * ARC_GROW_SAMPLES).astype(np.int32),
            ARC_GROW_SAMPLES,
        ).tolist()
    )

    retained = []
    seed_overlap_any = False
    min_component_bins = max(3, round(ARC_GROW_SAMPLES * 0.01))

    for component in components:
        actual_indices = component[supported[component]]
        if len(actual_indices) < 5:
            continue
        density = len(actual_indices) / len(component)
        if len(component) < min_component_bins and density < 0.8:
            continue
        overlap = sum(int(index) in seed_bins for index in component)
        if overlap:
            seed_overlap_any = True
        retained.append((component, actual_indices, density, overlap))

    if not retained or not seed_overlap_any:
        return None

    # Keep all reasonably supported components.  Tiny unrelated fragments are
    # already rejected by the component size/density gate above.
    segments = []
    all_points = []
    total_bins = 0
    largest_bins = 0
    support_count = 0
    for component, actual_indices, density, overlap in retained:
        segment_points = np.column_stack(
            (found_x[actual_indices], found_y[actual_indices])
        ).astype(np.float64)
        segments.append(segment_points)
        all_points.append(segment_points)
        total_bins += len(component)
        largest_bins = max(largest_bins, len(component))
        support_count += len(actual_indices)

    return {
        "points": np.vstack(all_points),
        "arc_segments": segments,
        "coverage": total_bins / ARC_GROW_SAMPLES,
        "total_supported_coverage": total_bins / ARC_GROW_SAMPLES,
        "largest_segment_coverage": largest_bins / ARC_GROW_SAMPLES,
        "segment_count": len(segments),
        "supported_fraction": support_count / max(total_bins, 1),
        "support_count": int(support_count),
    }


def growth_compatible(seed, grown):
    scale = max(seed["equivalent_radius"], 1.0)
    sx, sy = seed["center"]
    gx, gy = grown["center"]
    if math.hypot(sx - gx, sy - gy) > 0.24 * scale:
        return False
    if abs(seed["major"] - grown["major"]) > 0.30 * scale:
        return False
    if abs(seed["minor"] - grown["minor"]) > 0.30 * scale:
        return False
    return True


def grow_candidate(mask, seed, min_radius, max_radius, max_error,
                   min_coverage, horizon=None):
    # A seed that already satisfies the FINAL constraints remains a legitimate
    # fallback.  Arc growth is an improvement stage, not a requirement that can
    # erase an otherwise valid short arc.  Seeds outside the final axis range are
    # allowed to continue searching but cannot be returned unchanged.
    fallback = None
    if (
        axes_in_range(seed, min_radius, max_radius)
        and seed.get("relative_error", math.inf) <= max_error
        and seed.get("coverage", 0.0) >= min_coverage
    ):
        fallback = seed.copy()
        fallback.setdefault("arc_segments", [np.asarray(seed["points"], np.float64)])
        fallback.setdefault("total_supported_coverage", fallback["coverage"])
        fallback.setdefault("largest_segment_coverage", fallback["coverage"])
        fallback.setdefault("segment_count", 1)
        fallback.setdefault("supported_fraction", 1.0)
        fallback.setdefault("support_count", len(fallback["points"]))
        if horizon is not None:
            margin = max(2.0, HORIZON_EXCLUSION_RADIUS * fallback["equivalent_radius"])
            if not np.all(points_on_visible_side(fallback["points"], horizon, margin)):
                fallback = None

    support = grow_arc_support(mask, seed, horizon=horizon)
    if support is None:
        return fallback

    best = None
    for ellipse in fit_ellipse_options(support["points"]):
        # Exact user axis limits are imposed here, after all disconnected support
        # available to the model has been gathered.
        if not axes_in_range(ellipse, min_radius, max_radius):
            continue

        relative_error, _ = ellipse_error_and_coverage(support["points"], ellipse)
        if relative_error > max_error:
            continue

        candidate = {
            **ellipse,
            **support,
            "relative_error": relative_error,
            "start": seed.get("start", 0),
            "length": support["support_count"],
        }
        # A seed already inside the final axis range should not drift far while
        # growing.  A deliberately relaxed intermediate seed, however, is allowed
        # to move substantially: the whole purpose of the relaxed search envelope
        # is to let short, poorly constrained arcs converge once more real support
        # is available.  Final axis/error/polarity checks remain mandatory.
        if axes_in_range(seed, min_radius, max_radius):
            if not growth_compatible(seed, candidate):
                continue

        # Horizon constraint is rechecked after refitting because the fitted model
        # can move slightly relative to the support from the seed geometry.
        if horizon is not None:
            margin = max(2.0, HORIZON_EXCLUSION_RADIUS * candidate["equivalent_radius"])
            if not np.all(points_on_visible_side(candidate["points"], horizon, margin)):
                continue

        polarity, inside_fraction, sample_count = boundary_polarity(mask, candidate)
        if sample_count < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue

        candidate["boundary_polarity"] = polarity
        candidate["interior_fraction"] = inside_fraction
        candidate["score"] = (
            candidate["coverage"]
            * (0.65 + 0.35 * candidate["supported_fraction"])
            * (0.65 + 0.35 * polarity)
            / (1.0 + 4.0 * relative_error)
        )
        if candidate["coverage"] < min_coverage:
            continue

        if best is None or (
            candidate["coverage"],
            candidate["largest_segment_coverage"],
            candidate["score"],
            candidate["support_count"],
        ) > (
            best["coverage"],
            best["largest_segment_coverage"],
            best["score"],
            best["support_count"],
        ):
            best = candidate
    return best if best is not None else fallback


def prepare_candidates(mask, candidates, min_radius, max_radius, max_error,
                       min_coverage, horizon=None):
    prepared = []
    for seed in candidates[:MAX_CLASS_CANDIDATES]:
        candidate = grow_candidate(
            mask,
            seed,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
            horizon=horizon,
        )
        if candidate is None:
            continue
        if any(same_ellipse(candidate, old, 0.08, 0.08, 12.0) for old in prepared):
            continue
        prepared.append(candidate)

    prepared.sort(
        key=lambda item: (
            item["coverage"],
            item.get("largest_segment_coverage", 0.0),
            item.get("supported_fraction", 1.0),
            item.get("boundary_polarity", 0.0),
            -item["relative_error"],
            item.get("score", 0.0),
        ),
        reverse=True,
    )
    return prepared


# ---------------------------------------------------------------------------
# Dark/light detection orchestration
# ---------------------------------------------------------------------------
def detect(gray, threshold, min_radius, max_radius, max_error, min_coverage,
           max_contours, max_points, morphology=False,
           outer_limb_assistance=False, color_image=None):
    """Detect at most one dark and one light ellipse plus an optional horizon."""
    _, dark_mask = cv2.threshold(
        gray, int(threshold), 255, cv2.THRESH_BINARY_INV
    )
    _, light_mask = cv2.threshold(
        gray, int(threshold), 255, cv2.THRESH_BINARY
    )

    dark_seeds = find_candidates(
        dark_mask,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
        morphology=morphology,
        outer_limb_assistance=outer_limb_assistance,
    )
    light_seeds = find_candidates(
        light_mask,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
        morphology=morphology,
        outer_limb_assistance=outer_limb_assistance,
    )

    # Prepare provisional light candidates without a horizon solely so a straight
    # clipping edge can be tested relative to a plausible solar geometry.
    provisional_light_candidates = prepare_candidates(
        light_mask,
        light_seeds,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        horizon=None,
    )
    provisional_light = (
        provisional_light_candidates[0] if provisional_light_candidates else None
    )

    # Horizon detection is always attempted.  Color information, when available
    # in the GUI/preview path, supplies only a coarse solar-location hint; the
    # actual horizon is still measured from the original grayscale threshold
    # boundary.  A provisional light ellipse far from the warm/bright Sun is not
    # allowed to steer horizon detection to an unrelated skyline edge.
    solar_hint = adaptive_solar_hint(color_image) if color_image is not None else None
    horizon_reference = provisional_light
    if solar_hint is not None and solar_hint.get("center") is not None:
        expected_radius = 0.5 * (min_radius + max_radius)
        if horizon_reference is not None:
            distance = math.hypot(
                horizon_reference["center"][0] - solar_hint["center"][0],
                horizon_reference["center"][1] - solar_hint["center"][1],
            )
            if distance > 1.25 * expected_radius:
                horizon_reference = None
        if horizon_reference is None:
            horizon_reference = {
                "center": tuple(solar_hint["center"]),
                "major": expected_radius,
                "minor": expected_radius,
                "angle": 0.0,
                "equivalent_radius": expected_radius,
            }

    horizon = detect_horizon(light_mask, horizon_reference)

    # Once a horizon is accepted, both classes are re-grown with the half-plane
    # constraint so no support survives on the physically dark side.
    dark_candidates = prepare_candidates(
        dark_mask,
        dark_seeds,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        horizon=horizon,
    )
    if horizon is None:
        light_candidates = provisional_light_candidates
    else:
        light_candidates = prepare_candidates(
            light_mask,
            light_seeds,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
            horizon=horizon,
        )

    dark = dark_candidates[0].copy() if dark_candidates else None
    if dark is not None:
        dark["class"] = "below threshold"

    # Preserve the established totality safeguard for now.  The user explicitly
    # requested that this remain unchanged rather than become configurable yet.
    if dark is not None and dark["coverage"] >= DARK_DOMINANT_COVERAGE:
        return light_mask, [dark], horizon

    light = None
    for candidate in light_candidates:
        if dark is not None and same_ellipse(candidate, dark, 0.08, 0.08, 12.0):
            continue
        if dark is not None:
            fraction, visible_pixels = visible_fraction(light_mask, candidate, dark)
            if visible_pixels < 32 or fraction < 0.40:
                continue
        light = candidate.copy()
        light["class"] = "above threshold"
        break

    ellipses = [ellipse for ellipse in (dark, light) if ellipse is not None]
    return light_mask, ellipses, horizon


# ---------------------------------------------------------------------------
# Rendering and scale conversion
# ---------------------------------------------------------------------------
def scale_ellipse(ellipse, factor):
    """Map a detection ellipse/support/horizon-compatible geometry by scale."""
    result = ellipse.copy()
    result["center"] = (
        ellipse["center"][0] * factor,
        ellipse["center"][1] * factor,
    )
    result["major"] = ellipse["major"] * factor
    result["minor"] = ellipse["minor"] * factor
    result["equivalent_radius"] = ellipse["equivalent_radius"] * factor
    result["points"] = np.asarray(ellipse["points"], np.float64) * factor
    result["arc_segments"] = [
        np.asarray(segment, np.float64) * factor
        for segment in ellipse.get("arc_segments", [ellipse["points"]])
    ]
    return result


def scale_horizon(horizon, factor):
    """Map normalized line from working coordinates to original coordinates."""
    if horizon is None:
        return None
    # n*x_work + c_work = 0 and x_work = x_full / factor, therefore
    # n*x_full + c_work*factor = 0 in full coordinates.
    result = horizon.copy()
    result["c"] = horizon["c"] * factor
    result["span"] = horizon["span"] * factor
    result["residual"] = horizon["residual"] * factor
    result["segment"] = None
    return result


def center_color_image_on_light_ellipse(color_image, ellipses):
    """Integer-translate the original color raster to center the light ellipse."""
    if color_image is None:
        return None
    light = next(
        (e for e in ellipses if e.get("class") == "above threshold"),
        None,
    )
    if light is None:
        return color_image.copy()

    height, width = color_image.shape[:2]
    source_x = int(round(light["center"][0]))
    source_y = int(round(light["center"][1]))
    shift_x = width // 2 - source_x
    shift_y = height // 2 - source_y

    sx0 = max(0, -shift_x)
    sy0 = max(0, -shift_y)
    dx0 = max(0, shift_x)
    dy0 = max(0, shift_y)
    copy_width = min(width - sx0, width - dx0)
    copy_height = min(height - sy0, height - dy0)

    centered = np.zeros_like(color_image)
    if copy_width > 0 and copy_height > 0:
        centered[dy0:dy0 + copy_height, dx0:dx0 + copy_width] = (
            color_image[sy0:sy0 + copy_height, sx0:sx0 + copy_width]
        )
    return centered


def annotate_threshold(binary, ellipses, horizon):
    preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for ellipse in ellipses:
        color = (
            DARK_ELLIPSE_COLOR
            if ellipse["class"] == "below threshold"
            else LIGHT_ELLIPSE_COLOR
        )

        # Draw each actual support segment independently.  Missing gaps remain
        # visually empty and do not become artificial magenta connections.
        for segment in ellipse.get("arc_segments", [ellipse["points"]]):
            support = np.rint(segment).astype(np.int32).reshape(-1, 1, 2)
            if len(support) >= 2:
                cv2.polylines(
                    preview,
                    [support],
                    False,
                    ARC_COLOR,
                    ARC_LINE_THICKNESS,
                    cv2.LINE_AA,
                )

        cx, cy = ellipse["center"]
        center = (int(round(cx)), int(round(cy)))
        axes = (
            max(1, int(round(ellipse["major"]))),
            max(1, int(round(ellipse["minor"]))),
        )
        cv2.ellipse(
            preview,
            center,
            axes,
            ellipse["angle"],
            0,
            360,
            color,
            ELLIPSE_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.circle(preview, center, 3, color, -1)

        kind = "DARK <= T" if ellipse["class"] == "below threshold" else "BRIGHT > T"
        text = (
            f"{kind} a={ellipse['major']:.1f} b={ellipse['minor']:.1f} "
            f"support={ellipse['coverage'] * 100:.1f}% "
            f"segments={ellipse.get('segment_count', 1)} "
            f"err={ellipse['relative_error']:.3f}"
        )
        cv2.putText(
            preview,
            text,
            (center[0] + 10, max(18, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if horizon is not None:
        segment = horizon.get("segment")
        if segment is None:
            segment = line_segment_across_image(
                horizon,
                preview.shape[1],
                preview.shape[0],
            )
        if segment is not None:
            p1, p2 = segment
            cv2.line(
                preview,
                (int(round(p1[0])), int(round(p1[1]))),
                (int(round(p2[0])), int(round(p2[1]))),
                HORIZON_COLOR,
                HORIZON_LINE_THICKNESS,
                cv2.LINE_AA,
            )
    return preview


def process_working_image(gray, threshold, min_radius, max_radius, max_error,
                          min_coverage, max_contours, max_points, morphology,
                          outer_limb_assistance, color_image=None):
    binary, ellipses, horizon = detect(
        gray,
        threshold,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
        morphology=morphology,
        outer_limb_assistance=outer_limb_assistance,
        color_image=color_image,
    )
    threshold_preview = annotate_threshold(binary, ellipses, horizon)
    return threshold_preview, ellipses, horizon


def resize_for_detection(image, max_dim):
    """Return aspect-preserving reduced BGR image and working/original scale."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image.copy(), 1.0
    scale = max_dim / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), scale


# ---------------------------------------------------------------------------
# Threshold candidate generation and automatic initialization
# ---------------------------------------------------------------------------
def adaptive_solar_hint(bgr):
    """Locate a coarse warm/bright solar region for threshold initialization.

    The hint is deliberately *not* an authoritative ellipse result.  It only
    provides a color-aware location and several grayscale anchors so automatic
    threshold selection does not prefer a geometrically plausible edge elsewhere
    in a difficult sunset frame.
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)

    v_hi = float(np.percentile(v, 99.8))
    g_hi = float(np.percentile(gray, 99.8))
    v_thr = int(max(20, min(245, max(v_hi * 0.72, float(v.max()) * 0.65))))
    g_thr = int(max(20, min(245, max(g_hi * 0.72, float(gray.max()) * 0.65))))

    # Warm saturated pixels cover orange/red sunset Suns; the low-saturation arm
    # keeps pale/white solar disks eligible earlier in the sequence.
    warm = ((h <= 40) | (h >= 170)) & ((s >= 10) | (s <= 35))
    bright = (v >= v_thr) | (gray >= g_thr)
    solar_like = np.where(warm & bright, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        solar_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) >= 20]
    center = None
    visible_radius = 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )
            visible_radius = math.sqrt(max(cv2.contourArea(contour), 1.0) / math.pi)

    hints = {g_thr}
    values = gray[solar_like != 0]
    if values.size >= 100:
        for q in (1, 5, 10, 20):
            hints.add(int(np.percentile(values, q)))

        ys, xs = np.nonzero(solar_like)
        if len(xs):
            pad = max(10, int(0.15 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(gray.shape[1], int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(gray.shape[0], int(ys.max()) + pad + 1)
            roi = gray[y0:y1, x0:x1]
            if roi.size:
                otsu, _ = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                hints.add(int(round(otsu)))

    return {
        "center": center,
        "visible_radius": float(visible_radius),
        "gray_threshold": int(g_thr),
        "threshold_hints": [int(np.clip(v, 0, 255)) for v in hints],
        "mask": solar_like,
    }


def adaptive_brightness_estimate(bgr):
    """Return grayscale threshold anchors from adaptive color/brightness evidence."""
    return adaptive_solar_hint(bgr)["threshold_hints"]

def build_threshold_candidates(gray, bgr=None, max_colors=20, fallback=8):
    """Choose thresholds that are likely to change segmentation meaningfully.

    Unlike the former palette, this function does not simply return abundant
    grayscale values.  It prioritizes boundaries around strong histogram peaks,
    valleys between populations, large histogram slopes, and adaptive brightness
    hints.  Values are returned sorted for an understandable UI.
    """
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    # Mild smoothing makes derivative/valley decisions less sensitive to JPEG
    # quantization while exact peak neighbors are still included separately.
    smooth = np.convolve(histogram, np.ones(5) / 5.0, mode="same")
    candidates = {int(np.clip(fallback, 0, 255))}

    # Strong exact peaks: include the transition immediately before the peak, the
    # peak itself, and the next level.  For <=T semantics, g-1 versus g is often
    # particularly informative because all pixels equal to g switch class there.
    peak_order = np.argsort(histogram)[::-1]
    strong_peaks = []
    for value in peak_order:
        value = int(value)
        if histogram[value] <= 0:
            break
        if all(abs(value - old) >= 3 for old in strong_peaks):
            strong_peaks.append(value)
        if len(strong_peaks) >= 6:
            break
    for peak in strong_peaks:
        for value in (peak - 1, peak, peak + 1):
            if 0 <= value <= 255:
                candidates.add(value)

    # Large positive/negative slopes mark class-boundary positions that can alter
    # the mask more than a histogram mode itself.
    derivative = np.diff(smooth)
    slope_indices = np.argsort(np.abs(derivative))[::-1][:10]
    for index in slope_indices:
        candidates.add(int(np.clip(index, 0, 255)))
        candidates.add(int(np.clip(index + 1, 0, 255)))

    # Valleys between major separated peaks are classic threshold choices.
    sorted_peaks = sorted(strong_peaks)
    for left, right in zip(sorted_peaks, sorted_peaks[1:]):
        if right - left < 4:
            continue
        valley = left + int(np.argmin(smooth[left:right + 1]))
        candidates.add(valley)

    if bgr is not None:
        for hint in adaptive_brightness_estimate(bgr):
            for delta in (-2, -1, 0, 1, 2):
                value = hint + delta
                if 0 <= value <= 255:
                    candidates.add(value)

    # Rank candidates by how much histogram mass changes close to the threshold,
    # plus preference for valleys/slopes.  This ranking determines which 20 are
    # offered; final presentation remains numerically sorted.
    scored = []
    total = max(float(histogram.sum()), 1.0)
    for value in candidates:
        neighborhood = histogram[max(0, value - 1):min(256, value + 2)].sum() / total
        slope = abs(derivative[value]) if value < 255 else abs(derivative[-1])
        valley_bonus = 1.0 / (smooth[value] / max(smooth.max(), 1.0) + 0.05)
        score = neighborhood * 4.0 + slope / max(smooth.max(), 1.0) + 0.02 * valley_bonus
        scored.append((score, value))

    scored.sort(reverse=True)
    selected = []
    for _, value in scored:
        if value not in selected:
            selected.append(value)
        if len(selected) >= max_colors:
            break

    # Ensure fallback remains available even if ranking displaced it.
    if fallback not in selected:
        if len(selected) >= max_colors:
            selected[-1] = fallback
        else:
            selected.append(fallback)
    return sorted(set(int(v) for v in selected))[:max_colors]


def score_detection_for_threshold(ellipses, horizon, solar_hint=None, expected_radius=None):
    """Heuristic used only to choose an initial threshold on a small preview."""
    if not ellipses:
        return -1e9
    score = 0.0
    classes = {ellipse.get("class") for ellipse in ellipses}
    if "above threshold" in classes:
        score += 2.0
    if "below threshold" in classes:
        score += 1.0
    # When a threshold can recover both physical limbs, strongly prefer it over a
    # single-class threshold with merely a somewhat longer arc.
    if len(ellipses) == 2:
        score += 5.0

    for ellipse in ellipses:
        score += 5.0 * ellipse.get("coverage", 0.0)
        score += 1.2 * ellipse.get("boundary_polarity", 0.0)
        score += 0.5 * ellipse.get("supported_fraction", 1.0)
        score -= 3.0 * ellipse.get("relative_error", 1.0)

    # Color-aware location is used only during initialization.  It prevents a
    # large skyline/background edge from winning simply because it fits an ellipse
    # somewhere far from the warm bright Sun.
    if solar_hint is not None and solar_hint.get("center") is not None:
        light = next(
            (e for e in ellipses if e.get("class") == "above threshold"),
            None,
        )
        if light is not None:
            radius = max(float(expected_radius or light["equivalent_radius"]), 1.0)
            distance = math.hypot(
                light["center"][0] - solar_hint["center"][0],
                light["center"][1] - solar_hint["center"][1],
            ) / radius
            score += max(-10.0, 5.0 - 6.0 * distance)
        else:
            score -= 1.5

    if horizon is not None:
        score += 0.4 * min(1.0, horizon.get("score", 0.0))
    return score

def auto_select_threshold(color_image, fallback_threshold, min_radius, max_radius,
                          max_error, min_coverage, max_contours, max_points,
                          palette_size=20):
    """Select a per-image starting threshold using cheap reduced-size detection."""
    working, scale = resize_for_detection(color_image, AUTO_THRESHOLD_MAX_DIM)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    solar_hint = adaptive_solar_hint(working)
    all_candidates = build_threshold_candidates(
        gray,
        working,
        max_colors=max(20, min(palette_size, 30)),
        fallback=fallback_threshold,
    )
    # Auto-initialization evaluates only a compact set: thresholds around the
    # established fallback and around the adaptive color/brightness anchor.
    # This keeps first-load latency bounded while still exploring both regimes.
    anchor = int(solar_hint["gray_threshold"])
    candidate_pool = set(all_candidates)
    candidate_pool.add(int(fallback_threshold))
    candidate_pool.add(anchor)
    for delta in (-2, -1, 1, 2):
        if 0 <= anchor + delta <= 255:
            candidate_pool.add(anchor + delta)
        if 0 <= fallback_threshold + delta <= 255:
            candidate_pool.add(int(fallback_threshold + delta))

    ordered = sorted(
        candidate_pool,
        key=lambda value: min(
            abs(value - int(fallback_threshold)),
            abs(value - anchor),
        ),
    )
    candidates = ordered[:10]

    scaled_min = max(1.0, min_radius * scale)
    scaled_max = max(scaled_min, max_radius * scale)
    best_threshold = int(fallback_threshold)
    best_score = -1e18
    best_has_near_light = False
    expected_radius = 0.5 * (scaled_min + scaled_max)

    # Use tighter search caps during initialization.  The selected threshold is
    # then immediately evaluated by the normal Refresh Preview pass.
    test_contours = min(max_contours, 12) if max_contours > 0 else 12
    test_points = min(max_points, 120) if max_points > 0 else 120

    for threshold in candidates:
        try:
            _, ellipses, horizon = detect(
                gray,
                threshold,
                scaled_min,
                scaled_max,
                max_error,
                min_coverage,
                test_contours,
                test_points,
                morphology=False,
                outer_limb_assistance=False,
                color_image=working,
            )
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            continue
        score = score_detection_for_threshold(
            ellipses, horizon, solar_hint=solar_hint, expected_radius=expected_radius
        )
        near_light = False
        if solar_hint.get("center") is not None:
            light = next((e for e in ellipses if e.get("class") == "above threshold"), None)
            if light is not None:
                near_light = (
                    math.hypot(
                        light["center"][0] - solar_hint["center"][0],
                        light["center"][1] - solar_hint["center"][1],
                    )
                    <= 1.0 * expected_radius
                )
        # Small tie preference toward thresholds nearer the legacy fallback so
        # equally good results do not jump to extreme values unnecessarily.
        score -= 0.0005 * abs(threshold - fallback_threshold)
        if score > best_score:
            best_score = score
            best_threshold = int(threshold)
            best_has_near_light = near_light

    # If no tested threshold produced a light ellipse in the color-indicated Sun
    # neighborhood, prefer the adaptive grayscale anchor instead of a confident
    # false ellipse elsewhere in the frame.  Manual tuning/outer-limb assistance
    # can then work from a physically sensible starting point.
    if solar_hint.get("center") is not None and not best_has_near_light:
        anchor = int(solar_hint["gray_threshold"])
        best_threshold = min(candidates, key=lambda value: abs(value - anchor))

    return int(best_threshold)


# ---------------------------------------------------------------------------
# Tkinter UI
# ---------------------------------------------------------------------------
class DetectorApp:
    """Single-window multi-image detector with preview/full-resolution passes."""

    def __init__(self, root, image_paths, args):
        self.root = root
        self.args = args
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path = None
        self.color_image = None
        self.gray = None
        self.palette = []

        # Each initialized image ALWAYS has a threshold entry.  Other values are
        # stored only when they deviate from session defaults.
        self.image_overrides = {}
        self.restoring_settings = False

        self.last_threshold_preview = None
        self.last_color_preview = None
        self.threshold_photo = None
        self.color_photo = None
        self.resize_job = None
        self.result_mode = None

        defaults = self.effective_default_settings()
        self.threshold = tk.IntVar(value=defaults["threshold"])
        self.min_radius = tk.IntVar(value=round(defaults["min_radius"]))
        self.max_radius = tk.IntVar(value=round(defaults["max_radius"]))
        self.max_error = tk.DoubleVar(value=defaults["max_error"] * 100)
        self.min_coverage = tk.IntVar(value=round(defaults["min_coverage"] * 100))
        self.morphology = tk.BooleanVar(value=False)
        self.outer_limb_assistance = tk.BooleanVar(value=False)

        self.status = tk.StringVar(value="Load images to begin.")
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector")
        root.minsize(1050, 760)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.build_navigation()
        self.build_controls()
        self.build_previews()

        # Enter remains the authoritative full-resolution action.
        root.bind("<Return>", lambda _event: self.apply_full_resolution())
        root.bind("<Escape>", lambda _event: self.close())

        if self.image_paths:
            self.load_image_at(0)
        else:
            self.update_navigation_state()

    # ------------------------------------------------------------------
    # Per-image state
    # ------------------------------------------------------------------
    def effective_default_settings(self):
        min_radius = max(1.0, float(self.args.min_radius))
        if self.args.max_radius == 0:
            if self.gray is not None:
                max_radius = float(max(self.gray.shape))
            else:
                max_radius = max(1600.0, min_radius)
        else:
            max_radius = float(self.args.max_radius)
        max_radius = max(max_radius, min_radius)
        return {
            "threshold": int(self.args.threshold),
            "min_radius": min_radius,
            "max_radius": max_radius,
            "max_error": float(self.args.max_error),
            "min_coverage": float(self.args.min_coverage),
            "morphology": False,
            "outer_limb_assistance": False,
        }

    def settings_dict(self):
        min_radius = max(1.0, float(self.min_radius.get()))
        max_radius = max(1.0, float(self.max_radius.get()))
        if max_radius < min_radius:
            max_radius = min_radius
            self.restoring_settings = True
            try:
                self.max_radius.set(round(max_radius))
            finally:
                self.restoring_settings = False

        return {
            "threshold": int(self.threshold.get()),
            "min_radius": min_radius,
            "max_radius": max_radius,
            "max_error": max(0.001, self.max_error.get() / 100),
            "min_coverage": float(np.clip(self.min_coverage.get() / 100, 0, 1)),
            "morphology": bool(self.morphology.get()),
            "outer_limb_assistance": bool(self.outer_limb_assistance.get()),
        }

    @staticmethod
    def setting_equal(name, value, default):
        if isinstance(default, bool):
            return bool(value) == bool(default)
        if name == "threshold":
            return int(value) == int(default)
        return math.isclose(float(value), float(default), rel_tol=0.0, abs_tol=1e-9)

    def store_current_overrides(self):
        if self.current_path is None:
            return
        current = self.settings_dict()
        defaults = self.effective_default_settings()

        # Threshold is always explicit once the image exists in the session.
        overrides = {"threshold": int(current["threshold"])}
        for name in NON_THRESHOLD_SETTING_NAMES:
            if not self.setting_equal(name, current[name], defaults[name]):
                overrides[name] = current[name]
        self.image_overrides[self.current_path] = overrides

    def restore_settings_for_current_image(self):
        defaults = self.effective_default_settings()
        values = dict(defaults)
        values.update(self.image_overrides.get(self.current_path, {}))

        self.restoring_settings = True
        try:
            self.threshold.set(round(values["threshold"]))
            self.min_radius.set(round(values["min_radius"]))
            self.max_radius.set(round(values["max_radius"]))
            self.max_error.set(values["max_error"] * 100)
            self.min_coverage.set(round(values["min_coverage"] * 100))
            self.morphology.set(bool(values["morphology"]))
            self.outer_limb_assistance.set(bool(values["outer_limb_assistance"]))
        finally:
            self.restoring_settings = False

    # ------------------------------------------------------------------
    # Image list and automatic initialization
    # ------------------------------------------------------------------
    def load_images(self):
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Select eclipse images",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not selected:
            return
        self.store_current_overrides()
        self.image_paths = [os.path.abspath(path) for path in selected]
        self.current_index = -1
        self.current_path = None
        self.color_image = None
        self.gray = None
        self.palette = []
        self.load_image_at(0)

    def initialize_threshold_if_needed(self):
        if self.current_path in self.image_overrides and "threshold" in self.image_overrides[self.current_path]:
            return

        defaults = self.effective_default_settings()
        self.status.set("Analyzing image to choose an initial threshold...")
        self.root.update_idletasks()
        threshold = auto_select_threshold(
            self.color_image,
            defaults["threshold"],
            defaults["min_radius"],
            defaults["max_radius"],
            defaults["max_error"],
            defaults["min_coverage"],
            self.args.max_contours,
            self.args.max_search_points,
            palette_size=self.args.palette_size,
        )
        self.image_overrides[self.current_path] = {"threshold": int(threshold)}

    def load_image_at(self, index):
        if not 0 <= index < len(self.image_paths):
            return
        if self.current_path is not None:
            self.store_current_overrides()

        path = self.image_paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        self.current_index = index
        self.current_path = path

        if image is None:
            self.color_image = None
            self.gray = None
            self.palette = []
            self.last_threshold_preview = None
            self.last_color_preview = None
            self.threshold_canvas.delete("all")
            self.color_canvas.delete("all")
            self.placeholder(self.threshold_canvas, "Threshold preview")
            self.placeholder(self.color_canvas, "Unreadable image")
            self.update_navigation_state()
            self.status.set(f"Could not load image: {path}")
            return

        self.color_image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # First encounter: compute and explicitly store a threshold.  Returning to
        # the image later simply restores that stored value, even when it is 8.
        self.initialize_threshold_if_needed()
        self.restore_settings_for_current_image()
        self.update_radius_scale_limits()

        self.palette = build_threshold_candidates(
            self.gray,
            self.color_image,
            max_colors=self.args.palette_size,
            fallback=int(self.threshold.get()),
        )
        self.rebuild_palette_buttons()

        self.last_threshold_preview = None
        self.last_color_preview = self.color_image.copy()
        self.result_mode = None
        self.threshold_photo = None
        self.threshold_canvas.delete("all")
        self.placeholder(self.threshold_canvas, "Threshold preview")
        self.redraw()
        self.update_navigation_state()

        # Loading/navigation now performs the cheap preview pass, not an automatic
        # full-resolution recomputation.
        self.refresh_preview()

    def previous_image(self):
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_image(self):
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    def update_navigation_state(self):
        count = len(self.image_paths)
        has_current = 0 <= self.current_index < count
        readable = has_current and self.gray is not None and self.color_image is not None

        self.previous_button.config(
            state=tk.NORMAL if has_current and self.current_index > 0 else tk.DISABLED
        )
        self.next_button.config(
            state=tk.NORMAL if has_current and self.current_index < count - 1 else tk.DISABLED
        )
        self.preview_button.config(state=tk.NORMAL if readable else tk.DISABLED)
        self.full_button.config(state=tk.NORMAL if readable else tk.DISABLED)

        if has_current:
            self.image_info.set(
                f"{self.current_index + 1} / {count}   {os.path.basename(self.current_path)}"
            )
        else:
            self.image_info.set("No image loaded" if not count else f"0 / {count}")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_navigation(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(3, weight=1)

        tk.Button(frame, text="Load images...", width=14, command=self.load_images).grid(
            row=0, column=0, padx=(0, 8)
        )
        self.previous_button = tk.Button(
            frame, text="◀ Previous", width=12, command=self.previous_image
        )
        self.previous_button.grid(row=0, column=1, padx=(0, 5))
        self.next_button = tk.Button(
            frame, text="Next ▶", width=12, command=self.next_image
        )
        self.next_button.grid(row=0, column=2, padx=(0, 10))
        tk.Label(frame, textvariable=self.image_info, anchor="w").grid(
            row=0, column=3, sticky="ew"
        )

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        defaults = self.effective_default_settings()
        radius_limit = max(1600, round(defaults["max_radius"] * 1.5))
        self.radius_scales = {}

        rows = [
            ("threshold", "Brightness threshold (dark <= T, light > T)", self.threshold,
             0, 255, 1, lambda v: str(int(v))),
            ("min_radius", "Minimum FINAL fitted semi-axis radius (px)", self.min_radius,
             1, radius_limit, 1, lambda v: f"{int(v)} px"),
            ("max_radius", "Maximum FINAL fitted semi-axis radius (px)", self.max_radius,
             1, radius_limit, 1, lambda v: f"{int(v)} px"),
            ("max_error", "Maximum average normalized ellipse error (%)", self.max_error,
             0.5, 50, 0.1, lambda v: f"{float(v):.1f}%"),
            ("min_coverage", "Minimum TOTAL supported ellipse arc (%)", self.min_coverage,
             0, 100, 1, lambda v: f"{int(v)}% (~{int(v) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            name, *args = spec
            scale = self.add_scale(frame, row, *args)
            if name in ("min_radius", "max_radius"):
                self.radius_scales[name] = scale

        options = tk.Frame(frame)
        options.grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 5))
        tk.Checkbutton(
            options,
            text="Morphology cleanup for candidate search",
            variable=self.morphology,
            command=self.pending,
        ).grid(row=0, column=0, sticky="w", padx=(0, 20))
        tk.Checkbutton(
            options,
            text="Outer-limb assistance",
            variable=self.outer_limb_assistance,
            command=self.pending,
        ).grid(row=0, column=1, sticky="w")

        tk.Label(frame, text="Useful threshold candidates:").grid(
            row=6, column=0, sticky="nw"
        )
        self.palette_frame = tk.Frame(frame)
        self.palette_frame.grid(
            row=6, column=1, columnspan=2, sticky="w", pady=(0, 8)
        )

        button_frame = tk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self.preview_button = tk.Button(
            button_frame,
            text="Refresh Preview",
            width=16,
            command=self.refresh_preview,
        )
        self.preview_button.grid(row=0, column=0, padx=(0, 8))
        self.full_button = tk.Button(
            button_frame,
            text="Apply Full Resolution",
            width=20,
            command=self.apply_full_resolution,
        )
        self.full_button.grid(row=0, column=1)

        tk.Label(
            frame,
            textvariable=self.status,
            anchor="w",
            justify="left",
            wraplength=1150,
        ).grid(row=8, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=42, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        scale = tk.Scale(
            parent,
            from_=low,
            to=high,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            showvalue=False,
            length=420,
            highlightthickness=0,
        )
        scale.grid(row=row, column=1, sticky="ew", pady=2)
        label = tk.Label(parent, width=18, anchor="e")
        label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            label.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update_value)
        update_value()
        return scale

    def update_radius_scale_limits(self):
        defaults = self.effective_default_settings()
        current_high = max(
            defaults["max_radius"],
            float(self.min_radius.get()),
            float(self.max_radius.get()),
        )
        radius_limit = max(1600, round(current_high * 1.5))
        for scale in self.radius_scales.values():
            scale.config(to=radius_limit)

    def rebuild_palette_buttons(self):
        for child in self.palette_frame.winfo_children():
            child.destroy()
        selected = int(self.threshold.get())
        for index, shade in enumerate(self.palette):
            is_selected = shade == selected
            button = tk.Button(
                self.palette_frame,
                text=(f"[{shade}]" if is_selected else str(shade)),
                width=5,
                bg=gray_hex(shade),
                fg=text_color(shade),
                relief=(tk.SUNKEN if is_selected else tk.RAISED),
                command=lambda value=shade: self.pick(value),
            )
            button.grid(row=index // 10, column=index % 10, padx=2, pady=2)

    def build_previews(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(
            frame,
            text="Threshold preview: arcs / ellipses / detected horizon",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(
            frame,
            text="Full-color image centered on detected light ellipse",
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.threshold_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.color_canvas = tk.Canvas(
            frame, bg="#202020", highlightthickness=1, highlightbackground="#808080"
        )
        self.threshold_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.color_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.threshold_canvas.bind("<Configure>", self.schedule_redraw)
        self.color_canvas.bind("<Configure>", self.schedule_redraw)
        self.placeholder(self.threshold_canvas, "Threshold preview")
        self.placeholder(self.color_canvas, "Color image")

    # ------------------------------------------------------------------
    # User actions and two-level recomputation
    # ------------------------------------------------------------------
    def pick(self, value):
        self.threshold.set(value)
        self.store_current_overrides()
        self.rebuild_palette_buttons()
        self.status.set(f"Threshold set to {value}. Refresh Preview or Apply Full Resolution.")

    def pending(self, *_args):
        if self.restoring_settings:
            return
        if self.current_path is not None:
            self.store_current_overrides()
            self.status.set(
                "Settings changed and remembered for this image. "
                "Refresh Preview or Apply Full Resolution."
            )

    def run_detection(self, full_resolution):
        if self.color_image is None or self.gray is None or self.current_path is None:
            self.status.set("Load at least one readable image first.")
            return

        settings = self.settings_dict()
        self.store_current_overrides()
        started = time.perf_counter()

        if full_resolution:
            working_color = self.color_image
            scale = 1.0
            mode_label = "FULL RESOLUTION"
        else:
            working_color, scale = resize_for_detection(self.color_image, PREVIEW_MAX_DIM)
            mode_label = "PREVIEW"

        working_gray = cv2.cvtColor(working_color, cv2.COLOR_BGR2GRAY)
        working_min = max(1.0, settings["min_radius"] * scale)
        working_max = max(working_min, settings["max_radius"] * scale)

        threshold_preview, working_ellipses, working_horizon = process_working_image(
            working_gray,
            settings["threshold"],
            working_min,
            working_max,
            settings["max_error"],
            settings["min_coverage"],
            self.args.max_contours,
            self.args.max_search_points,
            settings["morphology"],
            settings["outer_limb_assistance"],
            color_image=working_color,
        )

        # Map preview geometry to original coordinates only for the full-color
        # centering pane.  Full-resolution mode already has factor 1.
        to_original = 1.0 / scale
        original_ellipses = [scale_ellipse(e, to_original) for e in working_ellipses]
        original_horizon = scale_horizon(working_horizon, to_original)
        if original_horizon is not None:
            original_horizon["segment"] = line_segment_across_image(
                original_horizon,
                self.color_image.shape[1],
                self.color_image.shape[0],
            )

        color_preview = center_color_image_on_light_ellipse(
            self.color_image,
            original_ellipses,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.last_threshold_preview = threshold_preview
        self.last_color_preview = color_preview
        self.result_mode = mode_label
        self.redraw()

        horizon_text = "horizon=yes" if working_horizon is not None else "horizon=no"
        self.status.set(
            f"{mode_label}: working={working_gray.shape[1]}x{working_gray.shape[0]}; "
            f"T={settings['threshold']}; final semi-axes={settings['min_radius']:.0f}-"
            f"{settings['max_radius']:.0f}px; error={settings['max_error']:.1%}; "
            f"support={settings['min_coverage']:.0%}; morphology="
            f"{'on' if settings['morphology'] else 'off'}; outer-limb="
            f"{'on' if settings['outer_limb_assistance'] else 'off'}; "
            f"{len(working_ellipses)} ellipse(s); {horizon_text}; {elapsed_ms:.1f} ms."
        )

        for ellipse in original_ellipses:
            print(
                f"{os.path.basename(self.current_path)} | {mode_label} | "
                f"{ellipse['class']}: center=({ellipse['center'][0]:.2f}, "
                f"{ellipse['center'][1]:.2f}), a={ellipse['major']:.2f}, "
                f"b={ellipse['minor']:.2f}, angle={ellipse['angle']:.1f}, "
                f"support={ellipse['coverage'] * 360:.1f}°, "
                f"segments={ellipse.get('segment_count', 1)}, "
                f"error={ellipse['relative_error']:.4f}"
            )
        if original_horizon is not None:
            print(
                f"{os.path.basename(self.current_path)} | {mode_label} | horizon: "
                f"dark-light={original_horizon['dark_light_fraction']:.1%}, "
                f"visible-light={original_horizon['visible_light_fraction']:.1%}, "
                f"residual={original_horizon['residual']:.2f}px"
            )

    def refresh_preview(self):
        self.run_detection(full_resolution=False)

    def apply_full_resolution(self):
        self.run_detection(full_resolution=True)

    def schedule_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        self.resize_job = None
        if self.last_threshold_preview is not None:
            self.threshold_photo = self.show_image(
                self.threshold_canvas,
                self.last_threshold_preview,
            )
        if self.last_color_preview is not None:
            self.color_photo = self.show_image(
                self.color_canvas,
                self.last_color_preview,
            )

    @staticmethod
    def show_image(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = max(
            min(canvas_width / image_width, canvas_height / image_height),
            1e-6,
        )
        size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(image, size, interpolation=interpolation)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
        photo = tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"),
            format="png",
        )
        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2 + 1,
            canvas_height // 2 + 1,
            image=photo,
            anchor="center",
        )
        return photo

    @staticmethod
    def placeholder(canvas, text):
        canvas.create_text(
            160,
            120,
            text=text,
            fill="#cccccc",
            justify="center",
        )

    def close(self):
        self.store_current_overrides()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Small UI helpers and CLI
# ---------------------------------------------------------------------------
def gray_hex(value):
    value = int(np.clip(value, 0, 255))
    return f"#{value:02x}{value:02x}{value:02x}"


def text_color(gray_value):
    return "#111111" if gray_value >= 150 else "#f7f7f7"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detect dark/light eclipse ellipses with preview and full-resolution modes."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="Optional ordered input image list; more images can be loaded in the GUI",
    )
    # This remains the emergency fallback used only when automatic threshold
    # selection cannot find a stronger initial candidate.
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--min-radius", type=float, default=1000.0)
    parser.add_argument("--max-radius", type=float, default=1500.0)
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    parser.add_argument("--max-contours", type=int, default=100)
    parser.add_argument("--max-search-points", type=int, default=500)
    parser.add_argument("--palette-size", type=int, default=20)
    return parser


def validate_args(args, parser):
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be 0..255")
    if args.min_radius <= 0:
        parser.error("--min-radius must be > 0")
    if args.max_radius < 0:
        parser.error("--max-radius must be >= 0")
    if args.max_error <= 0:
        parser.error("--max-error must be > 0")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be 0..1")
    if args.max_contours < 0 or args.max_search_points < 0:
        parser.error("search limits must be >= 0")
    if not 1 <= args.palette_size <= 40:
        parser.error("--palette-size must be 1..40")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    root = tk.Tk()
    DetectorApp(root, args.images, args)
    root.mainloop()


if __name__ == "__main__":
    main()
