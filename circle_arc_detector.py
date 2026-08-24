"""Interactive eclipse limb detector based on thresholded ellipse arcs.

The script is intentionally self-contained.  It performs three jobs:

1. Convert the input image to grayscale and threshold it into two complementary
   masks:
      * dark mask  -> grayscale <= threshold
      * light mask -> grayscale > threshold
   Equality therefore belongs to the dark class by design.

2. Search contour fragments in each mask for a plausible rotated ellipse.  The
   search is deliberately tolerant of short and partially obscured eclipse
   limbs.  It tries several OpenCV ellipse fits plus a circle-shaped stabilizing
   prior, checks local boundary polarity, refines promising contour windows,
   and then attempts to grow each accepted arc along the predicted ellipse.

3. Present the result in a single Tkinter window.  The threshold preview shows
   the detected support arc in magenta, the dark ellipse in blue, and the light
   ellipse in golden yellow.  The second preview remains plain grayscale.

4. Manage an ordered list of input images.  Previous/Next arrow buttons load one
   image at a time.  Settings that differ from the session defaults are kept in
   an in-memory ``image_overrides`` dictionary keyed by absolute file path.  On
   navigation, the defaults are restored first, then that image's overrides are
   re-applied, and the normal Apply computation is run once automatically.

Changing sliders or threshold buttons still does not recompute detection by
probe/drag; it only updates the pending per-image settings.  Apply/Enter remains
the manual recomputation path between image loads.

Detection is limited to at most two ellipses: one dark-class ellipse and one
light-class ellipse.  All final fitted semi-axes must independently stay inside
user-selected minimum/maximum radius limits.

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
# Detection tuning constants
# ---------------------------------------------------------------------------
# Fraction of sampled boundary points that must show the expected transition
# from inside the selected mask to outside it.  This rejects unrelated contour
# fragments even when their geometry happens to resemble an ellipse.
MIN_BOUNDARY_POLARITY = 0.42

# Only the strongest candidates are worth the comparatively expensive arc-
# growth stage.  Keeping the cap also bounds UI recomputation time.
MAX_CLASS_CANDIDATES = 30

# Each source contour may contribute several geometrically distinct ellipse
# hypotheses, but not an unbounded number of near-duplicates.
MAX_REGIONS_PER_CONTOUR = 4

# Arc growth searches along each ellipse normal within +/- 15% of the smaller
# semi-axis.  There is intentionally NO pixel cap; for a 1200 px semi-axis this
# is an approximately +/-180 px search envelope.
EDGE_SEARCH_FRACTION = 0.15

# Number of parametric samples around a full ellipse during arc growth.
# 720 samples corresponds to 0.5 degree angular spacing.
ARC_GROW_SAMPLES = 720

# A dark candidate covering most of the circumference is treated as the
# totality/near-totality case.  In that situation bright corona structure can
# easily create a spurious second ellipse, so the dark ellipse is returned by
# itself.
DARK_DOMINANT_COVERAGE = 0.55

# Refinement changes the cyclic contour window by one point at a time.  These
# moves expand either end, shrink either end, or slide the window while keeping
# its length unchanged.
REFINE_MOVES = (
    (-1, 1),
    (0, 1),
    (1, -1),
    (0, -1),
    (-1, 0),
    (1, 0),
)

# Preview colors use OpenCV BGR order.
ARC_COLOR = (255, 0, 255)          # magenta
DARK_ELLIPSE_COLOR = (255, 0, 0)  # blue
LIGHT_ELLIPSE_COLOR = (0, 190, 255)  # golden/orange yellow
ELLIPSE_LINE_THICKNESS = 3
ARC_LINE_THICKNESS = 2

# These are exactly the controls whose values may vary per image.  Search caps
# and threshold-palette generation parameters remain global/session options.
SETTING_NAMES = (
    "threshold",
    "min_radius",
    "max_radius",
    "max_error",
    "min_coverage",
)

# Tk's multi-file chooser returns the selected paths in one tuple.  OpenCV can
# read these common formats; "All files" remains available for camera-specific
# extensions that OpenCV may support on a particular installation.
IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)


# ---------------------------------------------------------------------------
# Ellipse fitting and geometry helpers
# ---------------------------------------------------------------------------
def normalize_ellipse(raw):
    """Convert an OpenCV ellipse tuple to the detector's canonical format.

    OpenCV returns ``((cx, cy), (width, height), angle)``.  The detector stores
    semi-axis lengths instead of full diameters and always names the larger
    semi-axis ``major``.  When the OpenCV width/height order has to be swapped,
    the orientation is rotated by 90 degrees so the geometry remains identical.

    ``equivalent_radius = sqrt(major * minor)`` is not an acceptance radius; it
    is only a convenient scale for duplicate and compatibility comparisons.
    """
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
        major = semi_x
        minor = semi_y
        major_angle = angle
    else:
        major = semi_y
        minor = semi_x
        major_angle = angle + 90.0

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
    """Return True only when BOTH fitted semi-axes satisfy the user limits."""
    return (
        min_radius <= ellipse["major"] <= max_radius
        and min_radius <= ellipse["minor"] <= max_radius
    )


def fit_circle_prior(points):
    """Fit a circle and expose it as a zero-eccentricity ellipse candidate.

    Short arcs do not constrain all five parameters of a general ellipse very
    strongly.  A circle fit is therefore useful as a stabilizing hypothesis.  It
    does not force the final result to be circular: the same arc is also tested
    with the three general OpenCV ellipse fitters below.
    """
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

    radius_sq = float(squared_distance.mean()) + u_center * u_center + v_center * v_center
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
    """Generate several ellipse hypotheses for the same arc points.

    OpenCV exposes multiple fitting algorithms with different numerical
    behavior on short/noisy arcs.  Trying Direct, AMS, and the standard method,
    then adding a circle prior, is more robust than relying on one fitter alone.
    Invalid fits are quietly discarded.
    """
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
    """Transform image points into normalized coordinates of an ellipse.

    On an exact ellipse the transformed points satisfy ``x^2 + y^2 == 1``.
    These normalized coordinates are used for both residual measurement and
    parametric arc-angle measurement.
    """
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
    """Measure normalized fit residual and represented arc fraction.

    Residual:
        For normalized point radius ``rho = hypot(x, y)``, an exact ellipse has
        rho == 1.  The returned error is ``mean(abs(rho - 1))``.

    Coverage:
        Points are converted to ellipse parametric angles.  Their unwrapped
        angular span is divided by 2*pi, giving the visible fraction of the full
        fitted ellipse.
    """
    if len(points) < 2:
        return math.inf, 0.0

    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    relative_error = float(np.mean(np.abs(np.hypot(x_norm, y_norm) - 1.0)))

    angles = np.unwrap(np.arctan2(y_norm, x_norm))
    coverage = float(np.clip(np.ptp(angles) / (2.0 * np.pi), 0.0, 1.0))
    return relative_error, coverage


def boundary_polarity(mask, ellipse):
    """Check that fitted arc points cross from mask-inside to mask-outside.

    Whole-ellipse interior brightness is not a reliable classifier during an
    eclipse: the Moon can make most of the Sun's projected disk dark.  Instead,
    classification is based locally at the fitted boundary.

    For every support point we sample slightly inward and outward along the
    ellipse's normalized radial direction.  A correct boundary for the supplied
    binary mask should be True just inside and False just outside.

    Returns ``(polarity_fraction, inside_fraction, valid_sample_count)``.
    """
    points = np.asarray(ellipse["points"], np.float64)
    if len(points) < 5:
        return 0.0, 0.0, 0

    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    theta = np.arctan2(y_norm, x_norm)

    # Sample about 0.8% of the ellipse scale away from the boundary, with a
    # three-pixel floor so very small test images still get separated samples.
    radius = ellipse["equivalent_radius"]
    sample_distance = max(3.0, radius * 0.008)
    scale_delta = sample_distance / max(radius, 1.0)

    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    cx, cy = ellipse["center"]

    def sample_at_scale(scale):
        local_x = ellipse["major"] * scale * np.cos(theta)
        local_y = ellipse["minor"] * scale * np.sin(theta)
        x = np.rint(cx + cos_a * local_x - sin_a * local_y).astype(np.int32)
        y = np.rint(cy + sin_a * local_x + cos_a * local_y).astype(np.int32)
        return x, y

    inside_x, inside_y = sample_at_scale(1.0 - scale_delta)
    outside_x, outside_y = sample_at_scale(1.0 + scale_delta)

    height, width = mask.shape
    valid = (
        (inside_x >= 0)
        & (inside_x < width)
        & (inside_y >= 0)
        & (inside_y < height)
        & (outside_x >= 0)
        & (outside_x < width)
        & (outside_y >= 0)
        & (outside_y < height)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0, 0.0, 0

    inside_values = mask[inside_y[valid], inside_x[valid]] != 0
    outside_values = mask[outside_y[valid], outside_x[valid]] != 0

    polarity = float(np.mean(inside_values & ~outside_values))
    inside_fraction = float(np.mean(inside_values))
    return polarity, inside_fraction, valid_count


# ---------------------------------------------------------------------------
# Contour-window search and refinement
# ---------------------------------------------------------------------------
def window_lengths(point_count, minimum):
    """Return coarse contour-window lengths from ``minimum`` to full contour.

    Geometric growth (x1.45) examines short, medium, and long arcs without
    testing every possible length during the expensive coarse search.
    """
    minimum = max(5, min(minimum, point_count))
    lengths = {minimum, point_count}

    length = minimum
    while length < point_count:
        lengths.add(length)
        length = min(
            point_count,
            max(length + 1, int(round(length * 1.45))),
        )

    return sorted(lengths)


def evaluate_region(
    extended_points,
    point_count,
    start,
    length,
    mask,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
):
    """Fit and score one cyclic contour window; return its best hypothesis.

    ``extended_points`` contains the contour twice, allowing a slice to pass
    through the original array boundary without special wraparound code.
    """
    if length < 5 or length > point_count:
        return None

    start %= point_count
    region = extended_points[start : start + length]
    best = None

    for ellipse in fit_ellipse_options(region):
        # The user requires BOTH final semi-axes to satisfy the selected range.
        # The current search also enforces that same range for its hypotheses so
        # behavior remains identical to the established detector.
        if not axes_in_range(ellipse, min_radius, max_radius):
            continue

        relative_error, coverage = ellipse_error_and_coverage(region, ellipse)
        if relative_error > max_error or coverage < min_coverage:
            continue

        candidate = {
            **ellipse,
            "relative_error": relative_error,
            "coverage": coverage,
            "points": region.copy(),
            "start": start,
            "length": length,
        }

        polarity, inside_fraction, sample_count = boundary_polarity(mask, candidate)
        if (
            sample_count < 5
            or polarity < MIN_BOUNDARY_POLARITY
            or inside_fraction < 0.5
        ):
            continue

        # Geometry score rewards longer meaningful arcs and more contour support,
        # while penalizing residual error and extreme ellipse aspect ratios.
        contour_support = length / point_count
        axis_ratio = candidate["major"] / max(candidate["minor"], 1.0)
        shape_penalty = 1.0 + 0.06 * max(0.0, axis_ratio - 1.0)
        geometry_score = (
            coverage**1.5
            * math.sqrt(length)
            * (0.75 + 0.25 * math.sqrt(contour_support))
            / ((relative_error + 0.002) * shape_penalty)
        )

        candidate["boundary_polarity"] = polarity
        candidate["interior_fraction"] = inside_fraction
        candidate["score"] = geometry_score * polarity**2

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


def angle_difference_180(a, b):
    """Smallest orientation difference for ellipses, modulo 180 degrees."""
    difference = abs((a - b) % 180.0)
    return min(difference, 180.0 - difference)


def same_ellipse(
    first,
    second,
    center_fraction=0.12,
    axis_fraction=0.12,
    angle_tolerance=15.0,
):
    """Return True when two candidates represent the same physical ellipse.

    Tolerances scale with ellipse size.  Orientation is ignored for nearly
    circular fits because a circle's major-axis angle is numerically arbitrary.
    """
    first_x, first_y = first["center"]
    second_x, second_y = second["center"]
    scale = max(
        first["equivalent_radius"],
        second["equivalent_radius"],
        1.0,
    )

    if math.hypot(first_x - second_x, first_y - second_y) >= center_fraction * scale:
        return False
    if abs(first["major"] - second["major"]) >= axis_fraction * scale:
        return False
    if abs(first["minor"] - second["minor"]) >= axis_fraction * scale:
        return False

    first_eccentricity = (
        (first["major"] - first["minor"]) / max(first["major"], 1.0)
    )
    second_eccentricity = (
        (second["major"] - second["minor"]) / max(second["major"], 1.0)
    )
    if max(first_eccentricity, second_eccentricity) < 0.05:
        return True

    return angle_difference_180(first["angle"], second["angle"]) < angle_tolerance


def refine_region(
    candidate,
    extended_points,
    point_count,
    minimum_length,
    mask,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
):
    """Hill-climb around a coarse contour window to improve its score."""
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
                mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1.0 + 1e-9):
                best = trial
                improved = True

        if not improved:
            break

    return best


def find_regions(
    points,
    mask,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
    minimum=12,
):
    """Find a few distinct ellipse hypotheses within one ordered contour."""
    points = np.asarray(points, np.float64)
    point_count = len(points)
    if point_count < max(5, minimum):
        return []

    minimum = min(minimum, point_count)
    extended_points = np.vstack((points, points))

    # Coarse phase: sample many cyclic windows at geometrically spaced lengths.
    coarse = []
    for length in window_lengths(point_count, minimum):
        step = max(1, length // 5)
        for start in range(0, point_count, step):
            candidate = evaluate_region(
                extended_points,
                point_count,
                start,
                length,
                mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if candidate is not None:
                coarse.append(candidate)

    coarse.sort(key=lambda item: item["score"], reverse=True)

    # Keep only strong, geometrically different seeds before local refinement.
    seeds = []
    seed_limit = max(10, MAX_REGIONS_PER_CONTOUR * 5)
    for candidate in coarse:
        if any(same_ellipse(candidate, existing, 0.10, 0.10) for existing in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= seed_limit:
            break

    # Refine each seed and de-duplicate once more at the tighter final stage.
    refined = []
    for seed in seeds:
        candidate = refine_region(
            seed,
            extended_points,
            point_count,
            minimum,
            mask,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
        )
        if not any(same_ellipse(candidate, existing) for existing in refined):
            refined.append(candidate)

    refined.sort(key=lambda item: item["score"], reverse=True)
    return refined[:MAX_REGIONS_PER_CONTOUR]


def find_candidates(
    mask,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
    max_contours,
    max_points,
):
    """Extract contours from one mask and collect all ellipse hypotheses."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_points = 12

    # Very short contours cannot represent the requested minimum visible arc.
    # This perimeter gate is intentionally loose because the visible arc may be
    # distorted and because ellipse perimeter is not simply 2*pi*radius.
    min_perimeter = max(12.0, min_radius * min_coverage * math.pi)

    usable = []
    for contour in contours:
        if len(contour) < minimum_points:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter >= min_perimeter:
            usable.append((perimeter, contour))

    # Search longer contours first; they are more likely to contain meaningful
    # limb geometry.  max_contours == 0 intentionally means "no cap".
    usable.sort(key=lambda item: item[0], reverse=True)
    if max_contours > 0:
        usable = usable[:max_contours]

    candidates = []
    for _, contour in usable:
        points = contour.reshape(-1, 2).astype(np.float64, copy=False)

        # Downsample only the search representation, not the source image.  This
        # keeps long high-resolution contours computationally manageable.
        if max_points > 0 and len(points) > max_points:
            indices = np.linspace(
                0,
                len(points) - 1,
                max_points,
                dtype=np.int32,
            )
            points = points[indices]

        candidates.extend(
            find_regions(
                points,
                mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
                minimum_points,
            )
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Ellipse interior checks used when selecting dark/light classes
# ---------------------------------------------------------------------------
def ellipse_inside(x_coordinates, y_coordinates, ellipse):
    """Vectorized point-in-rotated-ellipse test."""
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    dx = x_coordinates - cx
    dy = y_coordinates - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy

    return (
        (local_x / ellipse["major"]) ** 2
        + (local_y / ellipse["minor"]) ** 2
        <= 1.0
    )


def ellipse_bounds(ellipse, width, height):
    """Return the clipped image-space bounding box of a rotated ellipse."""
    cx, cy = ellipse["center"]
    major = ellipse["major"]
    minor = ellipse["minor"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    extent_x = math.sqrt((major * cos_a) ** 2 + (minor * sin_a) ** 2)
    extent_y = math.sqrt((major * sin_a) ** 2 + (minor * cos_a) ** 2)

    return (
        max(0, int(math.floor(cx - extent_x))),
        max(0, int(math.floor(cy - extent_y))),
        min(width, int(math.ceil(cx + extent_x)) + 1),
        min(height, int(math.ceil(cy + extent_y)) + 1),
    )


def visible_fraction(mask, candidate, exclude=None):
    """Measure mask occupancy inside a candidate, optionally excluding another.

    This is only used to reject an implausible light ellipse after a dark ellipse
    has already been selected.  Pixels geometrically owned by the dark ellipse
    are removed before evaluating how much of the remaining light candidate is
    actually bright.
    """
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

    matching_pixels = int(np.count_nonzero(mask[y0:y1, x0:x1][valid]))
    return matching_pixels / visible_pixels, visible_pixels


# ---------------------------------------------------------------------------
# Arc-growth stage
# ---------------------------------------------------------------------------
def ellipse_points_and_normals(ellipse, theta):
    """Return ellipse points and unit outward normals for parametric angles."""
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

    # Gradient of x^2/a^2 + y^2/b^2 = 1 gives the local ellipse normal.
    normal_x_local = cos_t / max(ellipse["major"], 1e-9)
    normal_y_local = sin_t / max(ellipse["minor"], 1e-9)
    normal_x = cos_a * normal_x_local - sin_a * normal_y_local
    normal_y = sin_a * normal_x_local + cos_a * normal_y_local

    norm = np.hypot(normal_x, normal_y)
    return x, y, normal_x / norm, normal_y / norm


def circular_components(mask):
    """Return contiguous True-index runs on a circular boolean array."""
    count = len(mask)
    if count == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [np.arange(count)]

    # Start immediately after a False element so a wrapped True run becomes one
    # ordinary linear run rather than two fragments at indices 0 and n-1.
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
    """Join tiny missing runs between nearby supported angular samples."""
    if max_gap <= 0 or not np.any(supported):
        return supported

    bridged = supported.copy()
    for shift in range(1, max_gap + 1):
        left = np.roll(supported, shift)
        right = np.roll(supported, -shift)
        bridged |= left & right

    return bridged


def grow_arc_support(mask, ellipse):
    """Search around a fitted ellipse and recover the largest connected arc.

    At each of 720 ellipse angles, search along the local normal for the nearest
    inside->outside transition in the supplied mask.  The allowed normal search
    distance is 15% of the smaller semi-axis with no fixed pixel cap.

    The connected component overlapping the original seed arc is chosen so the
    growth phase expands the same physical boundary rather than jumping to an
    unrelated edge elsewhere in the search envelope.
    """
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        ARC_GROW_SAMPLES,
        endpoint=False,
    )
    predicted_x, predicted_y, normal_x, normal_y = ellipse_points_and_normals(
        ellipse,
        theta,
    )

    edge_search_distance = EDGE_SEARCH_FRACTION * min(
        ellipse["major"],
        ellipse["minor"],
    )

    # Use at most roughly 180 normal-direction intervals across the full search
    # band.  On small ellipses the one-pixel floor avoids redundant subpixel
    # samples that would round to the same image coordinates.
    step = max(1.0, edge_search_distance / 90.0)
    offsets = np.arange(
        -edge_search_distance,
        edge_search_distance + 0.5 * step,
        step,
        dtype=np.float32,
    )

    sample_x = np.rint(
        predicted_x[:, None] + normal_x[:, None] * offsets[None, :]
    ).astype(np.int32)
    sample_y = np.rint(
        predicted_y[:, None] + normal_y[:, None] * offsets[None, :]
    ).astype(np.int32)

    height, width = mask.shape
    valid = (
        (sample_x >= 0)
        & (sample_x < width)
        & (sample_y >= 0)
        & (sample_y < height)
    )

    # Out-of-image samples remain zero and are excluded by the valid mask when
    # transitions are computed.
    values = np.zeros(sample_x.shape, np.uint8)
    values[valid] = (mask[sample_y[valid], sample_x[valid]] != 0).astype(np.uint8)

    # Look specifically for an inside (1) -> outside (0) mask transition as we
    # travel along the outward ellipse normal.
    transitions = (
        (values[:, :-1] == 1)
        & (values[:, 1:] == 0)
        & valid[:, :-1]
        & valid[:, 1:]
    )

    midpoint_offsets = 0.5 * (offsets[:-1] + offsets[1:])
    costs = np.where(
        transitions,
        np.abs(midpoint_offsets)[None, :],
        np.inf,
    )

    # For each ellipse angle choose the valid transition nearest the predicted
    # ellipse.  Angles with no transition remain unsupported.
    best_index = np.argmin(costs, axis=1)
    best_cost = costs[np.arange(ARC_GROW_SAMPLES), best_index]
    supported = np.isfinite(best_cost)
    if not np.any(supported):
        return None

    found_offset = midpoint_offsets[best_index]
    found_x = predicted_x + normal_x * found_offset
    found_y = predicted_y + normal_y * found_offset

    # Convert original seed points to the same 720-bin angle representation.
    seed_x, seed_y = ellipse_coordinates(ellipse["points"], ellipse)
    seed_theta = np.mod(np.arctan2(seed_y, seed_x), 2.0 * np.pi)
    seed_bins = set(
        np.mod(
            np.rint(
                seed_theta / (2.0 * np.pi) * ARC_GROW_SAMPLES
            ).astype(np.int32),
            ARC_GROW_SAMPLES,
        ).tolist()
    )

    # Bridge only very small interruptions (about 0.8% of a circumference), then
    # identify connected angular components on the circular sample array.
    max_gap = max(1, round(ARC_GROW_SAMPLES * 0.008))
    connected = bridge_small_gaps(supported, max_gap)
    components = circular_components(connected)
    if not components:
        return None

    def component_key(component):
        seed_overlap = sum(int(index) in seed_bins for index in component)
        actual_support = int(np.count_nonzero(supported[component]))
        return seed_overlap, actual_support, len(component)

    component = max(components, key=component_key)
    if component_key(component)[0] == 0:
        return None

    actual_indices = component[supported[component]]
    if len(actual_indices) < 5:
        return None

    return {
        "points": np.column_stack(
            (found_x[actual_indices], found_y[actual_indices])
        ).astype(np.float64),
        "coverage": len(component) / ARC_GROW_SAMPLES,
        "supported_fraction": len(actual_indices) / len(component),
        "support_count": int(len(actual_indices)),
    }


def growth_compatible(seed, grown):
    """Reject a grown/refitted ellipse that drifted too far from its seed."""
    scale = max(seed["equivalent_radius"], 1.0)
    seed_x, seed_y = seed["center"]
    grown_x, grown_y = grown["center"]

    if math.hypot(seed_x - grown_x, seed_y - grown_y) > 0.18 * scale:
        return False
    if abs(seed["major"] - grown["major"]) > 0.20 * scale:
        return False
    if abs(seed["minor"] - grown["minor"]) > 0.20 * scale:
        return False

    return True


def grow_candidate(
    mask,
    seed,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
):
    """Expand a seed arc, refit its geometry, and keep the best larger result."""
    support = grow_arc_support(mask, seed)
    if support is None or support["coverage"] <= seed["coverage"]:
        return seed

    best = None
    for ellipse in fit_ellipse_options(support["points"]):
        if not axes_in_range(ellipse, min_radius, max_radius):
            continue

        relative_error, _ = ellipse_error_and_coverage(
            support["points"],
            ellipse,
        )
        if relative_error > max_error:
            continue

        candidate = {
            **ellipse,
            **support,
            "relative_error": relative_error,
            # Preserve contour-window metadata expected by downstream logic.
            "start": seed["start"],
            "length": support["support_count"],
        }

        if not growth_compatible(seed, candidate):
            continue

        polarity, inside_fraction, sample_count = boundary_polarity(mask, candidate)
        if sample_count < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue

        candidate["boundary_polarity"] = polarity
        candidate["interior_fraction"] = inside_fraction

        # At this stage coverage is deliberately the dominant signal.  Support
        # density, polarity, and residual only refine the choice among arcs of
        # similar angular extent.
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
            candidate["score"],
            candidate["support_count"],
        ) > (
            best["coverage"],
            best["score"],
            best["support_count"],
        ):
            best = candidate

    return best if best is not None else seed


def prepare_candidates(
    mask,
    candidates,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
):
    """Grow, de-duplicate, and rank the strongest candidates for one class."""
    prepared = []

    for seed in candidates[:MAX_CLASS_CANDIDATES]:
        candidate = grow_candidate(
            mask,
            seed,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
        )

        if any(
            same_ellipse(candidate, existing, 0.08, 0.08, 12.0)
            for existing in prepared
        ):
            continue

        prepared.append(candidate)

    # Coverage comes first by design: when multiple plausible fits exist, prefer
    # the candidate that explains the largest coherent observed limb.
    prepared.sort(
        key=lambda item: (
            item["coverage"],
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
def detect(
    gray,
    threshold,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
    max_contours,
    max_points,
):
    """Detect at most one dark ellipse and one light ellipse.

    Threshold semantics are intentionally asymmetric at equality:
        dark mask  = gray <= threshold
        light mask = gray > threshold

    The dark class is selected first.  Its geometric interior is excluded when
    validating the light class so overlap is owned by the dark ellipse.
    """
    _, dark_mask = cv2.threshold(
        gray,
        int(threshold),
        255,
        cv2.THRESH_BINARY_INV,
    )
    _, light_mask = cv2.threshold(
        gray,
        int(threshold),
        255,
        cv2.THRESH_BINARY,
    )

    dark_candidates = find_candidates(
        dark_mask,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
    )
    dark_candidates = prepare_candidates(
        dark_mask,
        dark_candidates,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
    )

    light_candidates = find_candidates(
        light_mask,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
    )
    light_candidates = prepare_candidates(
        light_mask,
        light_candidates,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
    )

    # The prepared list is already ordered primarily by recovered arc coverage.
    dark = dark_candidates[0].copy() if dark_candidates else None
    if dark is not None:
        dark["class"] = "below threshold"

    # During totality a dominant dark limb is the expected physical result; a
    # second bright ellipse is more likely to be corona/ring structure.
    if dark is not None and dark["coverage"] >= DARK_DOMINANT_COVERAGE:
        return light_mask, [dark]

    light = None
    for candidate in light_candidates:
        # Never return the same geometry once as dark and again as light.
        if dark is not None and same_ellipse(candidate, dark, 0.08, 0.08, 12.0):
            continue

        if dark is not None:
            light_fraction, visible_pixels = visible_fraction(
                light_mask,
                candidate,
                dark,
            )
            if visible_pixels < 32 or light_fraction < 0.40:
                continue

        light = candidate.copy()
        light["class"] = "above threshold"
        break

    ellipses = [
        ellipse
        for ellipse in (dark, light)
        if ellipse is not None
    ]
    return light_mask, ellipses


# ---------------------------------------------------------------------------
# Rendering and grayscale threshold palette
# ---------------------------------------------------------------------------
def process_image(
    gray,
    threshold,
    min_radius,
    max_radius,
    max_error,
    min_coverage,
    max_contours,
    max_points,
):
    """Run detection and build the two images displayed by the Tkinter UI."""
    binary, ellipses = detect(
        gray,
        threshold,
        min_radius,
        max_radius,
        max_error,
        min_coverage,
        max_contours,
        max_points,
    )

    # The left pane is a color version of the threshold mask so annotations can
    # be drawn without altering threshold semantics.  The right pane is kept as
    # a strictly grayscale visual reference.
    threshold_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    grayscale_preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for ellipse in ellipses:
        cx, cy = ellipse["center"]
        center = (int(round(cx)), int(round(cy)))
        axes = (
            max(1, int(round(ellipse["major"]))),
            max(1, int(round(ellipse["minor"]))),
        )

        ellipse_color = (
            DARK_ELLIPSE_COLOR
            if ellipse["class"] == "below threshold"
            else LIGHT_ELLIPSE_COLOR
        )

        # Magenta marks the actual contour/grown support used by the fitted
        # candidate.  The complete inferred ellipse is drawn in class color.
        support = np.rint(ellipse["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support) >= 2:
            cv2.polylines(
                threshold_preview,
                [support],
                False,
                ARC_COLOR,
                ARC_LINE_THICKNESS,
                cv2.LINE_AA,
            )

        cv2.ellipse(
            threshold_preview,
            center,
            axes,
            ellipse["angle"],
            0,
            360,
            ellipse_color,
            ELLIPSE_LINE_THICKNESS,
            cv2.LINE_AA,
        )
        cv2.circle(threshold_preview, center, 3, ellipse_color, -1)

        kind = (
            "DARK <= T"
            if ellipse["class"] == "below threshold"
            else "BRIGHT > T"
        )
        text = (
            f"{kind}  a={ellipse['major']:.1f}  b={ellipse['minor']:.1f}  "
            f"arc={ellipse['coverage'] * 360:.0f}deg  "
            f"err={ellipse['relative_error']:.3f}"
        )
        cv2.putText(
            threshold_preview,
            text,
            (center[0] + 10, max(18, center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            ellipse_color,
            1,
            cv2.LINE_AA,
        )

    return threshold_preview, grayscale_preview, ellipses


def build_palette(gray, max_colors=20, min_gap=10):
    """Choose useful grayscale threshold buttons from ALL image pixels.

    The first five entries are the most frequent exact grayscale levels, with no
    spacing restriction.  Remaining slots favor other dense histogram regions
    while requiring at least ``min_gap`` grayscale levels from every previously
    selected threshold.  The returned buttons are sorted numerically.
    """
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(float)

    dominant_seed_count = min(5, max_colors)
    shades = []

    # Seed with the most common exact tones before applying spacing.
    for shade in np.argsort(histogram)[::-1]:
        shade = int(shade)
        if histogram[shade] <= 0:
            break
        shades.append(shade)
        if len(shades) >= dominant_seed_count:
            break

    # Smoothed histogram density is more stable than ranking individual bins for
    # the remaining suggestions.
    smooth_radius = max(2, min_gap // 3)
    density = np.convolve(
        histogram,
        np.ones(2 * smooth_radius + 1),
        mode="same",
    )

    for shade in np.argsort(density)[::-1]:
        shade = int(shade)
        if density[shade] <= 0:
            break
        if shade in shades:
            continue
        if any(abs(shade - selected) < min_gap for selected in shades):
            continue

        shades.append(shade)
        if len(shades) >= max_colors:
            break

    return [int(shade) for shade in sorted(shades)]


def gray_hex(value):
    """Convert an integer grayscale value to a Tk-compatible #RRGGBB string."""
    value = int(np.clip(value, 0, 255))
    return f"#{value:02x}{value:02x}{value:02x}"


def text_color(gray_value):
    """Choose readable text for grayscale threshold-picker buttons."""
    return "#111111" if gray_value >= 150 else "#f7f7f7"


# ---------------------------------------------------------------------------
# Tkinter interface: image-list navigation + per-image override dictionary
# ---------------------------------------------------------------------------
class DetectorApp:
    """Single-window detector UI with an ordered image list and navigation.

    The detector settings have two layers:

    * ``effective_default_settings()`` returns the session defaults for the
      currently loaded image.  Normally these are the CLI defaults.  The legacy
      ``--max-radius 0`` shorthand is resolved per image to that image's largest
      dimension, so multi-image use does not weaken the old behavior.
    * ``image_overrides`` contains ONLY values that differ from those defaults,
      keyed by absolute image path.  An all-default image has no dictionary
      entry at all.

    When an image is loaded, defaults are restored first, saved overrides are
    overlaid second, then ``apply()`` is invoked exactly once.  Subsequent slider
    edits stay pending until Apply/Enter or another image load occurs.
    """

    def __init__(self, root, image_paths, args):
        self.root = root
        self.args = args

        # Normalize paths once.  This ensures the same physical file uses the
        # same dictionary key whether it arrived from a relative CLI argument or
        # from Tk's absolute-path file chooser.
        self.image_paths = [os.path.abspath(path) for path in image_paths]
        self.current_index = -1
        self.current_path = None
        self.gray = None
        self.palette = []

        # Per-image settings live only when they deviate from defaults, e.g.:
        # {
        #   "C:/eclipse/frame_0123.jpg": {"threshold": 11, "min_coverage": 0.10},
        #   "C:/eclipse/frame_0124.jpg": {"max_error": 0.06},
        # }
        # Reverting every control to defaults removes that image's entry.
        self.image_overrides = {}

        # Variable traces fire even for programmatic Tk variable changes.  This
        # guard prevents restoring saved settings from being mistaken for a user
        # edit and immediately re-writing the override dictionary.
        self.restoring_settings = False

        # Cached previews.  Resizing the Tk window redraws these cached arrays;
        # it does not rerun the expensive detector.
        self.last_threshold_preview = None
        self.last_grayscale_preview = None
        self.threshold_photo = None
        self.grayscale_photo = None
        self.resize_job = None

        # Before an image is loaded, use a harmless placeholder max radius if
        # --max-radius=0.  The exact per-image default is resolved after load.
        initial_defaults = self.effective_default_settings()
        self.threshold = tk.IntVar(value=initial_defaults["threshold"])
        self.min_radius = tk.IntVar(value=round(initial_defaults["min_radius"]))
        self.max_radius = tk.IntVar(value=round(initial_defaults["max_radius"]))
        self.max_error = tk.DoubleVar(value=initial_defaults["max_error"] * 100)
        self.min_coverage = tk.IntVar(
            value=round(initial_defaults["min_coverage"] * 100)
        )

        self.status = tk.StringVar(
            value="Load images or adjust settings, then click Apply."
        )
        self.image_info = tk.StringVar(value="No image loaded")

        root.title("Ellipse / Arc Detector")
        root.minsize(980, 720)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(2, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.build_navigation()
        self.build_controls()
        self.build_previews()

        # Enter deliberately reuses the same Apply path as the button.  Escape
        # closes after capturing any current per-image overrides.  Plain Left/
        # Right are NOT bound globally because those keys should remain usable
        # for fine keyboard adjustment when a Tk Scale has focus.
        root.bind("<Return>", lambda _event: self.apply())
        root.bind("<Escape>", lambda _event: self.close())

        # A CLI list is treated exactly like a list chosen through the dialog:
        # load its first frame, restore settings, and run Apply once.
        if self.image_paths:
            self.load_image_at(0)
        else:
            self.update_navigation_state()

    # ------------------------------------------------------------------
    # Per-image defaults / override dictionary
    # ------------------------------------------------------------------
    def effective_default_settings(self):
        """Return session defaults in the same units consumed by detection.

        ``--max-radius 0`` historically meant "largest image dimension".  With
        an image list the correct interpretation is therefore per image rather
        than once for the whole session.
        """
        min_radius = max(1.0, float(self.args.min_radius))

        if self.args.max_radius == 0:
            if self.gray is not None:
                max_radius = float(max(self.gray.shape))
            else:
                # No image exists yet, so this value is only a temporary UI
                # placeholder.  It will be replaced on first successful load.
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
        }

    def settings_dict(self):
        """Read and sanitize the five user-facing settings from Tk controls."""
        min_radius = max(1.0, float(self.min_radius.get()))
        max_radius = max(1.0, float(self.max_radius.get()))

        # Keep the UI internally valid.  The guard suppresses the trace emitted
        # by max_radius.set(), so this repair is not treated as a second edit.
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
            "min_coverage": float(
                np.clip(self.min_coverage.get() / 100, 0, 1)
            ),
        }

    @staticmethod
    def setting_equal(name, value, default):
        """Compare a current setting with its default without float noise."""
        if name == "threshold":
            return int(value) == int(default)
        return math.isclose(
            float(value),
            float(default),
            rel_tol=0.0,
            abs_tol=1e-9,
        )

    def store_current_overrides(self):
        """Update ``image_overrides`` with only non-default current values."""
        if self.current_path is None:
            return

        current = self.settings_dict()
        defaults = self.effective_default_settings()
        overrides = {
            name: current[name]
            for name in SETTING_NAMES
            if not self.setting_equal(name, current[name], defaults[name])
        }

        if overrides:
            self.image_overrides[self.current_path] = overrides
        else:
            # No redundant all-default dictionaries are retained.
            self.image_overrides.pop(self.current_path, None)

    def restore_settings_for_current_image(self):
        """Restore defaults, then overlay saved settings for this image path."""
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
        finally:
            self.restoring_settings = False

    # ------------------------------------------------------------------
    # Image-list loading and navigation
    # ------------------------------------------------------------------
    def load_images(self):
        """Choose an ordered image list, replace the active list, load item 1."""
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Select eclipse images",
            filetypes=IMAGE_FILE_TYPES,
        )
        if not selected:
            return

        # Preserve pending edits from the old list before replacing it.  The
        # dictionary itself is retained, so reloading the same path later in the
        # session restores its remembered values.
        self.store_current_overrides()
        self.image_paths = [os.path.abspath(path) for path in selected]
        self.current_index = -1
        self.current_path = None
        self.gray = None
        self.palette = []
        self.load_image_at(0)

    def load_image_at(self, index):
        """Load one image, restore its settings, then run Apply exactly once."""
        if not 0 <= index < len(self.image_paths):
            return

        # Capture even unapplied slider changes before leaving the old image.
        if self.current_path is not None:
            self.store_current_overrides()

        path = self.image_paths[index]
        image = cv2.imread(path)

        if image is None:
            # Keep navigation usable even when one selected file cannot be read.
            # The failed entry remains in the list so Previous/Next ordering is
            # not silently changed behind the user's back.
            self.current_index = index
            self.current_path = path
            self.gray = None
            self.palette = []
            self.last_threshold_preview = None
            self.last_grayscale_preview = None
            self.threshold_canvas.delete("all")
            self.grayscale_canvas.delete("all")
            self.placeholder(self.threshold_canvas, "Threshold preview")
            self.placeholder(self.grayscale_canvas, "Unreadable image")
            self.update_navigation_state()
            self.status.set(f"Could not load image: {path}")
            return

        self.current_index = index
        self.current_path = path
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold suggestions are image-specific, so regenerate them from all
        # grayscale pixels every time a different image is loaded.
        self.palette = build_palette(
            self.gray,
            self.args.palette_size,
            self.args.palette_min_gap,
        )

        # Restore detector controls BEFORE automatic Apply.  This ordering is the
        # key to getting the same fitted result when returning to a tuned frame.
        self.restore_settings_for_current_image()
        self.update_radius_scale_limits()
        self.rebuild_palette_buttons()

        # Never leave the previous frame's threshold result visible.  The new raw
        # grayscale frame can be shown immediately while Apply begins.
        self.last_threshold_preview = None
        self.last_grayscale_preview = cv2.cvtColor(
            self.gray,
            cv2.COLOR_GRAY2BGR,
        )
        self.threshold_photo = None
        self.threshold_canvas.delete("all")
        self.placeholder(self.threshold_canvas, "Threshold preview")
        self.redraw()

        self.update_navigation_state()
        self.status.set(
            "Image loaded; running detection with restored settings..."
        )
        self.root.update_idletasks()

        # Requirement: every successful image load executes the existing Apply
        # recomputation behavior exactly once, including startup and navigation.
        self.apply()

    def previous_image(self):
        """Load the preceding list entry; stop at the first image (no wrap)."""
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def next_image(self):
        """Load the following list entry; stop at the last image (no wrap)."""
        if 0 <= self.current_index < len(self.image_paths) - 1:
            self.load_image_at(self.current_index + 1)

    def update_navigation_state(self):
        """Enable/disable arrows and update the current image counter/name."""
        count = len(self.image_paths)
        has_current = 0 <= self.current_index < count
        has_readable_image = has_current and self.gray is not None

        self.previous_button.config(
            state=(
                tk.NORMAL
                if has_current and self.current_index > 0
                else tk.DISABLED
            )
        )
        self.next_button.config(
            state=(
                tk.NORMAL
                if has_current and self.current_index < count - 1
                else tk.DISABLED
            )
        )
        self.apply_button.config(
            state=tk.NORMAL if has_readable_image else tk.DISABLED
        )

        if has_current:
            filename = os.path.basename(self.current_path)
            self.image_info.set(
                f"{self.current_index + 1} / {count}   {filename}"
            )
        elif count:
            self.image_info.set(f"0 / {count}")
        else:
            self.image_info.set("No image loaded")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def build_navigation(self):
        """Create multi-file loading and Previous/Next arrow controls."""
        # Keep tuple padding in grid(), not the Frame constructor.  This is
        # important for Windows Tk, where widget constructor pady=(8, 0) can be
        # parsed as the invalid screen-distance string "8 0".
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        frame.columnconfigure(3, weight=1)

        tk.Button(
            frame,
            text="Load images...",
            width=14,
            command=self.load_images,
        ).grid(row=0, column=0, padx=(0, 8))

        self.previous_button = tk.Button(
            frame,
            text="◀ Previous",
            width=12,
            command=self.previous_image,
        )
        self.previous_button.grid(row=0, column=1, padx=(0, 5))

        self.next_button = tk.Button(
            frame,
            text="Next ▶",
            width=12,
            command=self.next_image,
        )
        self.next_button.grid(row=0, column=2, padx=(0, 10))

        tk.Label(
            frame,
            textvariable=self.image_info,
            anchor="w",
        ).grid(row=0, column=3, sticky="ew")

    def build_controls(self):
        """Create sliders, threshold palette, Apply, and status text."""
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=1, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        defaults = self.effective_default_settings()
        radius_limit = max(1600, round(defaults["max_radius"] * 1.5))

        # Keep references to the two radius Scales because max-radius=0 can make
        # their useful range image-dependent as the user navigates.
        self.radius_scales = {}

        rows = [
            (
                "threshold",
                "Brightness threshold (0=black, 255=white)",
                self.threshold,
                0,
                255,
                1,
                lambda value: str(int(value)),
            ),
            (
                "min_radius",
                "Minimum fitted semi-axis radius (px)",
                self.min_radius,
                1,
                radius_limit,
                1,
                lambda value: f"{int(value)} px",
            ),
            (
                "max_radius",
                "Maximum fitted semi-axis radius (px)",
                self.max_radius,
                1,
                radius_limit,
                1,
                lambda value: f"{int(value)} px",
            ),
            (
                "max_error",
                "Maximum average normalized ellipse error (%)",
                self.max_error,
                0.5,
                50,
                0.1,
                lambda value: f"{float(value):.1f}%",
            ),
            (
                "min_coverage",
                "Minimum visible ellipse arc (%)",
                self.min_coverage,
                0,
                100,
                1,
                lambda value: (
                    f"{int(value)}% (~{int(value) * 3.6:.0f}°)"
                ),
            ),
        ]

        for row, spec in enumerate(rows):
            name, *scale_args = spec
            scale = self.add_scale(frame, row, *scale_args)
            if name in ("min_radius", "max_radius"):
                self.radius_scales[name] = scale

        tk.Label(
            frame,
            text="Pick threshold from grayscale tones:",
        ).grid(row=5, column=0, sticky="nw")

        self.palette_frame = tk.Frame(frame)
        self.palette_frame.grid(
            row=5,
            column=1,
            columnspan=2,
            sticky="w",
            pady=(0, 8),
        )

        self.apply_button = tk.Button(
            frame,
            text="Apply",
            width=12,
            command=self.apply,
        )
        self.apply_button.grid(
            row=6,
            column=0,
            sticky="w",
            pady=(2, 0),
        )

        tk.Label(
            frame,
            textvariable=self.status,
            anchor="w",
            justify="left",
            wraplength=1100,
        ).grid(
            row=7,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 0),
        )

    def add_scale(
        self,
        parent,
        row,
        text,
        variable,
        low,
        high,
        resolution,
        formatter,
    ):
        """Create one labeled slider and return the Scale for later updates."""
        tk.Label(
            parent,
            text=text,
            width=38,
            anchor="w",
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=2,
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

        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            value_label.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update_value)
        update_value()
        return scale

    def update_radius_scale_limits(self):
        """Expand radius slider ranges when the active image/default requires it."""
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
        """Rebuild grayscale threshold suggestions for the active image."""
        for child in self.palette_frame.winfo_children():
            child.destroy()

        for index, shade in enumerate(self.palette):
            tk.Button(
                self.palette_frame,
                text=str(shade),
                width=4,
                bg=gray_hex(shade),
                fg=text_color(shade),
                command=lambda value=shade: self.pick(value),
            ).grid(
                row=index // 10,
                column=index % 10,
                padx=2,
                pady=2,
            )

    def build_previews(self):
        """Create the two aspect-preserving image preview canvases."""
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(
            frame,
            text="Threshold preview with detected arcs and ellipses",
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        tk.Label(
            frame,
            text="Grayscale image",
        ).grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.threshold_canvas = tk.Canvas(
            frame,
            bg="#202020",
            highlightthickness=1,
            highlightbackground="#808080",
        )
        self.grayscale_canvas = tk.Canvas(
            frame,
            bg="#202020",
            highlightthickness=1,
            highlightbackground="#808080",
        )

        self.threshold_canvas.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 5),
        )
        self.grayscale_canvas.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(5, 0),
        )

        # Canvas resize events redraw cached previews only; they never recompute
        # contours, candidates, or ellipse fits.
        self.threshold_canvas.bind("<Configure>", self.schedule_redraw)
        self.grayscale_canvas.bind("<Configure>", self.schedule_redraw)

        self.placeholder(self.threshold_canvas, "Threshold preview")
        self.placeholder(self.grayscale_canvas, "Grayscale image")

    # ------------------------------------------------------------------
    # User actions and Apply-based recomputation
    # ------------------------------------------------------------------
    def pick(self, value):
        """Set a suggested grayscale threshold without rerunning detection."""
        self.threshold.set(value)
        self.status.set(
            f"Threshold set to {value}. Click Apply to recompute."
        )

    def pending(self, *_args):
        """Store a control edit for this image, but do not rerun detection."""
        if self.restoring_settings:
            return

        if self.current_path is not None:
            # Keep the dictionary synchronized even before Apply.  Therefore an
            # unapplied adjustment survives an immediate Previous/Next click.
            self.store_current_overrides()
            self.status.set(
                "Settings changed and remembered for this image. "
                "Click Apply to recompute the result."
            )

    def settings(self):
        """Return the current settings tuple in the order expected by detection."""
        values = self.settings_dict()
        return tuple(values[name] for name in SETTING_NAMES)

    def apply(self):
        """Run detection once on the current image with the current controls."""
        if self.gray is None or self.current_path is None:
            self.status.set(
                "Load at least one readable image before applying detection."
            )
            return

        threshold, min_radius, max_radius, max_error, min_coverage = self.settings()

        # settings_dict() may sanitize min/max radius, so store after reading it.
        self.store_current_overrides()

        started = time.perf_counter()
        threshold_preview, grayscale_preview, ellipses = process_image(
            self.gray,
            threshold,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
            self.args.max_contours,
            self.args.max_search_points,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.last_threshold_preview = threshold_preview
        self.last_grayscale_preview = grayscale_preview
        self.redraw()

        override_count = len(
            self.image_overrides.get(self.current_path, {})
        )
        self.status.set(
            f"Applied: T={threshold}; "
            f"semi-axes={min_radius:.0f}-{max_radius:.0f}px; "
            f"error={max_error:.1%}; arc={min_coverage:.0%}; "
            f"{len(ellipses)} ellipse(s); {elapsed_ms:.1f} ms; "
            f"{override_count} per-image override(s)."
        )

        # Keep console diagnostics for difficult eclipse frames without adding
        # extra GUI controls.  Prefix each line with the filename so sequential
        # navigation logs remain unambiguous.
        for ellipse in ellipses:
            print(
                f"{os.path.basename(self.current_path)} | "
                f"{ellipse['class']}: "
                f"center=({ellipse['center'][0]:.2f}, "
                f"{ellipse['center'][1]:.2f}), "
                f"a={ellipse['major']:.2f}, "
                f"b={ellipse['minor']:.2f}, "
                f"angle={ellipse['angle']:.1f}, "
                f"arc={ellipse['coverage'] * 360:.1f}°, "
                f"error={ellipse['relative_error']:.4f}, "
                f"interior={ellipse['interior_fraction']:.1%}"
            )

    def schedule_redraw(self, _event=None):
        """Debounce resize events so cached-image redraws do not thrash Tk."""
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        """Resize cached images to current canvases without rerunning detection."""
        self.resize_job = None

        if self.last_threshold_preview is not None:
            self.threshold_photo = self.show_image(
                self.threshold_canvas,
                self.last_threshold_preview,
            )

        if self.last_grayscale_preview is not None:
            self.grayscale_photo = self.show_image(
                self.grayscale_canvas,
                self.last_grayscale_preview,
            )

    @staticmethod
    def show_image(canvas, image):
        """Fit an OpenCV image inside a Tk canvas while preserving aspect ratio."""
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]

        scale = max(
            min(canvas_width / image_width, canvas_height / image_height),
            1e-6,
        )
        fitted_size = (
            max(1, round(image_width * scale)),
            max(1, round(image_height * scale)),
        )

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        fitted = cv2.resize(
            image,
            fitted_size,
            interpolation=interpolation,
        )

        # OpenCV handles resize and PNG encoding, so Pillow remains unnecessary.
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
        """Show initial guidance when no computed preview is available."""
        canvas.create_text(
            160,
            120,
            text=text + "\nClick Apply",
            fill="#cccccc",
            justify="center",
        )

    def close(self):
        """Capture pending per-image settings, then close the Tk session."""
        self.store_current_overrides()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------
def build_parser():
    """Create CLI options; positional images may now contain an ordered list."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect up to two inferred ellipses from thresholded image arcs."
        )
    )
    parser.add_argument(
        "images",
        nargs="*",
        help=(
            "Optional ordered input image list; more images can be loaded "
            "through the Tk interface"
        ),
    )
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument(
        "--min-radius",
        type=float,
        default=1000.0,
        help="Minimum allowed value for each ellipse semi-axis",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=1500.0,
        help=(
            "Maximum allowed value for each ellipse semi-axis; "
            "0 = largest dimension of the active image"
        ),
    )
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    parser.add_argument("--max-contours", type=int, default=100)
    parser.add_argument("--max-search-points", type=int, default=500)
    parser.add_argument("--palette-size", type=int, default=20)
    parser.add_argument("--palette-min-gap", type=int, default=10)
    return parser


def validate_args(args, parser):
    """Fail early on invalid detector defaults before opening the Tk window."""
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
    if args.palette_size < 1 or not 1 <= args.palette_min_gap <= 255:
        parser.error("invalid palette settings")


def main():
    """Start the Tk UI with zero, one, or many initial image paths."""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    root = tk.Tk()
    DetectorApp(root, args.images, args)
    root.mainloop()


if __name__ == "__main__":
    main()
