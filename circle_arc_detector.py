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
5. Propose a horizon directly from a globally straight, sharp threshold border
   before ellipse fitting. Horizon orientation is unrestricted. Solar-bright
   pixels must lie on one half-plane and the opposite side must remain dark at
   the border; curved ellipse fits are then tested with that line excluded.
6. Grow each ellipse around all 360 degrees and retain *multiple disconnected
   real support segments*.  Gaps remain gaps and do not count toward coverage.
   For a provisional horizon, only light/Sun evidence is restricted to the visible
   half-plane; dark/Moon evidence merely excludes the straight horizon band.
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
from dataclasses import dataclass
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

# Prefer the simplest/roundest geometric model unless the extra ellipse degrees
# of freedom produce a materially better normalized radial fit on the *same*
# support points.  This stabilizes short or noisy eclipse arcs, which can make a
# free ellipse look more eccentric than the physical limb really is.
ROUNDNESS_MIN_ABS_ERROR_IMPROVEMENT = 0.0015
ROUNDNESS_MIN_REL_ERROR_IMPROVEMENT = 0.10

# Intermediate ellipse hypotheses are deliberately allowed outside the strict
# final user range.  Final candidates must still pass the exact selected limits.
INTERMEDIATE_MIN_FACTOR = 0.65
INTERMEDIATE_MAX_FACTOR = 1.35

# Reduced-resolution working sizes.  Automatic threshold initialization uses a
# very small image because it needs only brightness/color statistics; it never
# runs contour, ellipse, arc, or horizon detection.
PREVIEW_MAX_DIM = 2400
AUTO_THRESHOLD_MAX_DIM = 600

# Horizon geometry validation.  The raster threshold remains authoritative.
HORIZON_EXCLUSION_RADIUS = 0.012

# Solar-component guidance margins are expressed in expected solar radii.  They
# restrict *where candidate evidence is searched*; they never clip the final
# ellipse, alter the authoritative threshold mask, or relax final validation.
LIGHT_COMPONENT_GUIDANCE_RADIUS = 0.25
HORIZON_COMPONENT_GUIDANCE_RADIUS = 0.75


# Horizon proposals are found before ellipse fitting. These values express
# threshold-border quality rather than a minimum horizon length. Straightness is
# normalized by the sagitta a circular solar limb would have over the same span;
# it is used only for proposal ranking.
HORIZON_PROPOSAL_MAX_WRONG_BRIGHT = 0.08
HORIZON_PROPOSAL_MAX_DARK_GAP = 0.45
HORIZON_PROPOSAL_MIN_TRANSITION = 0.28
HORIZON_PROPOSAL_MIN_BRIGHT_SIDE = 0.30
HORIZON_PROPOSAL_MAX_DARK_SIDE = 0.35
HORIZON_SHARPNESS_MIN_LEVELS = 4.0
HORIZON_SHARPNESS_NOISE_MULTIPLIER = 3.0
HORIZON_PROPOSAL_LIMIT = 6

# Small morphology is guidance-only.  It can help contour discovery but can
# never redefine the final measured limb.
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
NON_THRESHOLD_SETTING_NAMES = (
    "min_radius",
    "max_radius",
    "max_error",
    "min_coverage",
    "morphology",
    "outer_limb_assistance",
    "use_horizon",
)

IMAGE_FILE_TYPES = (
    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
    ("All files", "*.*"),
)


@dataclass(frozen=True)
class DetectionSettings:
    """Geometry/search settings shared by the detector pipeline."""

    min_radius: float
    max_radius: float
    max_error: float
    min_coverage: float
    max_contours: int
    max_search_points: int
    morphology: bool = False
    outer_limb_assistance: bool = False



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


def axes_in_range(ellipse, settings):
    """Strict FINAL constraint: both semi-axis radii must be inside the range."""
    return (
        settings.min_radius <= ellipse["major"] <= settings.max_radius
        and settings.min_radius <= ellipse["minor"] <= settings.max_radius
    )


def axes_in_intermediate_range(ellipse, settings):
    """Loose search envelope used before multi-segment support is recovered."""
    search_min = max(1.0, settings.min_radius * INTERMEDIATE_MIN_FACTOR)
    search_max = max(settings.max_radius, settings.min_radius) * INTERMEDIATE_MAX_FACTOR
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
    """Return a circle-constrained fit first, then free ellipse fits.

    The circle is deliberately evaluated first.  Model selection later compares
    all fits on the same support points and keeps the rounder one unless the extra
    ellipse degrees of freedom improve normalized radial residual significantly.
    """
    points = np.asarray(points, np.float32)
    if len(points) < 5:
        return []

    options = []
    circle = fit_circle_prior(points)
    if circle is not None:
        options.append(circle)

    shaped = points.reshape(-1, 1, 2)
    for fitter in (cv2.fitEllipseDirect, cv2.fitEllipseAMS, cv2.fitEllipse):
        try:
            ellipse = normalize_ellipse(fitter(shaped))
        except cv2.error:
            ellipse = None
        if ellipse is not None:
            options.append(ellipse)
    return options


def ellipse_eccentricity_fraction(ellipse):
    """Simple 0..1 eccentricity proxy used only for model preference."""
    return max(
        0.0,
        1.0 - ellipse["minor"] / max(ellipse["major"], 1e-9),
    )


def fit_improvement_is_significant(baseline_error, improved_error):
    """Whether a more flexible model materially improves radial residual."""
    if not (np.isfinite(baseline_error) and np.isfinite(improved_error)):
        return False
    improvement = baseline_error - improved_error
    required = max(
        ROUNDNESS_MIN_ABS_ERROR_IMPROVEMENT,
        ROUNDNESS_MIN_REL_ERROR_IMPROVEMENT * max(baseline_error, 0.0),
    )
    return improvement > required


def prefer_rounder_fit(candidates):
    """Choose the roundest fit whose residual is effectively best.

    ``candidates`` must describe fits evaluated on the same support points and
    contain ``relative_error``.  First find the lowest residual.  Any candidate
    for which that best residual is *not* a significant improvement is considered
    statistically equivalent for this detector, and the roundest equivalent model
    wins.  Existing detector score is used only as a final tie-breaker.
    """
    if not candidates:
        return None

    best_error = min(item["relative_error"] for item in candidates)
    equivalent = [
        item for item in candidates
        if not fit_improvement_is_significant(
            item["relative_error"], best_error
        )
    ]
    return min(
        equivalent,
        key=lambda item: (
            ellipse_eccentricity_fraction(item),
            item["relative_error"],
            -item.get("score", 0.0),
        ),
    )


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


def evaluate_region(extended_points, point_count, start, length, original_mask, settings):
    """Fit one cyclic contour window using relaxed intermediate axis limits."""
    if length < 5 or length > point_count:
        return None

    start %= point_count
    region = extended_points[start:start + length]

    # A window that is geometrically too small to span even the shortest
    # admissible intermediate arc cannot produce a valid fit.  Reject it before
    # invoking four comparatively expensive circle/ellipse fitters.  The 0.60
    # margin is deliberately conservative for noisy points and the relaxed
    # intermediate residual allowance.
    search_min = max(1.0, settings.min_radius * INTERMEDIATE_MIN_FACTOR)
    required_coverage = max(0.02, settings.min_coverage * 0.65)
    minimum_span = (
        0.60
        * 2.0
        * search_min
        * math.sin(math.pi * min(required_coverage, 0.5))
    )
    extent = np.ptp(region, axis=0)
    if math.hypot(float(extent[0]), float(extent[1])) < minimum_span:
        return None

    valid_candidates = []
    for ellipse in fit_ellipse_options(region):
        if not axes_in_intermediate_range(ellipse, settings):
            continue

        relative_error, coverage = ellipse_error_and_coverage(region, ellipse)
        # Intermediate residual can be somewhat looser; final growth/refit will
        # reapply the exact selected error threshold.
        if relative_error > max(settings.max_error * 1.5, settings.max_error + 0.01):
            continue
        if coverage < max(0.02, settings.min_coverage * 0.65):
            continue

        candidate = {
            **ellipse,
            "relative_error": relative_error,
            "coverage": coverage,
            "points": region.copy(),
            "arc_segments": [region.copy()],
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
        candidate["score"] = geometry_score * polarity**2

        valid_candidates.append(candidate)

    return prefer_rounder_fit(valid_candidates)


def refine_region(candidate, extended_points, point_count, minimum_length,
                  original_mask, settings):
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
                settings,
            )
            if trial is not None and trial["score"] > best["score"] * (1 + 1e-9):
                best = trial
                improved = True
        if not improved:
            break
    return best


def find_regions(points, original_mask, settings, minimum=12):
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
                settings,
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
            settings,
        )
        if not any(same_ellipse(candidate, old) for old in refined):
            refined.append(candidate)

    refined.sort(key=lambda item: item["score"], reverse=True)
    return refined[:MAX_REGIONS_PER_CONTOUR]


def morphology_guidance(mask):
    """Return a cleaned contour-discovery copy; never used for final validation."""
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return cleaned


def component_for_hint(mask, center):
    """Return the threshold component containing ``center`` plus metadata.

    The returned component is a binary uint8 mask in full-image coordinates.
    Nothing here assumes that the component is a complete Sun: at sunset it may
    be only a small clipped fragment.  ``touches_border`` is intentionally
    reported because background-connected components are not safe guidance for
    automatic thresholding or candidate restriction.
    """
    if center is None or mask is None or mask.size == 0:
        return None

    binary = (mask != 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None

    height, width = mask.shape
    x = int(np.clip(round(center[0]), 0, width - 1))
    y = int(np.clip(round(center[1]), 0, height - 1))
    label = int(labels[y, x])
    if label <= 0:
        return None

    left, top, component_width, component_height, area = map(
        int, stats[label]
    )
    touches_border = (
        left == 0
        or top == 0
        or left + component_width >= width
        or top + component_height >= height
    )
    component = np.where(labels == label, 255, 0).astype(np.uint8)
    return {
        "mask": component,
        "label": label,
        "bbox": (left, top, component_width, component_height),
        "area": area,
        "area_fraction": area / max(mask.size, 1),
        "centroid": (
            float(centroids[label, 0]),
            float(centroids[label, 1]),
        ),
        "touches_border": bool(touches_border),
        "bright_fraction": float(np.count_nonzero(binary)) / max(mask.size, 1),
        "bright_dominance": area / max(int(np.count_nonzero(binary)), 1),
    }


def dilated_component_mask(component_mask, margin):
    """Return a rounded distance dilation without inventing physical edges.

    A distance transform is evaluated only in the component bounding box plus
    the requested margin, avoiding a very large morphology kernel and unnecessary
    full-frame work.  The ROI is guidance-only: callers must continue to sample
    and validate against the original threshold/grayscale images.
    """
    component = (component_mask != 0).astype(np.uint8)
    if not np.any(component):
        return np.zeros_like(component_mask, dtype=np.uint8)

    ys, xs = np.nonzero(component)
    height, width = component.shape
    padding = max(0, int(math.ceil(float(margin))))
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(width, int(xs.max()) + padding + 1)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(height, int(ys.max()) + padding + 1)

    crop = component[y0:y1, x0:x1]
    if padding == 0:
        dilated_crop = crop != 0
    else:
        # distanceTransform measures non-zero pixels to the nearest zero.  Making
        # the observed component zero therefore gives every outside pixel its
        # Euclidean distance to real solar evidence.
        outside = np.where(crop != 0, 0, 255).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
        dilated_crop = distance <= float(margin)

    result = np.zeros_like(component_mask, dtype=np.uint8)
    result[y0:y1, x0:x1] = np.where(dilated_crop, 255, 0).astype(np.uint8)
    return result


def component_guidance(mask, center, margin):
    """Return solar component metadata plus a rounded search-guidance mask."""
    info = component_for_hint(mask, center)
    if info is None:
        return None
    result = dict(info)
    result["guidance_mask"] = dilated_component_mask(info["mask"], margin)
    result["guidance_fraction"] = (
        float(np.count_nonzero(result["guidance_mask"])) / max(mask.size, 1)
    )
    return result


def contour_runs_for_horizon(points, horizon, margin, minimum_points,
                             visible_side_only=False):
    """Return contiguous real contour runs allowed by a horizon proposal.

    The raster mask is never edited.  A narrow band around the straight proposal
    is always excluded so that the horizon itself cannot seed a curved fit.  For
    light/Sun fitting, ``visible_side_only`` additionally rejects every contour
    point on the proposal's dark half-plane.  The ellipse algorithm itself is
    otherwise unchanged and sees only the physically eligible evidence.
    """
    points = np.asarray(points, np.float64)
    if horizon is None or len(points) == 0:
        return [points] if len(points) >= minimum_points else []

    signed = line_signed_distance(points, horizon)
    keep = np.abs(signed) > margin
    if visible_side_only:
        keep &= signed * horizon["visible_sign"] > margin

    count = len(points)
    if not np.any(keep):
        return []
    if np.all(keep):
        return [points] if count >= minimum_points else []

    start = (int(np.flatnonzero(~keep)[0]) + 1) % count
    order = (start + np.arange(count)) % count
    values = keep[order]
    runs = []
    index = 0
    while index < count:
        if not values[index]:
            index += 1
            continue
        end = index
        while end < count and values[end]:
            end += 1
        run = points[order[index:end]]
        if len(run) >= minimum_points:
            runs.append(run)
        index = end
    return runs


def outer_limb_seed_from_contour(contour, original_mask, settings):
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
        if not axes_in_intermediate_range(ellipse, settings):
            continue
        relative_error, coverage = ellipse_error_and_coverage(points, ellipse)
        if relative_error > max(settings.max_error * 1.5, settings.max_error + 0.01):
            continue

        candidate = {
            **ellipse,
            "relative_error": relative_error,
            "coverage": coverage,
            "points": points.copy(),
            "arc_segments": [points.copy()],
            "largest_segment_coverage": coverage,
            "segment_count": 1,
        }
        polarity, _inside_fraction, sample_count = boundary_polarity(
            original_mask, candidate
        )
        if sample_count < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue
        candidate["boundary_polarity"] = polarity
        candidate["score"] = (
            max(coverage, settings.min_coverage * 0.5)
            * math.sqrt(len(points))
            * polarity**2
            / (relative_error + 0.003)
        )
        candidates.append(candidate)
    return candidates


def find_candidates(original_mask, settings, *, horizon=None,
                    visible_side_only=False, allow_outer_limb=True):
    """Discover ellipse seeds from real contour evidence on the original mask.

    A horizon proposal may remove its straight line from contour evidence.  When
    fitting the light/Sun class, ``visible_side_only`` also removes the proposal's
    dark half-plane.  No pixels are altered and the normal ellipse-fitting code is
    reused unchanged on the remaining evidence.
    """
    guidance_mask = (
        morphology_guidance(original_mask) if settings.morphology else original_mask
    )
    contours, _ = cv2.findContours(
        guidance_mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    minimum_points = 12
    min_perimeter = max(
        12.0, settings.min_radius * settings.min_coverage * math.pi * 0.6
    )
    usable = []
    for contour in contours:
        if len(contour) < minimum_points:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter >= min_perimeter:
            usable.append((perimeter, contour))

    usable.sort(key=lambda item: item[0], reverse=True)
    if settings.max_contours > 0:
        usable = usable[:settings.max_contours]

    exclusion_margin = max(
        2.0,
        HORIZON_EXCLUSION_RADIUS * 0.5 * (settings.min_radius + settings.max_radius),
    )
    candidates = []
    for _, contour in usable:
        contour_points = contour.reshape(-1, 2).astype(np.float64, copy=False)
        point_sets = contour_runs_for_horizon(
            contour_points,
            horizon,
            exclusion_margin,
            minimum_points,
            visible_side_only=visible_side_only,
        )
        for points in point_sets:
            if settings.max_search_points > 0 and len(points) > settings.max_search_points:
                indices = np.linspace(
                    0, len(points) - 1, settings.max_search_points, dtype=np.int32
                )
                points = points[indices]

            candidates.extend(
                find_regions(
                    points,
                    original_mask,
                    settings,
                    minimum_points,
                )
            )

        # During horizon-constrained discovery the convex envelope can reconnect
        # points across the deliberately removed line gap. Skip that optional seed
        # source rather than reintroducing the horizon through a hull chord.
        if settings.outer_limb_assistance and allow_outer_limb and horizon is None:
            candidates.extend(
                outer_limb_seed_from_contour(
                    contour,
                    original_mask,
                    settings,
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


def visible_fraction(mask, candidate, exclude=None, horizon=None):
    """Fraction of eligible ellipse interior matching ``mask``.

    A provisional/active horizon may restrict this statistic to the Sun-visible
    half-plane.  This keeps dark-side pixels out of light-ellipse selection just
    as they are kept out of its contour and arc-support evidence.
    """
    height, width = mask.shape
    x0, y0, x1, y1 = ellipse_bounds(candidate, width, height)
    if x0 >= x1 or y0 >= y1:
        return 0.0, 0

    yy, xx = np.ogrid[y0:y1, x0:x1]
    valid = ellipse_inside(xx, yy, candidate)
    if exclude is not None:
        valid &= ~ellipse_inside(xx, yy, exclude)
    if horizon is not None:
        signed = xx * horizon["normal"][0] + yy * horizon["normal"][1] + horizon["c"]
        valid &= signed * horizon["visible_sign"] > 0.0

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


def _straight_edge_run(edge_points, seed, expected_radius):
    """Expand one Hough seed into its complete contiguous straight edge run."""
    x1, y1, x2, y2 = map(float, seed)
    direction = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    seed_length = float(np.linalg.norm(direction))
    if seed_length < 4.0:
        return None
    direction /= seed_length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    c = -float(normal @ np.array([x1, y1], dtype=np.float64))

    distance = np.abs(edge_points @ normal + c)
    projection = edge_points @ direction
    seed_projection = np.array([[x1, y1], [x2, y2]], dtype=np.float64) @ direction
    seed_min = float(seed_projection.min())
    seed_max = float(seed_projection.max())
    near = (
        (distance <= max(2.0, 0.008 * expected_radius))
        & (projection >= seed_min - max(5.0, 0.05 * expected_radius))
        & (projection <= seed_max + max(5.0, 0.05 * expected_radius))
    )
    points = edge_points[near]
    if len(points) < 6:
        return None

    fitted = _fit_line_to_points(points)
    if fitted is None:
        return None
    (vx, vy), (nx, ny), c = fitted
    direction = np.array([vx, vy], dtype=np.float64)
    normal = np.array([nx, ny], dtype=np.float64)

    # Keep the contiguous projection cluster that contains the original seed so
    # separate collinear scene edges do not get fused into one horizon proposal.
    distance = np.abs(edge_points @ normal + c)
    projection = edge_points @ direction
    indices = np.flatnonzero(distance <= max(2.0, 0.010 * expected_radius))
    if len(indices) < 6:
        return None
    values = projection[indices]
    order = np.argsort(values)
    values = values[order]
    indices = indices[order]
    gap = max(4.0, 0.025 * expected_radius)
    cuts = np.where(np.diff(values) > gap)[0] + 1
    groups = [group for group in np.split(indices, cuts) if len(group) >= 6]
    if not groups:
        return None

    seed_mid = 0.5 * float(seed_projection.sum())
    group = min(
        groups,
        key=lambda ids: abs(float(np.median(edge_points[ids] @ direction)) - seed_mid),
    )
    points = edge_points[group]

    # Two expansion/refit passes reach the complete available straight stretch
    # while still respecting the projection gap between separate objects.
    for _ in range(2):
        fitted = _fit_line_to_points(points)
        if fitted is None:
            return None
        (vx, vy), (nx, ny), c = fitted
        direction = np.array([vx, vy], dtype=np.float64)
        normal = np.array([nx, ny], dtype=np.float64)
        distance = np.abs(edge_points @ normal + c)
        projection = edge_points @ direction
        run_projection = points @ direction
        low = float(run_projection.min())
        high = float(run_projection.max())
        take = (
            (distance <= max(2.0, 0.012 * expected_radius))
            & (projection >= low - gap)
            & (projection <= high + gap)
        )
        expanded = edge_points[take]
        if len(expanded) >= len(points):
            points = expanded

    fitted = _fit_line_to_points(points)
    if fitted is None:
        return None
    (vx, vy), (nx, ny), c = fitted
    direction = np.array([vx, vy], dtype=np.float64)
    normal = np.array([nx, ny], dtype=np.float64)
    signed = points @ normal + c
    residual = float(np.sqrt(np.mean(signed**2)))
    run_projection = points @ direction
    low = float(run_projection.min())
    high = float(run_projection.max())
    span = high - low
    if span < 6.0:
        return None

    return direction, normal, float(c), residual, low, high, span


def _sample_binary_side(mask, base, normal, offset, sign, multiplier=1.0):
    """Sample one side of a proposed line and return coordinates/value validity."""
    points = np.rint(
        base + sign * offset * multiplier * normal[None, :]
    ).astype(np.int32)
    height, width = mask.shape
    valid = (
        (points[:, 0] >= 0) & (points[:, 0] < width)
        & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    values = np.zeros(len(base), dtype=bool)
    values[valid] = mask[points[valid, 1], points[valid, 0]] != 0
    return points, values, valid


def _horizon_line_metrics(gray, light_mask, solar_pixels, edge_points,
                          seed, expected_radius, hsv_image=None):
    """Measure physical horizon evidence for one Hough line seed."""
    run = _straight_edge_run(edge_points, seed, expected_radius)
    if run is None:
        return None
    direction, normal, c, residual, low, high, span = run

    sample_count = int(np.clip(math.ceil(span / 2.0), 12, 500))
    t = np.linspace(low, high, sample_count)
    p0 = -c * normal
    base = p0[None, :] + t[:, None] * direction[None, :]
    sample_offset = max(2.0, 0.012 * expected_radius)

    plus_points, plus, plus_valid = _sample_binary_side(
        light_mask, base, normal, sample_offset, 1.0
    )
    minus_points, minus, minus_valid = _sample_binary_side(
        light_mask, base, normal, sample_offset, -1.0
    )
    valid = plus_valid & minus_valid
    if np.count_nonzero(valid) < 8:
        return None

    plus_fraction = float(np.mean(plus[valid]))
    minus_fraction = float(np.mean(minus[valid]))
    visible_sign = 1.0 if plus_fraction >= minus_fraction else -1.0
    bright = plus if visible_sign > 0 else minus
    dark = minus if visible_sign > 0 else plus
    bright_points = plus_points if visible_sign > 0 else minus_points
    dark_points = minus_points if visible_sign > 0 else plus_points

    transition = float(np.mean(bright[valid] & ~dark[valid]))
    dark_gap = float(np.mean(~bright[valid] & ~dark[valid]))
    bright_fraction = float(np.mean(bright[valid]))
    dark_fraction = float(np.mean(dark[valid]))

    _, plus2, plus_valid2 = _sample_binary_side(
        light_mask, base, normal, sample_offset, 1.0, 2.0
    )
    _, minus2, minus_valid2 = _sample_binary_side(
        light_mask, base, normal, sample_offset, -1.0, 2.0
    )
    valid2 = plus_valid2 & minus_valid2
    bright2 = plus2 if visible_sign > 0 else minus2
    dark2 = minus2 if visible_sign > 0 else plus2
    transition2 = (
        float(np.mean(bright2[valid2] & ~dark2[valid2])) if np.any(valid2) else 0.0
    )
    dark_gap2 = (
        float(np.mean(~bright2[valid2] & ~dark2[valid2])) if np.any(valid2) else 1.0
    )

    # Global physical criterion: essentially all threshold-bright solar pixels
    # must lie on one side. Dark pixels may still occur on the visible side.
    solar_distance = solar_pixels @ normal + c
    wrong_bright = (
        float(np.mean(
            solar_distance * visible_sign < -max(1.0, 0.006 * expected_radius)
        ))
        if len(solar_pixels) else 1.0
    )

    # A binary contour can also arise from a smooth red-disk intensity gradient.
    # Require a real grayscale jump across the same border relative to local noise.
    gray_bright = gray[
        bright_points[valid, 1], bright_points[valid, 0]
    ].astype(np.float64)
    gray_dark = gray[
        dark_points[valid, 1], dark_points[valid, 0]
    ].astype(np.float64)
    gray_contrast = (
        float(np.median(gray_bright - gray_dark)) if len(gray_bright) else 0.0
    )
    local_noise = 0.0
    if len(gray_bright) > 2:
        local_noise = max(
            float(np.median(np.abs(np.diff(gray_bright)))),
            float(np.median(np.abs(np.diff(gray_dark)))),
        )
    sharpness_floor = max(
        HORIZON_SHARPNESS_MIN_LEVELS,
        HORIZON_SHARPNESS_NOISE_MULTIPLIER * local_noise,
    )
    sharp = gray_contrast >= sharpness_floor

    # Straightness is a ranking diagnostic, normalized by the sagitta that a
    # circular solar limb would have over the same observed chord.
    chord = min(span, 1.98 * expected_radius)
    if chord < 2.0 * expected_radius:
        sagitta = max(
            0.25,
            expected_radius - math.sqrt(max(
                0.0, expected_radius**2 - (0.5 * chord)**2
            )),
        )
    else:
        sagitta = expected_radius
    straightness = residual / sagitta

    warm_fraction = 0.0
    if hsv_image is not None and np.any(valid):
        q = bright_points[valid]
        hue = hsv_image[q[:, 1], q[:, 0], 0]
        saturation = hsv_image[q[:, 1], q[:, 0], 1]
        warm = ((hue <= 45) | (hue >= 170)) & (saturation >= 100)
        warm_fraction = float(np.mean(warm)) if len(warm) else 0.0

    score = (
        transition * 2.5
        + transition2 * 1.5
        + (1.0 - wrong_bright) * 2.0
        + min(span / max(expected_radius, 1.0), 1.5) * 0.7
        - dark_gap * 1.8
        - dark_gap2 * 0.8
        - min(straightness, 2.0) * 0.35
        - dark_fraction * 0.8
    )
    return {
        "direction": (float(direction[0]), float(direction[1])),
        "normal": (float(normal[0]), float(normal[1])),
        "c": c,
        "visible_sign": float(visible_sign),
        "span": float(span),
        "residual": residual,
        "straightness": float(straightness),
        "transition": transition,
        "dark_gap": dark_gap,
        "bright_fraction": bright_fraction,
        "dark_light_fraction": dark_fraction,
        "wrong_bright": wrong_bright,
        "sharp": bool(sharp),
        "warm_fraction": warm_fraction,
        "score": float(score),
    }

def prefilter_hough_lines(light_mask, solar_pixels, lines, roi_offset,
                          expected_radius):
    """Reject obviously impossible Hough seeds before full horizon measurement.

    Low thresholds can turn background texture into thousands of Hough segments.
    The full horizon metric scans the complete edge cloud for each segment, so a
    cheap rejection pass is essential.  This pass deliberately uses relaxed
    versions of the final physical tests and preserves the original Hough order;
    it neither ranks nor merges survivors.  Final measurement, ranking,
    deduplication, and geometry validation therefore remain authoritative.
    """
    if lines is None or len(lines) == 0:
        return []

    solar_sample = np.asarray(solar_pixels, np.float64)
    if len(solar_sample) > 2048:
        indices = np.linspace(0, len(solar_sample) - 1, 2048, dtype=np.int32)
        solar_sample = solar_sample[indices]

    offset_x, offset_y = roi_offset
    sample_offset = max(2.0, 0.012 * expected_radius)
    wrong_margin = max(1.0, 0.006 * expected_radius)
    relaxed_transition = 0.40 * HORIZON_PROPOSAL_MIN_TRANSITION
    relaxed_bright = 0.65 * HORIZON_PROPOSAL_MIN_BRIGHT_SIDE
    relaxed_dark = HORIZON_PROPOSAL_MAX_DARK_SIDE + 0.20
    relaxed_gap = HORIZON_PROPOSAL_MAX_DARK_GAP + 0.20
    relaxed_wrong = HORIZON_PROPOSAL_MAX_WRONG_BRIGHT + 0.07

    survivors = []
    for raw in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, raw)
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        dx = x2 - x1
        dy = y2 - y1
        span = math.hypot(dx, dy)
        if span < 6.0:
            continue

        direction = np.array((dx / span, dy / span), dtype=np.float64)
        normal = np.array((-direction[1], direction[0]), dtype=np.float64)
        sample_count = int(np.clip(math.ceil(span / 4.0), 12, 64))
        t = np.linspace(0.0, 1.0, sample_count)
        base = np.column_stack((x1 + t * dx, y1 + t * dy))

        _, plus, plus_valid = _sample_binary_side(
            light_mask, base, normal, sample_offset, 1.0
        )
        _, minus, minus_valid = _sample_binary_side(
            light_mask, base, normal, sample_offset, -1.0
        )
        valid = plus_valid & minus_valid
        if np.count_nonzero(valid) < 8:
            continue

        plus_transition = float(np.mean(plus[valid] & ~minus[valid]))
        minus_transition = float(np.mean(minus[valid] & ~plus[valid]))
        if minus_transition > plus_transition:
            bright = minus
            dark = plus
            visible_sign = -1.0
            transition = minus_transition
        else:
            bright = plus
            dark = minus
            visible_sign = 1.0
            transition = plus_transition

        bright_fraction = float(np.mean(bright[valid]))
        dark_fraction = float(np.mean(dark[valid]))
        dark_gap = float(np.mean(~bright[valid] & ~dark[valid]))
        if transition < relaxed_transition:
            continue
        if bright_fraction < relaxed_bright:
            continue
        if dark_fraction > relaxed_dark or dark_gap > relaxed_gap:
            continue

        if len(solar_sample):
            c = -(normal[0] * x1 + normal[1] * y1)
            distance = solar_sample @ normal + c
            wrong_bright = float(np.mean(
                distance * visible_sign < -wrong_margin
            ))
            if wrong_bright > relaxed_wrong:
                continue

        survivors.append((x1, y1, x2, y2))

    return survivors


def find_horizon_proposals(gray, light_mask, color_image, settings):
    """Find straight threshold-border proposals before any ellipse is fitted."""
    height, width = gray.shape
    expected_radius = max(1.0, 0.5 * (settings.min_radius + settings.max_radius))
    search_radius = 1.35 * settings.max_radius

    center = None
    if color_image is not None:
        hint = adaptive_solar_hint(color_image)
        center = hint.get("center")
    if center is None:
        high = gray >= np.percentile(gray, 99.8)
        ys, xs = np.nonzero(high)
        if len(xs):
            center = (float(xs.mean()), float(ys.mean()))
        else:
            center = (0.5 * width, 0.5 * height)

    hsv_image = (
        cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        if color_image is not None else None
    )

    cx, cy = center
    x0 = max(0, int(math.floor(cx - search_radius)))
    x1 = min(width, int(math.ceil(cx + search_radius)) + 1)
    y0 = max(0, int(math.floor(cy - search_radius)))
    y1 = min(height, int(math.ceil(cy + search_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return []

    roi = light_mask[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    near = (xx - cx)**2 + (yy - cy)**2 <= search_radius**2
    solar_y, solar_x = np.nonzero((roi != 0) & near)
    solar_pixels = np.column_stack((solar_x + x0, solar_y + y0)).astype(np.float64)
    if len(solar_pixels) < 10:
        return []

    edges = cv2.Canny(roi, 40, 120)
    edge_y, edge_x = np.nonzero(edges)
    edge_points = np.column_stack((edge_x + x0, edge_y + y0)).astype(np.float64)
    if len(edge_points) < 10:
        return []

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(8, int(0.04 * expected_radius)),
        minLineLength=max(8, int(0.04 * expected_radius)),
        maxLineGap=max(3, int(0.02 * expected_radius)),
    )
    if lines is None:
        return []

    seeds = prefilter_hough_lines(
        light_mask,
        solar_pixels,
        lines,
        (x0, y0),
        expected_radius,
    )

    proposals = []
    for seed in seeds:
        candidate = _horizon_line_metrics(
            gray,
            light_mask,
            solar_pixels,
            edge_points,
            seed,
            expected_radius,
            hsv_image=hsv_image,
        )
        if candidate is None:
            continue
        if candidate["transition"] < HORIZON_PROPOSAL_MIN_TRANSITION:
            continue
        if candidate["bright_fraction"] < HORIZON_PROPOSAL_MIN_BRIGHT_SIDE:
            continue
        if candidate["dark_light_fraction"] > HORIZON_PROPOSAL_MAX_DARK_SIDE:
            continue
        if candidate["dark_gap"] > HORIZON_PROPOSAL_MAX_DARK_GAP:
            continue
        if candidate["wrong_bright"] > HORIZON_PROPOSAL_MAX_WRONG_BRIGHT:
            continue
        if not candidate["sharp"]:
            continue
        proposals.append(candidate)

    # Rank longer/globally straighter physical borders first, then deduplicate
    # nearly identical lines. Straightness is not itself a hard acceptance rule.
    proposals.sort(
        key=lambda item: (
            item["score"],
            item["span"],
            -item["straightness"],
        ),
        reverse=True,
    )
    unique = []
    for candidate in proposals:
        angle = math.atan2(candidate["direction"][1], candidate["direction"][0])
        duplicate = False
        for previous in unique:
            old_angle = math.atan2(previous["direction"][1], previous["direction"][0])
            angle_delta = abs((angle - old_angle + math.pi / 2) % math.pi - math.pi / 2)
            if angle_delta >= math.radians(4.0):
                continue
            # Compare perpendicular distance of both lines at the solar hint.
            dc = abs(
                abs(np.dot(center, candidate["normal"]) + candidate["c"])
                - abs(np.dot(center, previous["normal"]) + previous["c"])
            )
            if dc < 0.06 * expected_radius:
                duplicate = True
                break
        if not duplicate:
            candidate["segment"] = line_segment_across_image(candidate, width, height)
            unique.append(candidate)
        if len(unique) >= HORIZON_PROPOSAL_LIMIT:
            break
    return unique


def line_ellipse_intersections(proposal, ellipse):
    """Return the two signed coordinates where a line crosses an ellipse.

    ``direction`` and ``normal`` are orthonormal vectors describing the line
    ``normal.x * x + normal.y * y + c = 0``.  The returned values are distances
    along ``direction`` from the line point closest to the image origin.  This
    representation lets an observed straight contour run be compared directly
    with the exact chord predicted by a fitted ellipse, independent of camera
    rotation.
    """
    vx, vy = proposal["direction"]
    nx, ny = proposal["normal"]
    c = proposal["c"]

    # Because normal is unit length, -c * normal is the point on the line nearest
    # the origin.  Its dot product with direction is zero, so dot(point,
    # direction) is also the signed line coordinate used below.
    p0x = -c * nx
    p0y = -c * ny

    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Transform the line into normalized ellipse coordinates.  Intersection with
    # x^2 + y^2 = 1 then reduces to a quadratic in the line coordinate t.
    dx0 = p0x - cx
    dy0 = p0y - cy
    x0 = (cos_a * dx0 + sin_a * dy0) / ellipse["major"]
    y0 = (-sin_a * dx0 + cos_a * dy0) / ellipse["minor"]
    xv = (cos_a * vx + sin_a * vy) / ellipse["major"]
    yv = (-sin_a * vx + cos_a * vy) / ellipse["minor"]

    qa = xv * xv + yv * yv
    qb = 2.0 * (x0 * xv + y0 * yv)
    qc = x0 * x0 + y0 * y0 - 1.0
    if qa <= 1e-18:
        return None

    discriminant = qb * qb - 4.0 * qa * qc
    if discriminant < 0:
        # Permit tiny negative roundoff only; a genuinely non-intersecting line
        # is not a solar chord and therefore cannot be the horizon.
        if discriminant < -1e-9:
            return None
        discriminant = 0.0

    root = math.sqrt(discriminant)
    t0 = (-qb - root) / (2.0 * qa)
    t1 = (-qb + root) / (2.0 * qa)
    return (min(t0, t1), max(t0, t1))


def horizon_crosses_light_ellipse(proposal, light_ellipse):
    """Whether the proposed horizon is a true chord of the fitted Sun.

    Screening establishes that the line is a plausible physical bright/dark
    border.  The final geometric requirement is intentionally simple: after the
    Sun has been fitted from eligible light-side evidence, the infinite horizon
    line must intersect that ellipse at two distinct points.  Tangencies and
    non-intersections are rejected.
    """
    interval = line_ellipse_intersections(proposal, light_ellipse)
    if interval is None:
        return False
    chord_length = interval[1] - interval[0]
    tolerance = max(1e-6, 1e-6 * light_ellipse["equivalent_radius"])
    return chord_length > tolerance




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
        retained.append((component, actual_indices))

    if not retained or not seed_overlap_any:
        return None

    # Keep all reasonably supported components.  Tiny unrelated fragments are
    # already rejected by the component size/density gate above.
    segments = []
    all_points = []
    total_bins = 0
    largest_bins = 0
    support_count = 0
    for component, actual_indices in retained:
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
        "largest_segment_coverage": largest_bins / ARC_GROW_SAMPLES,
        "segment_count": len(segments),
        "supported_fraction": support_count / max(total_bins, 1),
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


def grow_candidate(mask, seed, settings, horizon=None):
    # A seed that already satisfies the FINAL constraints remains valid even if
    # arc growth finds no additional support.  Growth is an improvement stage,
    # not a prerequisite for accepting an otherwise valid short arc.
    accepted_seed = None
    if (
        axes_in_range(seed, settings)
        and seed.get("relative_error", math.inf) <= settings.max_error
        and seed.get("coverage", 0.0) >= settings.min_coverage
    ):
        accepted_seed = seed.copy()
        accepted_seed.setdefault("arc_segments", [np.asarray(seed["points"], np.float64)])
        accepted_seed.setdefault("largest_segment_coverage", accepted_seed["coverage"])
        accepted_seed.setdefault("segment_count", 1)
        accepted_seed.setdefault("supported_fraction", 1.0)
        if horizon is not None:
            margin = max(2.0, HORIZON_EXCLUSION_RADIUS * accepted_seed["equivalent_radius"])
            if not np.all(points_on_visible_side(accepted_seed["points"], horizon, margin)):
                accepted_seed = None

    support = grow_arc_support(mask, seed, horizon=horizon)
    if support is None:
        return accepted_seed

    grown_candidates = []
    for ellipse in fit_ellipse_options(support["points"]):
        # Exact user axis limits are imposed here, after all disconnected support
        # available to the model has been gathered.
        if not axes_in_range(ellipse, settings):
            continue

        relative_error, _ = ellipse_error_and_coverage(support["points"], ellipse)
        if relative_error > settings.max_error:
            continue

        candidate = {
            **ellipse,
            **support,
            "relative_error": relative_error,
        }
        # A seed already inside the final axis range should not drift far while
        # growing.  A deliberately relaxed intermediate seed, however, is allowed
        # to move substantially: the whole purpose of the relaxed search envelope
        # is to let short, poorly constrained arcs converge once more real support
        # is available.  Final axis/error/polarity checks remain mandatory.
        if axes_in_range(seed, settings):
            if not growth_compatible(seed, candidate):
                continue

        # Horizon constraint is rechecked after refitting because the fitted model
        # can move slightly relative to the support from the seed geometry.
        if horizon is not None:
            margin = max(2.0, HORIZON_EXCLUSION_RADIUS * candidate["equivalent_radius"])
            if not np.all(points_on_visible_side(candidate["points"], horizon, margin)):
                continue

        polarity, _inside_fraction, sample_count = boundary_polarity(mask, candidate)
        if sample_count < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue

        candidate["boundary_polarity"] = polarity
        candidate["score"] = (
            candidate["coverage"]
            * (0.65 + 0.35 * candidate["supported_fraction"])
            * (0.65 + 0.35 * polarity)
            / (1.0 + 4.0 * relative_error)
        )
        if candidate["coverage"] < settings.min_coverage:
            continue

        grown_candidates.append(candidate)

    best = prefer_rounder_fit(grown_candidates)
    return best if best is not None else accepted_seed


def prepare_candidates(mask, candidates, settings, horizon=None):
    prepared = []
    for seed in candidates[:MAX_CLASS_CANDIDATES]:
        candidate = grow_candidate(
            mask,
            seed,
            settings,
            horizon=horizon,
        )
        if candidate is None:
            continue
        if any(same_ellipse(candidate, old, 0.08, 0.08, 12.0) for old in prepared):
            continue
        prepared.append(candidate)

    def rank_key(item):
        key = (
            item["coverage"],
            item.get("largest_segment_coverage", 0.0),
            item.get("supported_fraction", 1.0),
            item.get("boundary_polarity", 0.0),
            -item["relative_error"],
        )
        # Preserve the established active-horizon ordering: once the visible
        # half-plane is enforced, candidates with otherwise identical support
        # are not re-ordered by the composite score.  The ordinary path keeps
        # score as its final tie-breaker.
        if horizon is None:
            key += (item.get("score", 0.0),)
        return key

    prepared.sort(key=rank_key, reverse=True)
    return prepared


def discover_candidates(mask, settings, *, horizon=None,
                        visible_side_only=False, allow_outer_limb=True):
    """Find, grow, validate, deduplicate, and rank one threshold class.

    ``visible_side_only`` is used only for Sun/light evidence under a provisional
    horizon.  It changes which contour/support points are eligible, not the
    ellipse fitting, scoring, growth, or model-selection algorithm.
    """
    seeds = find_candidates(
        mask,
        settings,
        horizon=horizon,
        visible_side_only=visible_side_only,
        allow_outer_limb=allow_outer_limb,
    )
    growth_horizon = horizon if visible_side_only else None
    return prepare_candidates(mask, seeds, settings, horizon=growth_horizon)


def select_ellipse_classes(light_mask, dark_candidates, light_candidates,
                           horizon=None):
    """Select the final dark and light class representatives from ranked lists."""
    dark = dark_candidates[0].copy() if dark_candidates else None
    if dark is not None:
        dark["class"] = "below threshold"

    light = None
    for candidate in light_candidates:
        if dark is not None and same_ellipse(candidate, dark, 0.08, 0.08, 12.0):
            continue
        if dark is not None:
            fraction, visible_pixels = visible_fraction(
                light_mask, candidate, dark, horizon=horizon
            )
            if visible_pixels < 32 or fraction < 0.40:
                continue
        light = candidate.copy()
        light["class"] = "above threshold"
        break
    return dark, light


# ---------------------------------------------------------------------------
# Dark/light detection orchestration
# ---------------------------------------------------------------------------
def detect(gray, threshold, settings, color_image=None, use_horizon=True):
    """Detect at most one dark/light ellipse plus an independent horizon."""
    _, dark_mask = cv2.threshold(
        gray, int(threshold), 255, cv2.THRESH_BINARY_INV
    )
    _, light_mask = cv2.threshold(
        gray, int(threshold), 255, cv2.THRESH_BINARY
    )

    # Screen straight physical-border proposals before ellipse fitting.  Each
    # proposal is then tested by fitting the Sun with the *normal* light-ellipse
    # algorithm while making the proposal's dark half-plane ineligible as light
    # evidence.  The proposal is valid only if its line subsequently crosses the
    # fitted Sun at two distinct points.
    proposals = find_horizon_proposals(
        gray,
        light_mask,
        color_image,
        settings,
    )

    horizon = None
    horizon_dark_candidates = []
    horizon_light_candidates = []
    for proposal in proposals:
        # The straight line itself is excluded from both classes so it cannot be
        # mistaken for curved support.  Only the Sun/light class is restricted to
        # the proposal's visible half-plane; Moon/dark fitting remains otherwise
        # unconstrained by the horizon.
        proposal_dark = discover_candidates(
            dark_mask,
            settings,
            horizon=proposal,
            visible_side_only=False,
            allow_outer_limb=False,
        )
        proposal_light = discover_candidates(
            light_mask,
            settings,
            horizon=proposal,
            visible_side_only=True,
            allow_outer_limb=False,
        )

        _proposal_dark, proposal_sun = select_ellipse_classes(
            light_mask,
            proposal_dark,
            proposal_light,
            horizon=proposal,
        )
        geometry_confirmed = (
            proposal_sun is not None
            and horizon_crosses_light_ellipse(proposal, proposal_sun)
        )

        if geometry_confirmed:
            horizon = {
                "normal": proposal["normal"],
                "direction": proposal["direction"],
                "c": proposal["c"],
                "visible_sign": proposal["visible_sign"],
                "residual": proposal["residual"],
                "dark_light_fraction": proposal["dark_light_fraction"],
                "visible_light_fraction": proposal["bright_fraction"],
                "segment": proposal.get("segment"),
            }
            horizon_dark_candidates = proposal_dark
            horizon_light_candidates = proposal_light
            break

    # Detection and application are deliberately separate.  The UI may keep a
    # detected horizon available while the user disables it as a false positive.
    detected_horizon = horizon
    if detected_horizon is not None:
        detected_horizon["active"] = bool(use_horizon)
    active_horizon = detected_horizon if use_horizon else None

    if active_horizon is None:
        # Ordinary no-horizon path remains unchanged, including optional morphology
        # and outer-limb assistance.
        dark_candidates = discover_candidates(dark_mask, settings)
        light_candidates = discover_candidates(light_mask, settings)
    else:
        # These candidates were already run through the shared discovery/growth
        # pipeline with the accepted proposal.  Reuse them directly: dark fitting
        # only excluded the straight line; light fitting additionally excluded all
        # evidence on the horizon's dark side.
        dark_candidates = horizon_dark_candidates
        light_candidates = horizon_light_candidates

    dark, light = select_ellipse_classes(
        light_mask,
        dark_candidates,
        light_candidates,
        horizon=active_horizon,
    )
    ellipses = [ellipse for ellipse in (dark, light) if ellipse is not None]
    return light_mask, ellipses, detected_horizon


# ---------------------------------------------------------------------------
# Rendering and scale conversion
# ---------------------------------------------------------------------------
def scale_ellipse(ellipse, factor):
    """Map detected ellipse geometry and support points by scale."""
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

    if horizon is not None and horizon.get("active", True):
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


def process_working_image(gray, threshold, settings, color_image=None, use_horizon=True):
    binary, ellipses, horizon = detect(
        gray,
        threshold,
        settings,
        color_image=color_image,
        use_horizon=use_horizon,
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
    h, _, v = cv2.split(hsv)

    v_hi = float(np.percentile(v, 99.8))
    g_hi = float(np.percentile(gray, 99.8))
    v_thr = int(max(20, min(245, max(v_hi * 0.72, float(v.max()) * 0.65))))
    g_thr = int(max(20, min(245, max(g_hi * 0.72, float(gray.max()) * 0.65))))

    # Warm-hue pixels locate the likely solar region; brightness is gated below.
    warm = (h <= 40) | (h >= 170)
    bright = (v >= v_thr) | (gray >= g_thr)
    solar_like = np.where(warm & bright, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        solar_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) >= 20]
    center = None
    if contours:
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            center = (
                float(moments["m10"] / moments["m00"]),
                float(moments["m01"] / moments["m00"]),
            )

    hints = {g_thr}
    values = gray[solar_like != 0]
    if values.size >= 100:
        for q in (1, 5, 10, 20):
            hints.add(int(np.percentile(values, q)))

        ys, xs = np.nonzero(solar_like)
        if len(xs):
            span = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            pad = max(10, int(0.15 * span))
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
        "threshold_hints": [int(np.clip(value, 0, 255)) for value in hints],
    }


def build_threshold_candidates(gray, bgr=None, max_colors=20, fallback=8):
    """Choose thresholds that are likely to change segmentation meaningfully.

    The palette does not simply return abundant grayscale values. It prioritizes
    boundaries around strong histogram peaks,
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
        for hint in adaptive_solar_hint(bgr).get("threshold_hints", []):
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


def auto_select_threshold(color_image, fallback_threshold):
    """Return a quick orientative per-image threshold estimate.

    Automatic initialization is deliberately cheap.  It downsizes the image,
    derives warm/bright solar evidence plus local grayscale/Otsu hints, and uses
    the lowest credible adaptive hint as a conservative starting threshold.

    This function performs no contour search, ellipse fitting, arc recovery,
    horizon detection, morphology, or multi-threshold detector scoring.  The
    normal Refresh Preview pass is the first real detector run; the threshold is
    then explicitly stored per image and remains available for manual fine tuning.
    """
    fallback = int(np.clip(fallback_threshold, 0, 255))
    if color_image is None or color_image.size == 0:
        return accepted_seed

    working, _ = resize_for_detection(color_image, AUTO_THRESHOLD_MAX_DIM)
    solar_hint = adaptive_solar_hint(working)
    hints = sorted({
        int(np.clip(value, 0, 255))
        for value in solar_hint.get("threshold_hints", [])
        if 0 < int(value) < 255
    })

    # The lowest credible adaptive hint is intentionally conservative.  It is an
    # orientation only, not an attempt to optimize detection automatically.
    return int(hints[0]) if hints else fallback


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

        defaults = self.effective_default_settings()
        self.threshold = tk.IntVar(value=defaults["threshold"])
        self.min_radius = tk.IntVar(value=round(defaults["min_radius"]))
        self.max_radius = tk.IntVar(value=round(defaults["max_radius"]))
        self.max_error = tk.DoubleVar(value=defaults["max_error"] * 100)
        self.min_coverage = tk.IntVar(value=round(defaults["min_coverage"] * 100))
        self.morphology = tk.BooleanVar(value=False)
        self.outer_limb_assistance = tk.BooleanVar(value=False)
        self.use_horizon = tk.BooleanVar(value=True)

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
            "use_horizon": True,
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
            "use_horizon": bool(self.use_horizon.get()),
        }

    @staticmethod
    def setting_equal(name, value, default):
        if isinstance(default, bool):
            return bool(value) == bool(default)
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
            self.use_horizon.set(bool(values["use_horizon"]))
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
        if (
            self.current_path in self.image_overrides
            and "threshold" in self.image_overrides[self.current_path]
        ):
            return

        defaults = self.effective_default_settings()
        self.status.set("Estimating a quick initial threshold...")
        self.root.update_idletasks()
        threshold = auto_select_threshold(
            self.color_image,
            defaults["threshold"],
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
        self.update_horizon_control_state(False)
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
        ).grid(row=0, column=1, sticky="w", padx=(0, 20))
        self.horizon_checkbox = tk.Checkbutton(
            options,
            text="Use detected horizon",
            variable=self.use_horizon,
            command=self.pending,
            state=tk.DISABLED,
        )
        self.horizon_checkbox.grid(row=0, column=2, sticky="w")

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

    def update_horizon_control_state(self, detected):
        """Enable the horizon override only when this run found a horizon."""
        self.horizon_checkbox.config(
            state=tk.NORMAL if detected else tk.DISABLED
        )

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

        detector_settings = DetectionSettings(
            min_radius=working_min,
            max_radius=working_max,
            max_error=settings["max_error"],
            min_coverage=settings["min_coverage"],
            max_contours=self.args.max_contours,
            max_search_points=self.args.max_search_points,
            morphology=settings["morphology"],
            outer_limb_assistance=settings["outer_limb_assistance"],
        )
        threshold_preview, working_ellipses, working_horizon = process_working_image(
            working_gray,
            settings["threshold"],
            detector_settings,
            color_image=working_color,
            use_horizon=settings["use_horizon"],
        )

        self.update_horizon_control_state(working_horizon is not None)

        # Map preview geometry to original coordinates only for the full-color
        # centering pane.  Full-resolution mode already has factor 1.
        to_original = 1.0 / scale
        original_ellipses = [scale_ellipse(e, to_original) for e in working_ellipses]
        color_preview = center_color_image_on_light_ellipse(
            self.color_image,
            original_ellipses,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        self.last_threshold_preview = threshold_preview
        self.last_color_preview = color_preview
        self.redraw()

        if working_horizon is None:
            horizon_text = "horizon=no"
        elif settings["use_horizon"]:
            horizon_text = "horizon=detected/applied"
        else:
            horizon_text = "horizon=detected/ignored"
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
        if working_horizon is not None:
            print(
                f"{os.path.basename(self.current_path)} | {mode_label} | horizon: "
                f"dark-light={working_horizon['dark_light_fraction']:.1%}, "
                f"visible-light={working_horizon['visible_light_fraction']:.1%}, "
                f"residual={working_horizon['residual'] / scale:.2f}px"
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
    # This remains the fallback if the quick adaptive threshold hint is unavailable.
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
