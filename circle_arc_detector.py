import argparse
import base64
import math
import time
import tkinter as tk

import cv2
import numpy as np

MIN_BOUNDARY_POLARITY = 0.42
MAX_CLASS_CANDIDATES = 30
MAX_REGIONS_PER_CONTOUR = 4
EDGE_SEARCH_FRACTION = 0.15
ARC_GROW_SAMPLES = 720
DARK_DOMINANT_COVERAGE = 0.55


def normalize_ellipse(raw):
    try:
        (cx, cy), (width, height), angle = raw
    except Exception:
        return None
    if not np.all(np.isfinite((cx, cy, width, height, angle))) or width <= 0 or height <= 0:
        return None
    semi_x, semi_y = width * 0.5, height * 0.5
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


def fit_circle_prior(points):
    points = np.asarray(points, np.float64)
    if len(points) < 3:
        return None
    x, y = points[:, 0], points[:, 1]
    xm, ym = float(x.mean()), float(y.mean())
    u, v = x - xm, y - ym
    z = u * u + v * v
    suu, svv, suv = np.dot(u, u), np.dot(v, v), np.dot(u, v)
    suz, svz = np.dot(u, z), np.dot(v, z)
    det = suu * svv - suv * suv
    if abs(det) <= 1e-12 * (suu * svv + 1.0):
        return None
    uc = 0.5 * (suz * svv - svz * suv) / det
    vc = 0.5 * (svz * suu - suz * suv) / det
    cx, cy = xm + uc, ym + vc
    radius_sq = float(z.mean()) + uc * uc + vc * vc
    if radius_sq <= 0 or not np.isfinite(radius_sq):
        return None
    radius = math.sqrt(radius_sq)
    return {
        "center": (float(cx), float(cy)),
        "major": radius,
        "minor": radius,
        "angle": 0.0,
        "equivalent_radius": radius,
        "fit_kind": "circle-prior",
    }


def fit_ellipse_options(points):
    points = np.asarray(points, np.float32)
    if len(points) < 5:
        return []
    shaped = points.reshape(-1, 1, 2)
    options = []
    for name, fitter in (
        ("direct", cv2.fitEllipseDirect),
        ("ams", cv2.fitEllipseAMS),
        ("standard", cv2.fitEllipse),
    ):
        try:
            ellipse = normalize_ellipse(fitter(shaped))
        except cv2.error:
            ellipse = None
        if ellipse is not None:
            ellipse["fit_kind"] = name
            options.append(ellipse)
    circle = fit_circle_prior(points)
    if circle is not None:
        options.append(circle)
    return options


def ellipse_coordinates(points, ellipse):
    points = np.asarray(points, np.float64)
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = points[:, 0] - cx, points[:, 1] - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return local_x / ellipse["major"], local_y / ellipse["minor"]


def ellipse_error_and_coverage(points, ellipse):
    if len(points) < 2:
        return math.inf, 0.0
    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    relative_error = float(np.mean(np.abs(np.hypot(x_norm, y_norm) - 1.0)))
    angles = np.unwrap(np.arctan2(y_norm, x_norm))
    coverage = float(np.clip(np.ptp(angles) / (2.0 * np.pi), 0.0, 1.0))
    return relative_error, coverage


def boundary_polarity(mask, ellipse):
    points = np.asarray(ellipse["points"], np.float64)
    if len(points) < 5:
        return 0.0, 0.0, 0.0, 0
    x_norm, y_norm = ellipse_coordinates(points, ellipse)
    theta = np.arctan2(y_norm, x_norm)
    radius = ellipse["equivalent_radius"]
    sample_distance = max(3.0, radius * 0.008)
    delta = sample_distance / max(radius, 1.0)
    angle = math.radians(ellipse["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
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
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0
    inside = mask[yi[valid], xi[valid]] != 0
    outside = mask[yo[valid], xo[valid]] != 0
    polarity = float(np.mean(inside & ~outside))
    return polarity, float(np.mean(inside)), float(np.mean(outside)), int(np.count_nonzero(valid))


def window_lengths(n, minimum):
    minimum = max(5, min(minimum, n))
    lengths = {minimum, n}
    length = minimum
    while length < n:
        lengths.add(length)
        length = min(n, max(length + 1, int(round(length * 1.45))))
    return sorted(lengths)


def eval_region(extended, n, start, length, mask, min_radius, max_radius, max_error, min_coverage):
    if length < 5 or length > n:
        return None
    start %= n
    region = extended[start : start + length]
    best = None
    for ellipse in fit_ellipse_options(region):
        if not (
            min_radius <= ellipse["major"] <= max_radius
            and min_radius <= ellipse["minor"] <= max_radius
        ):
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
        polarity, inside, outside, samples = boundary_polarity(mask, candidate)
        if samples < 5 or polarity < MIN_BOUNDARY_POLARITY or inside < 0.5:
            continue
        support = length / n
        axis_ratio = candidate["major"] / max(candidate["minor"], 1.0)
        shape_penalty = 1.0 + 0.06 * max(0.0, axis_ratio - 1.0)
        geometry_score = (
            coverage**1.5
            * math.sqrt(length)
            * (0.75 + 0.25 * math.sqrt(support))
            / ((relative_error + 0.002) * shape_penalty)
        )
        candidate.update(
            {
                "boundary_polarity": polarity,
                "interior_fraction": inside,
                "outside_fraction": outside,
                "score": geometry_score * polarity**2,
            }
        )
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def angle_difference_180(a, b):
    difference = abs((a - b) % 180.0)
    return min(difference, 180.0 - difference)


def same_ellipse(a, b, center_fraction=0.12, axis_fraction=0.12, angle_tolerance=15.0):
    ax, ay = a["center"]
    bx, by = b["center"]
    scale = max(a["equivalent_radius"], b["equivalent_radius"], 1.0)
    if math.hypot(ax - bx, ay - by) >= center_fraction * scale:
        return False
    if abs(a["major"] - b["major"]) >= axis_fraction * scale:
        return False
    if abs(a["minor"] - b["minor"]) >= axis_fraction * scale:
        return False
    a_eccentricity = (a["major"] - a["minor"]) / max(a["major"], 1.0)
    b_eccentricity = (b["major"] - b["minor"]) / max(b["major"], 1.0)
    if max(a_eccentricity, b_eccentricity) < 0.05:
        return True
    return angle_difference_180(a["angle"], b["angle"]) < angle_tolerance


def refine(candidate, extended, n, minimum, mask, min_radius, max_radius, max_error, min_coverage):
    best = candidate
    for _ in range(40):
        improved = False
        for ds, dl in ((-1, 1), (0, 1), (1, -1), (0, -1), (-1, 0), (1, 0)):
            new_length = best["length"] + dl
            if not minimum <= new_length <= n:
                continue
            trial = eval_region(
                extended,
                n,
                best["start"] + ds,
                new_length,
                mask,
                min_radius,
                max_radius,
                max_error,
                min_coverage,
            )
            if trial is not None and trial["score"] > best["score"] * (1.0 + 1e-9):
                best, improved = trial, True
        if not improved:
            break
    return best


def find_regions(points, mask, min_radius, max_radius, max_error, min_coverage, minimum=12):
    points = np.asarray(points, np.float64)
    n = len(points)
    if n < max(5, minimum):
        return []
    minimum = min(minimum, n)
    extended = np.vstack((points, points))
    coarse = []
    for length in window_lengths(n, minimum):
        step = max(1, length // 5)
        for start in range(0, n, step):
            candidate = eval_region(
                extended, n, start, length, mask,
                min_radius, max_radius, max_error, min_coverage,
            )
            if candidate is not None:
                coarse.append(candidate)
    coarse.sort(key=lambda item: item["score"], reverse=True)
    seeds = []
    for candidate in coarse:
        if any(same_ellipse(candidate, existing, 0.10, 0.10) for existing in seeds):
            continue
        seeds.append(candidate)
        if len(seeds) >= max(10, MAX_REGIONS_PER_CONTOUR * 5):
            break
    result = []
    for seed in seeds:
        candidate = refine(
            seed, extended, n, minimum, mask,
            min_radius, max_radius, max_error, min_coverage,
        )
        if not any(same_ellipse(candidate, existing) for existing in result):
            result.append(candidate)
    result.sort(key=lambda item: item["score"], reverse=True)
    return result[:MAX_REGIONS_PER_CONTOUR]


def find_candidates(mask, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    minimum = 12
    min_perimeter = max(12.0, min_radius * min_coverage * math.pi)
    usable = [(cv2.arcLength(contour, True), contour) for contour in contours if len(contour) >= minimum]
    usable = [(perimeter, contour) for perimeter, contour in usable if perimeter >= min_perimeter]
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
            find_regions(points, mask, min_radius, max_radius, max_error, min_coverage, minimum)
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def ellipse_inside(xx, yy, ellipse):
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = xx - cx, yy - cy
    local_x = cos_a * dx + sin_a * dy
    local_y = -sin_a * dx + cos_a * dy
    return (local_x / ellipse["major"]) ** 2 + (local_y / ellipse["minor"]) ** 2 <= 1.0


def ellipse_bounds(ellipse, width, height):
    cx, cy = ellipse["center"]
    major, minor = ellipse["major"], ellipse["minor"]
    angle = math.radians(ellipse["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    extent_x = math.sqrt((major * cos_a) ** 2 + (minor * sin_a) ** 2)
    extent_y = math.sqrt((major * sin_a) ** 2 + (minor * cos_a) ** 2)
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
    visible = int(np.count_nonzero(valid))
    if not visible:
        return 0.0, 0
    matching = int(np.count_nonzero(mask[y0:y1, x0:x1][valid]))
    return matching / visible, visible


def ellipse_points_and_normals(ellipse, theta):
    cx, cy = ellipse["center"]
    angle = math.radians(ellipse["angle"])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local_x = ellipse["major"] * cos_t
    local_y = ellipse["minor"] * sin_t
    x = cx + cos_a * local_x - sin_a * local_y
    y = cy + sin_a * local_x + cos_a * local_y
    normal_x_local = cos_t / max(ellipse["major"], 1e-9)
    normal_y_local = sin_t / max(ellipse["minor"], 1e-9)
    normal_x = cos_a * normal_x_local - sin_a * normal_y_local
    normal_y = sin_a * normal_x_local + cos_a * normal_y_local
    norm = np.hypot(normal_x, normal_y)
    return x, y, normal_x / norm, normal_y / norm


def circular_components(mask):
    n = len(mask)
    if n == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [np.arange(n)]
    start = (int(np.flatnonzero(~mask)[0]) + 1) % n
    order = (start + np.arange(n)) % n
    values = mask[order]
    components = []
    i = 0
    while i < n:
        if not values[i]:
            i += 1
            continue
        j = i
        while j < n and values[j]:
            j += 1
        components.append(order[i:j])
        i = j
    return components


def bridge_small_gaps(supported, max_gap):
    if max_gap <= 0 or not np.any(supported):
        return supported
    result = supported.copy()
    for shift in range(1, max_gap + 1):
        left = np.roll(supported, shift)
        right = np.roll(supported, -shift)
        result |= left & right
    return result


def grow_arc_support(mask, ellipse):
    theta = np.linspace(0.0, 2.0 * np.pi, ARC_GROW_SAMPLES, endpoint=False)
    x0, y0, nx, ny = ellipse_points_and_normals(ellipse, theta)
    edge_search_distance = EDGE_SEARCH_FRACTION * min(ellipse["major"], ellipse["minor"])
    step = max(1.0, edge_search_distance / 90.0)
    offsets = np.arange(
        -edge_search_distance,
        edge_search_distance + 0.5 * step,
        step,
        dtype=np.float32,
    )
    xs = np.rint(x0[:, None] + nx[:, None] * offsets[None, :]).astype(np.int32)
    ys = np.rint(y0[:, None] + ny[:, None] * offsets[None, :]).astype(np.int32)
    height, width = mask.shape
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    values = np.zeros(xs.shape, np.uint8)
    values[valid] = (mask[ys[valid], xs[valid]] != 0).astype(np.uint8)
    transitions = (
        (values[:, :-1] == 1)
        & (values[:, 1:] == 0)
        & valid[:, :-1]
        & valid[:, 1:]
    )
    midpoint_offsets = 0.5 * (offsets[:-1] + offsets[1:])
    costs = np.where(transitions, np.abs(midpoint_offsets)[None, :], np.inf)
    best_index = np.argmin(costs, axis=1)
    supported = np.isfinite(costs[np.arange(ARC_GROW_SAMPLES), best_index])
    if not np.any(supported):
        return None
    found_offset = midpoint_offsets[best_index]
    found_x = x0 + nx * found_offset
    found_y = y0 + ny * found_offset

    seed_x, seed_y = ellipse_coordinates(ellipse["points"], ellipse)
    seed_theta = np.mod(np.arctan2(seed_y, seed_x), 2.0 * np.pi)
    seed_bins = set(
        np.mod(
            np.rint(seed_theta / (2.0 * np.pi) * ARC_GROW_SAMPLES).astype(np.int32),
            ARC_GROW_SAMPLES,
        ).tolist()
    )
    connected = bridge_small_gaps(supported, max(1, round(ARC_GROW_SAMPLES * 0.008)))
    components = circular_components(connected)
    if not components:
        return None

    def component_key(component):
        overlap = sum(int(index) in seed_bins for index in component)
        actual_support = int(np.count_nonzero(supported[component]))
        return overlap, actual_support, len(component)

    component = max(components, key=component_key)
    if component_key(component)[0] == 0:
        return None
    actual_indices = component[supported[component]]
    if len(actual_indices) < 5:
        return None
    return {
        "points": np.column_stack((found_x[actual_indices], found_y[actual_indices])).astype(np.float64),
        "coverage": len(component) / ARC_GROW_SAMPLES,
        "supported_fraction": len(actual_indices) / len(component),
        "support_count": int(len(actual_indices)),
        "edge_search_distance": float(edge_search_distance),
    }


def growth_compatible(seed, grown):
    scale = max(seed["equivalent_radius"], 1.0)
    sx, sy = seed["center"]
    gx, gy = grown["center"]
    if math.hypot(sx - gx, sy - gy) > 0.18 * scale:
        return False
    if abs(seed["major"] - grown["major"]) > 0.20 * scale:
        return False
    if abs(seed["minor"] - grown["minor"]) > 0.20 * scale:
        return False
    return True


def grow_candidate(mask, seed, min_radius, max_radius, max_error, min_coverage):
    support = grow_arc_support(mask, seed)
    if support is None or support["coverage"] <= seed["coverage"]:
        return seed
    best = None
    for ellipse in fit_ellipse_options(support["points"]):
        if not (
            min_radius <= ellipse["major"] <= max_radius
            and min_radius <= ellipse["minor"] <= max_radius
        ):
            continue
        relative_error, _ = ellipse_error_and_coverage(support["points"], ellipse)
        if relative_error > max_error:
            continue
        candidate = {
            **ellipse,
            **support,
            "relative_error": relative_error,
            "start": seed["start"],
            "length": support["support_count"],
        }
        if not growth_compatible(seed, candidate):
            continue
        polarity, inside, outside, samples = boundary_polarity(mask, candidate)
        if samples < 5 or polarity < MIN_BOUNDARY_POLARITY:
            continue
        candidate.update(
            {
                "boundary_polarity": polarity,
                "interior_fraction": inside,
                "outside_fraction": outside,
            }
        )
        candidate["score"] = (
            candidate["coverage"]
            * (0.65 + 0.35 * candidate["supported_fraction"])
            * (0.65 + 0.35 * polarity)
            / (1.0 + 4.0 * relative_error)
        )
        if candidate["coverage"] < min_coverage:
            continue
        if best is None or (
            candidate["coverage"], candidate["score"], candidate["support_count"]
        ) > (
            best["coverage"], best["score"], best["support_count"]
        ):
            best = candidate
    return best if best is not None else seed


def prepare_candidates(mask, candidates, min_radius, max_radius, max_error, min_coverage):
    prepared = []
    for seed in candidates[:MAX_CLASS_CANDIDATES]:
        candidate = grow_candidate(mask, seed, min_radius, max_radius, max_error, min_coverage)
        if any(same_ellipse(candidate, existing, 0.08, 0.08, 12.0) for existing in prepared):
            continue
        prepared.append(candidate)
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


def detect(gray, threshold, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
    _, dark_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)
    _, light_mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    args = (min_radius, max_radius, max_error, min_coverage, max_contours, max_points)
    dark_candidates = prepare_candidates(
        dark_mask,
        find_candidates(dark_mask, *args),
        min_radius, max_radius, max_error, min_coverage,
    )
    light_candidates = prepare_candidates(
        light_mask,
        find_candidates(light_mask, *args),
        min_radius, max_radius, max_error, min_coverage,
    )

    dark = dark_candidates[0].copy() if dark_candidates else None
    if dark is not None:
        dark["class"] = "below threshold"

    if dark is not None and dark["coverage"] >= DARK_DOMINANT_COVERAGE:
        return light_mask, [dark]

    light = None
    for candidate in light_candidates:
        if dark is not None and same_ellipse(candidate, dark, 0.08, 0.08, 12.0):
            continue
        if dark is not None:
            fraction, visible = visible_fraction(light_mask, candidate, dark)
            if visible < 32 or fraction < 0.40:
                continue
        light = candidate.copy()
        light["class"] = "above threshold"
        break
    return light_mask, [ellipse for ellipse in (dark, light) if ellipse is not None]


def process_image(gray, threshold, min_radius, max_radius, max_error, min_coverage, max_contours, max_points):
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

    threshold_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    grayscale_preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    arc_color = (255, 0, 255)
    dark_ellipse_color = (255, 0, 0)
    light_ellipse_color = (0, 255, 255)

    for ellipse in ellipses:
        cx, cy = ellipse["center"]
        center = (int(round(cx)), int(round(cy)))
        axes = (max(1, int(round(ellipse["major"]))), max(1, int(round(ellipse["minor"]))))
        ellipse_color = dark_ellipse_color if ellipse["class"] == "below threshold" else light_ellipse_color

        support = np.rint(ellipse["points"]).astype(np.int32).reshape(-1, 1, 2)
        if len(support) >= 2:
            cv2.polylines(threshold_preview, [support], False, arc_color, 2, cv2.LINE_AA)

        cv2.ellipse(threshold_preview, center, axes, ellipse["angle"], 0, 360, ellipse_color, 2, cv2.LINE_AA)
        cv2.circle(threshold_preview, center, 3, ellipse_color, -1)

        kind = "DARK <= T" if ellipse["class"] == "below threshold" else "BRIGHT > T"
        text = (
            f"{kind}  a={ellipse['major']:.1f}  b={ellipse['minor']:.1f}  "
            f"arc={ellipse['coverage'] * 360:.0f}deg  err={ellipse['relative_error']:.3f}"
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
    grayscale_values = gray.reshape(-1)
    hist = np.bincount(grayscale_values, minlength=256).astype(float)

    dominant_seed_count = min(5, max_colors)
    shades = []
    for shade in np.argsort(hist)[::-1]:
        shade = int(shade)
        if hist[shade] <= 0:
            break
        shades.append(shade)
        if len(shades) >= dominant_seed_count:
            break

    smooth_radius = max(2, min_gap // 3)
    density = np.convolve(hist, np.ones(2 * smooth_radius + 1), mode="same")
    for shade in np.argsort(density)[::-1]:
        shade = int(shade)
        if density[shade] <= 0:
            break
        if shade in shades:
            continue
        if any(abs(shade - old) < min_gap for old in shades):
            continue
        shades.append(shade)
        if len(shades) >= max_colors:
            break

    return [int(shade) for shade in sorted(shades)]


def gray_hex(value):
    value = int(np.clip(value, 0, 255))
    return f"#{value:02x}{value:02x}{value:02x}"


def text_color(gray_value):
    return "#111111" if gray_value >= 150 else "#f7f7f7"


class DetectorApp:
    def __init__(self, root, gray, palette, args):
        self.root = root
        self.gray = gray
        self.palette = palette
        self.args = args
        self.last_threshold_preview = None
        self.last_result = None
        self.binary_photo = None
        self.result_photo = None
        self.resize_job = None

        self.threshold = tk.IntVar(value=args.threshold)
        self.min_radius = tk.IntVar(value=round(args.min_radius))
        self.max_radius = tk.IntVar(value=round(args.max_radius))
        self.max_error = tk.DoubleVar(value=args.max_error * 100)
        self.min_coverage = tk.IntVar(value=round(args.min_coverage * 100))
        self.status = tk.StringVar(value="Adjust settings, then click Apply.")

        root.title("Ellipse / Arc Detector")
        root.minsize(980, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self.build_controls()
        self.build_previews()

        root.bind("<Return>", lambda _event: self.apply())
        root.bind("<Escape>", lambda _event: root.destroy())

    def build_controls(self):
        frame = tk.Frame(self.root, padx=10, pady=8)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        radius_limit = max(1600, round(self.args.max_radius * 1.5))
        rows = [
            ("Brightness threshold (0=black, 255=white)", self.threshold, 0, 255, 1, lambda value: str(int(value))),
            ("Minimum fitted semi-axis radius (px)", self.min_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum fitted semi-axis radius (px)", self.max_radius, 1, radius_limit, 1, lambda value: f"{int(value)} px"),
            ("Maximum average normalized ellipse error (%)", self.max_error, 0.5, 50, 0.1, lambda value: f"{float(value):.1f}%"),
            ("Minimum visible ellipse arc (%)", self.min_coverage, 0, 100, 1, lambda value: f"{int(value)}% (~{int(value) * 3.6:.0f}°)"),
        ]
        for row, spec in enumerate(rows):
            self.add_scale(frame, row, *spec)

        tk.Label(frame, text="Pick threshold from grayscale tones:").grid(row=5, column=0, sticky="nw")
        palette_frame = tk.Frame(frame)
        palette_frame.grid(row=5, column=1, columnspan=2, sticky="w", pady=(0, 8))
        for index, shade in enumerate(self.palette):
            tk.Button(
                palette_frame,
                text=str(shade),
                width=4,
                bg=gray_hex(shade),
                fg=text_color(shade),
                command=lambda value=shade: self.pick(value),
            ).grid(row=index // 10, column=index % 10, padx=2, pady=2)

        tk.Button(frame, text="Apply", width=12, command=self.apply).grid(row=6, column=0, sticky="w", pady=(2, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=1100).grid(row=7, column=0, columnspan=3, sticky="ew", pady=(8, 0))

    def add_scale(self, parent, row, text, variable, low, high, resolution, formatter):
        tk.Label(parent, text=text, width=38, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        tk.Scale(
            parent,
            from_=low,
            to=high,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            variable=variable,
            showvalue=False,
            length=420,
            highlightthickness=0,
        ).grid(row=row, column=1, sticky="ew", pady=2)

        value = tk.Label(parent, width=18, anchor="e")
        value.grid(row=row, column=2, pady=2)

        def update(*_args):
            value.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update)
        update()

    def build_previews(self):
        frame = tk.Frame(self.root, padx=10)
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1, uniform="preview")
        frame.columnconfigure(1, weight=1, uniform="preview")

        tk.Label(frame, text="Threshold preview with detected arcs and ellipses").grid(row=0, column=0, sticky="w", pady=(0, 4))
        tk.Label(frame, text="Grayscale image").grid(row=0, column=1, sticky="w", pady=(0, 4))

        self.binary_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.result_canvas = tk.Canvas(frame, bg="#202020", highlightthickness=1, highlightbackground="#808080")
        self.binary_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.result_canvas.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        self.binary_canvas.bind("<Configure>", self.schedule_redraw)
        self.result_canvas.bind("<Configure>", self.schedule_redraw)
        self.placeholder(self.binary_canvas, "Threshold preview")
        self.placeholder(self.result_canvas, "Grayscale image")

    def pick(self, value):
        self.threshold.set(value)
        self.status.set(f"Threshold set to {value}. Click Apply to recompute.")

    def pending(self, *_args):
        if self.last_result is not None:
            self.status.set("Settings changed. Click Apply to recompute the result.")

    def settings(self):
        min_radius = max(1.0, float(self.min_radius.get()))
        max_radius = max(1.0, float(self.max_radius.get()))
        if max_radius < min_radius:
            max_radius = min_radius
            self.max_radius.set(round(max_radius))
        return (
            int(self.threshold.get()),
            min_radius,
            max_radius,
            max(0.001, self.max_error.get() / 100),
            float(np.clip(self.min_coverage.get() / 100, 0, 1)),
        )

    def apply(self):
        threshold, min_radius, max_radius, max_error, min_coverage = self.settings()
        started = time.perf_counter()
        threshold_preview, result, ellipses = process_image(
            self.gray,
            threshold,
            min_radius,
            max_radius,
            max_error,
            min_coverage,
            self.args.max_contours,
            self.args.max_search_points,
        )
        elapsed = (time.perf_counter() - started) * 1000

        self.last_threshold_preview = threshold_preview
        self.last_result = result
        self.redraw()

        self.status.set(
            f"Applied: T={threshold}; semi-axes={min_radius:.0f}-{max_radius:.0f}px; "
            f"error={max_error:.1%}; arc={min_coverage:.0%}; {len(ellipses)} ellipse(s); {elapsed:.1f} ms."
        )
        for ellipse in ellipses:
            print(
                f"{ellipse['class']}: center=({ellipse['center'][0]:.2f}, {ellipse['center'][1]:.2f}), "
                f"a={ellipse['major']:.2f}, b={ellipse['minor']:.2f}, angle={ellipse['angle']:.1f}, arc={ellipse['coverage'] * 360:.1f}°, "
                f"error={ellipse['relative_error']:.4f}, interior={ellipse['interior_fraction']:.1%}"
            )

    def schedule_redraw(self, _event=None):
        if self.resize_job is not None:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.redraw)

    def redraw(self):
        self.resize_job = None
        if self.last_threshold_preview is not None:
            self.binary_photo = self.show_image(self.binary_canvas, self.last_threshold_preview)
        if self.last_result is not None:
            self.result_photo = self.show_image(self.result_canvas, self.last_result)

    @staticmethod
    def show_image(canvas, image):
        canvas_width = max(2, canvas.winfo_width() - 2)
        canvas_height = max(2, canvas.winfo_height() - 2)
        image_height, image_width = image.shape[:2]
        scale = max(min(canvas_width / image_width, canvas_height / image_height), 1e-6)
        fitted_size = (max(1, round(image_width * scale)), max(1, round(image_height * scale)))
        fitted = cv2.resize(image, fitted_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ok, encoded = cv2.imencode(".png", fitted)
        if not ok:
            return None
        photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.delete("all")
        canvas.create_image(canvas_width // 2 + 1, canvas_height // 2 + 1, image=photo, anchor="center")
        return photo

    @staticmethod
    def placeholder(canvas, text):
        canvas.create_text(160, 120, text=text + "\nClick Apply", fill="#cccccc", justify="center")


def main():
    parser = argparse.ArgumentParser(description="Detect up to two inferred ellipses from thresholded image arcs.")
    parser.add_argument("image")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--min-radius", type=float, default=1000.0, help="Minimum allowed value for each ellipse semi-axis")
    parser.add_argument("--max-radius", type=float, default=1500.0, help="Maximum allowed value for each ellipse semi-axis; 0 = largest image dimension")
    parser.add_argument("--max-error", type=float, default=0.08)
    parser.add_argument("--min-coverage", type=float, default=0.08)
    parser.add_argument("--max-contours", type=int, default=100)
    parser.add_argument("--max-search-points", type=int, default=500)
    parser.add_argument("--palette-size", type=int, default=20)
    parser.add_argument("--palette-min-gap", type=int, default=10)
    args = parser.parse_args()

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

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not load image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if args.max_radius == 0:
        args.max_radius = float(max(gray.shape))
    args.max_radius = max(args.max_radius, args.min_radius)

    palette = build_palette(gray, args.palette_size, args.palette_min_gap)
    root = tk.Tk()
    DetectorApp(root, gray, palette, args)
    root.mainloop()


if __name__ == "__main__":
    main()
