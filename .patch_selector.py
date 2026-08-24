from pathlib import Path
import re
import textwrap

path = Path("circle_arc_detector.py")
text = path.read_text()
pattern = re.compile(
    r"def auto_select_threshold\(color_image, fallback_threshold, min_radius, max_radius,\n"
    r"\s+max_error, min_coverage, max_contours, max_points,\n"
    r"\s+palette_size=20\):.*?\n\n\n# ---------------------------------------------------------------------------\n# Tkinter UI",
    re.S,
)
replacement = textwrap.dedent(r'''
def auto_select_threshold(color_image, fallback_threshold, min_radius, max_radius,
                          max_error, min_coverage, max_contours, max_points,
                          palette_size=20):
    """Choose a fast, robust per-image starting threshold.

    Use the lowest credible adaptive brightness/color hint first.  That
    conservative value handles the normal partial-eclipse, totality and full-Sun
    cases cheaply.  Extra work is reserved for difficult frames: non-dominant
    dark-only, weak single-ellipse, or no-ellipse results.

    Difficult frames verify the narrow neighborhood around the legacy fallback
    (normally T=6..10 around T=8) at normal preview resolution.  Late-sunset
    frames can change result after a one-level threshold move, so this
    neighborhood is intentionally contiguous.  Only if it fails are the
    strongest remaining adaptive hints preview-verified.
    """
    working, auto_scale = resize_for_detection(
        color_image,
        AUTO_THRESHOLD_MAX_DIM,
    )
    solar_hint = adaptive_solar_hint(working)

    hints = sorted({
        int(np.clip(value, 0, 255))
        for value in solar_hint.get("threshold_hints", [])
        if 0 < int(value) < 255
    })
    primary = int(min(hints)) if hints else int(fallback_threshold)

    auto_min = max(1.0, min_radius * auto_scale)
    auto_max = max(auto_min, max_radius * auto_scale)
    auto_expected_radius = 0.5 * (auto_min + auto_max)
    test_contours = min(max_contours, 12) if max_contours > 0 else 12
    test_points = min(max_points, 120) if max_points > 0 else 120

    def run_detection(image, threshold, scale, hint, contours, points):
        work_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_min = max(1.0, min_radius * scale)
        scaled_max = max(scaled_min, max_radius * scale)
        try:
            _, ellipses, horizon = detect(
                work_gray,
                int(threshold),
                scaled_min,
                scaled_max,
                max_error,
                min_coverage,
                contours,
                points,
                morphology=False,
                outer_limb_assistance=False,
                color_image=image,
            )
        except (cv2.error, np.linalg.LinAlgError, ValueError):
            return [], None, -1e18

        expected_radius = 0.5 * (scaled_min + scaled_max)
        score = score_detection_for_threshold(
            ellipses,
            horizon,
            solar_hint=hint,
            expected_radius=expected_radius,
        )
        score -= 0.0005 * abs(
            int(threshold) - int(fallback_threshold)
        )
        return ellipses, horizon, score

    def light_near_hint(ellipses, hint, expected_radius):
        center = hint.get("center") if hint is not None else None
        if center is None:
            return True
        light = next(
            (
                ellipse
                for ellipse in ellipses
                if ellipse.get("class") == "above threshold"
            ),
            None,
        )
        if light is None:
            return False
        return (
            math.hypot(
                light["center"][0] - center[0],
                light["center"][1] - center[1],
            )
            <= 1.2 * max(expected_radius, 1.0)
        )

    def result_kind(ellipses, expected_radius, hint):
        dark = next(
            (
                ellipse
                for ellipse in ellipses
                if ellipse.get("class") == "below threshold"
            ),
            None,
        )
        light = next(
            (
                ellipse
                for ellipse in ellipses
                if ellipse.get("class") == "above threshold"
            ),
            None,
        )
        if dark is not None and light is not None:
            return (
                "both"
                if light_near_hint(ellipses, hint, expected_radius)
                else "weak"
            )
        if dark is not None:
            return (
                "dominant_dark"
                if dark.get("coverage", 0.0) >= DARK_DOMINANT_COVERAGE
                else "weak"
            )
        if light is not None:
            if (
                light.get("coverage", 0.0) >= max(0.20, min_coverage)
                and light_near_hint(ellipses, hint, expected_radius)
            ):
                return "light"
            return "weak"
        return "none"

    primary_ellipses, _, primary_score = run_detection(
        working,
        primary,
        auto_scale,
        solar_hint,
        test_contours,
        test_points,
    )
    primary_kind = result_kind(
        primary_ellipses,
        auto_expected_radius,
        solar_hint,
    )
    if primary_kind in ("both", "dominant_dark", "light"):
        return int(primary)

    # Difficult-case verification uses the same working resolution as Refresh
    # Preview.  This prevents the automatic choice from depending on a 600-pixel
    # proxy when the useful threshold window is only one or two grayscale levels.
    preview_image, preview_scale = resize_for_detection(
        color_image,
        PREVIEW_MAX_DIM,
    )
    preview_hint = adaptive_solar_hint(preview_image)
    preview_min = max(1.0, min_radius * preview_scale)
    preview_max = max(preview_min, max_radius * preview_scale)
    preview_expected_radius = 0.5 * (preview_min + preview_max)

    fallback = int(np.clip(fallback_threshold, 0, 255))
    low_order = []
    for value in (
        fallback - 1,
        fallback,
        fallback - 2,
        fallback + 1,
        fallback + 2,
    ):
        value = int(np.clip(value, 0, 255))
        if value not in low_order:
            low_order.append(value)

    best_threshold = int(primary)
    best_score = float(primary_score)

    # Normal search caps are intentional here.  Reduced contour/point caps were
    # observed to miss legitimate T=6/T=7 sunset limbs.
    for threshold in low_order:
        ellipses, _, score = run_detection(
            preview_image,
            threshold,
            preview_scale,
            preview_hint,
            max_contours,
            max_points,
        )
        kind = result_kind(
            ellipses,
            preview_expected_radius,
            preview_hint,
        )
        if score > best_score:
            best_threshold = int(threshold)
            best_score = score
        if kind == "both":
            return int(threshold)

    adaptive_pool = set(hints)
    for hint in list(hints):
        adaptive_pool.update(
            value
            for value in (hint - 1, hint + 1)
            if 0 <= value <= 255
        )
    adaptive_pool.difference_update(low_order)
    adaptive_pool.discard(primary)

    ranked = []
    for threshold in sorted(adaptive_pool):
        _, _, score = run_detection(
            working,
            threshold,
            auto_scale,
            solar_hint,
            test_contours,
            test_points,
        )
        ranked.append((score, int(threshold)))
    ranked.sort(reverse=True)

    for _, threshold in ranked[:2]:
        ellipses, _, score = run_detection(
            preview_image,
            threshold,
            preview_scale,
            preview_hint,
            max_contours,
            max_points,
        )
        kind = result_kind(
            ellipses,
            preview_expected_radius,
            preview_hint,
        )
        if score > best_score:
            best_threshold = int(threshold)
            best_score = score
        if kind == "both":
            return int(threshold)

    return int(best_threshold)


# ---------------------------------------------------------------------------
# Tkinter UI
''').strip()
new_text, count = pattern.subn(lambda _match: replacement, text)
if count != 1:
    raise SystemExit(f"auto_select_threshold replacement count={count}")
path.write_text(new_text + ("\n" if not new_text.endswith("\n") else ""))
