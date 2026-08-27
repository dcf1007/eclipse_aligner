"""Apply the approved second threshold sanitation pass exactly.

This script is intentionally assertion-heavy. It restructures the already validated
threshold implementation without changing its numerical/topological algorithm.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"

text = SOURCE.read_text(encoding="utf-8")
original = text

# ---------------------------------------------------------------------------
# Documentation: ImageSettings is structured state; auto_threshold is gray-only.
# ---------------------------------------------------------------------------
text = text.replace(
    "Each image entry is a normal dictionary with exactly two\n"
    "conceptual fields: ``settings`` contains only sparse overrides from that image's\n"
    "baseline values, while ``auto_threshold_result`` caches the complete automatic\n"
    "threshold result.",
    "Each image entry is a normal dictionary with exactly two\n"
    "conceptual fields: ``settings`` is an ``ImageSettings`` instance containing only\n"
    "sparse overrides from that image's baseline values, while\n"
    "``auto_threshold_result`` caches the complete automatic threshold result.",
)
text = text.replace(
    "The threshold finder itself remains independent of GUI state:\n"
    "it accepts image data and returns an ``AutoThresholdResult`` describing the\n"
    "selected threshold and the topology used to obtain it.",
    "The threshold finder itself remains independent of GUI state: it accepts the\n"
    "authoritative 8-bit grayscale brightness image and returns an\n"
    "``AutoThresholdResult`` describing the selected threshold and the topology used\n"
    "to obtain it. Color-to-grayscale conversion is an input-stage responsibility and\n"
    "is performed before the threshold algorithm is called.",
)

# ---------------------------------------------------------------------------
# ImageSettings: cohesive sparse application state, no ImageState wrapper class.
# ---------------------------------------------------------------------------
algorithm_header = "# ---------------------------------------------------------------------------\n# Grayscale automatic threshold finder\n# ---------------------------------------------------------------------------\n"
assert algorithm_header in text
image_settings = '''# ---------------------------------------------------------------------------\n# Per-image processing settings\n# ---------------------------------------------------------------------------\n@dataclass\nclass ImageSettings:\n    """Sparse per-image overrides; ``None`` means use that setting's baseline."""\n\n    threshold: int | None = None\n    min_radius: int | None = None\n    max_radius: int | None = None\n    max_error: float | None = None\n    min_coverage: int | None = None\n    morphology: bool | None = None\n    outer_limb_assistance: bool | None = None\n    use_horizon: bool | None = None\n    center_target: str | None = None\n\n\n'''
text = text.replace(algorithm_header, image_settings + algorithm_header, 1)

histogram_seed_block = '''@dataclass(frozen=True)\nclass HistogramSeed:\n    peak: int\n    left_edge: int\n    threshold: int\n    point: tuple[int, int]\n    mask: np.ndarray\n    area: int\n    bbox: tuple[int, int, int, int]\n\n\n'''
assert histogram_seed_block in text
text = text.replace(histogram_seed_block, "", 1)

# ---------------------------------------------------------------------------
# Histogram discovery: one cohesive function, same exact [1,2,1]/4 signal.
# ---------------------------------------------------------------------------
histogram_start = text.index("def local_peak_signal(")
histogram_end = text.index("def deepest_component_point(", histogram_start)
histogram_replacement = '''def find_rightmost_histogram_peak(gray: np.ndarray) -> tuple[int, int]:\n    """Return the rightmost 3-bin-smoothed mode and its preceding left valley."""\n    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)\n    signal = np.convolve(histogram, PEAK_KERNEL, mode="same")\n\n    peaks: list[int] = []\n    for index in range(1, len(signal) - 1):\n        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:\n            peaks.append(index)\n    # Saturation itself is allowed to be the rightmost histogram mode.\n    if len(signal) >= 2 and signal[-1] > signal[-2]:\n        peaks.append(len(signal) - 1)\n    peak = max(peaks or [int(np.argmax(signal))])\n\n    left_edge = 0\n    for index in range(peak - 1, 0, -1):\n        if signal[index] <= signal[index - 1] and signal[index] < signal[index + 1]:\n            left_edge = index\n            break\n\n    return int(peak), int(left_edge)\n\n\n'''
text = text[:histogram_start] + histogram_replacement + text[histogram_end:]

# ---------------------------------------------------------------------------
# Coarse stage: seed discovery + downward tracking in one logical function.
# Tiny one-use component helpers are absorbed; topology is unchanged.
# ---------------------------------------------------------------------------
component_metadata_start = text.index("def _component_metadata(")
coarse_end = text.index("def _map_interval(", component_metadata_start)
coarse_replacement = '''def largest_enclosed_bright_component(binary: np.ndarray) -> np.ndarray | None:\n    """Return the largest 8-connected bright component enclosed by the raster."""\n    count, labels, stats, _ = cv2.connectedComponentsWithStats(\n        (binary != 0).astype(np.uint8), 8\n    )\n    height, width = binary.shape\n    best = None\n    for label in range(1, count):\n        x, y, component_width, component_height, area = map(int, stats[label])\n        if (\n            x == 0\n            or y == 0\n            or x + component_width >= width\n            or y + component_height >= height\n        ):\n            continue\n        candidate = (area, label)\n        if best is None or candidate > best:\n            best = candidate\n    if best is None:\n        return None\n    return labels == best[1]\n\n\ndef _touches_image_border(component: np.ndarray | None) -> bool:\n    """Return whether a tracked component is absent or reaches the image boundary."""\n    if component is None or not np.any(component):\n        return True\n    return bool(\n        np.any(component[0])\n        or np.any(component[-1])\n        or np.any(component[:, 0])\n        or np.any(component[:, -1])\n    )\n\n\ndef coarse_threshold_search(work_gray: np.ndarray) -> CoarseThresholdResult:\n    """Establish the solar seed, then track it to the lowest enclosed coarse T."""\n    # Phase 1: histogram identity/start only. Descend from the left edge of the\n    # rightmost mode until the first enclosed bright component identifies the Sun.\n    peak, left_edge = find_rightmost_histogram_peak(work_gray)\n    seed_threshold = None\n    seed_point = None\n    seed_mask = None\n\n    for threshold in range(left_edge, -1, -1):\n        component = largest_enclosed_bright_component(work_gray > threshold)\n        if component is None:\n            continue\n        seed_threshold = int(threshold)\n        seed_mask = np.where(component, 255, 0).astype(np.uint8)\n        seed_point = deepest_component_point(seed_mask)\n        break\n\n    if seed_threshold is None or seed_point is None or seed_mask is None:\n        raise ThresholdResolutionError("No enclosed bright component exists")\n\n    # Phase 2: keep the same seed and lower T. Connectivity is monotone while T is\n    # lowered: pixels are only added, so once this tracked 8-connected component\n    # reaches the image border it cannot become enclosed again at a lower T.\n    lowest_threshold = seed_threshold\n    lowest_component = seed_mask != 0\n    seed_x, seed_y = seed_point\n    height, width = work_gray.shape\n\n    for threshold in range(seed_threshold, -1, -1):\n        binary = work_gray > threshold\n        if not (0 <= seed_x < width and 0 <= seed_y < height) or not bool(binary[seed_y, seed_x]):\n            continue\n\n        flooded = cv2.compare(binary.astype(np.uint8), 0, cv2.CMP_GT)\n        cv2.floodFill(flooded, None, seed_point, 128, flags=8)\n        component = flooded == 128\n        if _touches_image_border(component):\n            break\n\n        lowest_threshold = threshold\n        lowest_component = component\n\n    ys, xs = np.nonzero(lowest_component)\n    if len(xs) == 0:\n        raise ThresholdResolutionError("Empty component")\n    x0, x1 = int(xs.min()), int(xs.max()) + 1\n    y0, y1 = int(ys.min()), int(ys.max()) + 1\n    component_area = int(len(xs))\n    component_bbox = (x0, y0, x1 - x0, y1 - y0)\n\n    return CoarseThresholdResult(\n        histogram_peak=peak,\n        histogram_left_edge=left_edge,\n        seed_threshold=seed_threshold,\n        seed_point=seed_point,\n        seed_mask=seed_mask,\n        threshold=int(lowest_threshold),\n        component_mask=np.where(lowest_component, 255, 0).astype(np.uint8),\n        component_area=component_area,\n        component_bbox=component_bbox,\n    )\n\n\n'''
text = text[:component_metadata_start] + coarse_replacement + text[coarse_end:]

# Clear, specific names for the substantial full-resolution stages.
text = text.replace("def establish_full_seed(\n", "def establish_full_resolution_seed(\n", 1)
text = text.replace("establish_full_seed(\n", "establish_full_resolution_seed(\n")

full_flood_start = text.index("def _full_flood_component(")
observation_start = text.index("def _build_observation_region(", full_flood_start)
full_component_replacement = '''def find_full_resolution_enclosed_seed_component(\n    full_gray: np.ndarray,\n    coarse_threshold: int,\n    seed_point: tuple[int, int],\n):\n    """Find an actual enclosed full-resolution component containing the solar seed."""\n    start_threshold = max(0, int(coarse_threshold))\n    seed_x, seed_y = seed_point\n\n    for threshold in range(start_threshold, 256):\n        binary = cv2.compare(full_gray, int(threshold), cv2.CMP_GT)\n        if binary[seed_y, seed_x] == 0:\n            continue\n\n        cv2.floodFill(binary, None, seed_point, 128, flags=8)\n        component = binary == 128\n        if not _touches_image_border(component):\n            return int(threshold), component\n\n        # Usually resolved at the coarse T or one level higher. The loop remains\n        # exhaustive so reduced-resolution mismatch cannot determine the final T.\n\n    raise ThresholdResolutionError(\n        "No enclosed original-resolution solar component"\n    )\n\n\n'''
text = text[:full_flood_start] + full_component_replacement + text[observation_start:]

text = text.replace("def _region_status(", "def _evaluate_observation_region(", 1)
text = text.replace("_region_status(", "_evaluate_observation_region(")
text = text.replace("_touches_border(", "_touches_image_border(")

# ---------------------------------------------------------------------------
# One grayscale-only public threshold API; conversion happens before it is called.
# ---------------------------------------------------------------------------
auto_start = text.index("def auto_threshold_from_gray(")
auto_end = text.index("class DetectorApp:", auto_start)
auto_replacement = '''def auto_threshold(gray: np.ndarray) -> AutoThresholdResult:\n    """Select T from an authoritative 8-bit grayscale brightness image."""\n    work_gray = resize_gray_max_dim(gray)\n    peak, left_edge = find_rightmost_histogram_peak(work_gray)\n\n    try:\n        coarse = coarse_threshold_search(work_gray)\n        full_seed = establish_full_resolution_seed(\n            gray, work_gray.shape, coarse.seed_mask, coarse.seed_threshold\n        )\n        roi_seed_threshold, roi_seed_component = (\n            find_full_resolution_enclosed_seed_component(\n                gray, coarse.threshold, full_seed\n            )\n        )\n        final_threshold, used_guard = find_lowest_full_threshold(\n            gray,\n            full_seed,\n            roi_seed_component,\n        )\n        return AutoThresholdResult(\n            threshold=final_threshold,\n            histogram_peak=coarse.histogram_peak,\n            histogram_left_edge=coarse.histogram_left_edge,\n            seed_threshold=coarse.seed_threshold,\n            coarse_threshold=coarse.threshold,\n            roi_seed_threshold=roi_seed_threshold,\n            full_seed_point=full_seed,\n            used_guard=used_guard,\n            resolved=True,\n        )\n    except ThresholdResolutionError as exc:\n        # Deterministic fallback: if component topology cannot be resolved, use\n        # the left side of the rightmost histogram peak rather than introducing\n        # color, Otsu, fixed-T, or ellipse-dependent heuristics.\n        return AutoThresholdResult(\n            threshold=int(left_edge),\n            histogram_peak=int(peak),\n            histogram_left_edge=int(left_edge),\n            seed_threshold=int(left_edge),\n            coarse_threshold=None,\n            roi_seed_threshold=None,\n            full_seed_point=None,\n            used_guard=False,\n            resolved=False,\n            reason=str(exc),\n        )\n\n\n'''
text = text[:auto_start] + auto_replacement + text[auto_end:]
text = text.replace("auto_threshold_from_gray(self.gray_image)", "auto_threshold(self.gray_image)")

# ---------------------------------------------------------------------------
# Application settings use ImageSettings rather than sparse string-key dictionaries.
# ---------------------------------------------------------------------------
old_defaults = '''        self.default_settings = {\n            "min_radius": int(self.min_radius.get()),\n            "max_radius": int(self.max_radius.get()),\n            "max_error": float(self.max_error.get()),\n            "min_coverage": int(self.min_coverage.get()),\n            "morphology": bool(self.morphology.get()),\n            "outer_limb_assistance": bool(self.outer_limb_assistance.get()),\n            "use_horizon": bool(self.use_horizon.get()),\n            "center_target": self.center_target.get(),\n        }\n'''
new_defaults = '''        self.default_settings = ImageSettings(\n            min_radius=int(self.min_radius.get()),\n            max_radius=int(self.max_radius.get()),\n            max_error=float(self.max_error.get()),\n            min_coverage=int(self.min_coverage.get()),\n            morphology=bool(self.morphology.get()),\n            outer_limb_assistance=bool(self.outer_limb_assistance.get()),\n            use_horizon=bool(self.use_horizon.get()),\n            center_target=self.center_target.get(),\n        )\n'''
assert old_defaults in text
text = text.replace(old_defaults, new_defaults, 1)

old_new_state = '''                state = {\n                    "settings": {},\n                    "auto_threshold_result": result,\n                }\n'''
new_new_state = '''                state = {\n                    "settings": ImageSettings(),\n                    "auto_threshold_result": result,\n                }\n'''
assert old_new_state in text
text = text.replace(old_new_state, new_new_state, 1)

old_restore = '''            settings = state["settings"]\n            for setting_name, variable in self.setting_variables.items():\n                if setting_name == "threshold":\n                    baseline = int(result.threshold)\n                else:\n                    baseline = self.default_settings[setting_name]\n                variable.set(settings.get(setting_name, baseline))\n'''
new_restore = '''            settings = state["settings"]\n            for setting_name, variable in self.setting_variables.items():\n                if setting_name == "threshold":\n                    baseline = int(result.threshold)\n                else:\n                    baseline = getattr(self.default_settings, setting_name)\n                override = getattr(settings, setting_name)\n                variable.set(baseline if override is None else override)\n'''
assert old_restore in text
text = text.replace(old_restore, new_restore, 1)

old_commit = '''            if setting_name == "threshold":\n                baseline = int(state["auto_threshold_result"].threshold)\n            else:\n                baseline = self.default_settings[setting_name]\n\n            if value == baseline:\n                settings.pop(setting_name, None)\n            else:\n                settings[setting_name] = value\n'''
new_commit = '''            if setting_name == "threshold":\n                baseline = int(state["auto_threshold_result"].threshold)\n            else:\n                baseline = getattr(self.default_settings, setting_name)\n\n            setattr(settings, setting_name, None if value == baseline else value)\n'''
assert old_commit in text
text = text.replace(old_commit, new_commit, 1)

text = text.replace(
    "# Per-image state keeps sparse setting overrides plus the cached automatic\n"
    "        # threshold result. No additional ImageState class is needed: the nested\n"
    "        # dictionaries directly express the state hierarchy.",
    "# Per-image state keeps a sparse ImageSettings instance plus the cached\n"
    "        # automatic threshold result. No ImageState wrapper class is needed: the\n"
    "        # outer dictionary directly expresses the image-to-state hierarchy.",
)

# ---------------------------------------------------------------------------
# Tests/diagnostics: follow the public API and test behavior, not helper layout.
# ---------------------------------------------------------------------------
tests = ROOT / "tests"
for path in tests.glob("*.py"):
    test_text = path.read_text(encoding="utf-8")
    test_text = test_text.replace("auto_threshold_from_gray", "auto_threshold")
    test_text = test_text.replace("rightmost_histogram_peak", "find_rightmost_histogram_peak")
    path.write_text(test_text, encoding="utf-8")

(tests / "test_histogram_peak_3bin.py").write_text('''"""Behavioral regression tests for the validated 3-bin histogram mode signal."""\n\nimport numpy as np\n\nimport circle_arc_detector as cad\n\n\ndef test_rightmost_mode_and_preceding_valley_are_selected():\n    gray = np.concatenate((\n        np.full(100, 40, dtype=np.uint8),\n        np.full(200, 100, dtype=np.uint8),\n    ))\n    assert cad.find_rightmost_histogram_peak(gray) == (100, 98)\n\n\ndef test_saturation_can_be_the_rightmost_mode():\n    gray = np.concatenate((\n        np.full(50, 100, dtype=np.uint8),\n        np.full(200, 255, dtype=np.uint8),\n    ))\n    peak, left_edge = cad.find_rightmost_histogram_peak(gray)\n    assert peak == 255\n    assert 0 <= left_edge < peak\n''', encoding="utf-8")

threshold_test = tests / "test_threshold_finder.py"
threshold_text = threshold_test.read_text(encoding="utf-8")
negative_start = threshold_text.find("\ndef test_no_color_or_competitor_gain_path_exists():")
if negative_start != -1:
    threshold_text = threshold_text[:negative_start].rstrip() + "\n"
threshold_test.write_text(threshold_text, encoding="utf-8")

(tests / "test_gui_threshold_integration.py").write_text('''"""Behavioral integration checks for per-image settings and threshold preview routing."""\n\nimport numpy as np\n\nimport circle_arc_detector as appmod\n\n\nclass FakeVariable:\n    def __init__(self, value):\n        self.value = value\n\n    def get(self):\n        return self.value\n\n    def set(self, value):\n        self.value = value\n\n\ndef make_state_app():\n    app = object.__new__(appmod.DetectorApp)\n    app.current_path = "/tmp/image.jpg"\n    app.gray_image = np.array([[0, 9, 10, 255]], dtype=np.uint8)\n    app.threshold = FakeVariable(10)\n    app.min_radius = FakeVariable(1000)\n    app.max_radius = FakeVariable(1500)\n    app.max_error = FakeVariable(8.0)\n    app.min_coverage = FakeVariable(8)\n    app.morphology = FakeVariable(False)\n    app.outer_limb_assistance = FakeVariable(False)\n    app.use_horizon = FakeVariable(True)\n    app.center_target = FakeVariable("light")\n    app.default_settings = appmod.ImageSettings(\n        min_radius=1000,\n        max_radius=1500,\n        max_error=8.0,\n        min_coverage=8,\n        morphology=False,\n        outer_limb_assistance=False,\n        use_horizon=True,\n        center_target="light",\n    )\n    app.setting_variables = {\n        "threshold": app.threshold,\n        "min_radius": app.min_radius,\n        "max_radius": app.max_radius,\n        "max_error": app.max_error,\n        "min_coverage": app.min_coverage,\n        "morphology": app.morphology,\n        "outer_limb_assistance": app.outer_limb_assistance,\n        "use_horizon": app.use_horizon,\n        "center_target": app.center_target,\n    }\n    result = appmod.AutoThresholdResult(\n        threshold=10,\n        histogram_peak=20,\n        histogram_left_edge=10,\n        seed_threshold=15,\n        coarse_threshold=12,\n        roi_seed_threshold=12,\n        full_seed_point=(1, 1),\n        used_guard=False,\n        resolved=True,\n    )\n    app.image_state = {\n        app.current_path: {\n            "settings": appmod.ImageSettings(),\n            "auto_threshold_result": result,\n        }\n    }\n    app.refresh_count = 0\n    app._refresh_threshold_image = lambda: setattr(\n        app, "refresh_count", app.refresh_count + 1\n    )\n    return app\n\n\ndef test_each_processing_setting_is_stored_only_when_it_differs_from_baseline():\n    app = make_state_app()\n    changes = {\n        "threshold": 14,\n        "min_radius": 900,\n        "max_radius": 1400,\n        "max_error": 9.5,\n        "min_coverage": 12,\n        "morphology": True,\n        "outer_limb_assistance": True,\n        "use_horizon": False,\n        "center_target": "dark",\n    }\n\n    settings = app.image_state[app.current_path]["settings"]\n    for setting_name, value in changes.items():\n        app.setting_variables[setting_name].set(value)\n        app._commit_setting_change(setting_name)\n        assert getattr(settings, setting_name) == value\n\n    assert app.refresh_count == len(changes)\n\n\ndef test_returning_settings_to_their_baselines_clears_the_sparse_overrides():\n    app = make_state_app()\n    settings = app.image_state[app.current_path]["settings"]\n\n    app.min_radius.set(900)\n    app._commit_setting_change("min_radius")\n    assert settings.min_radius == 900\n    app.min_radius.set(1000)\n    app._commit_setting_change("min_radius")\n    assert settings.min_radius is None\n\n    app.threshold.set(14)\n    app._commit_setting_change("threshold")\n    assert settings.threshold == 14\n    app.threshold.set(10)\n    app._commit_setting_change("threshold")\n    assert settings.threshold is None\n\n\ndef test_auto_select_reuses_cached_result_without_rerunning_algorithm(monkeypatch):\n    app = make_state_app()\n    app.status = FakeVariable("")\n    app.threshold.set(14)\n    commits = []\n    app._commit_setting_change = lambda name: commits.append(name)\n\n    def fail_if_called(_gray):\n        raise AssertionError("cached AutoThresholdResult should have been reused")\n\n    monkeypatch.setattr(appmod, "auto_threshold", fail_if_called)\n    app.auto_select_threshold()\n\n    assert app.threshold.get() == 10\n    assert commits == ["threshold"]\n\n\ndef test_bw_renderer_uses_exact_gray_greater_than_t_semantics():\n    app = make_state_app()\n    app.threshold_preview = appmod.transparent_bgra()\n    app.threshold_photo = None\n    app._refresh_threshold_image = appmod.DetectorApp._refresh_threshold_image.__get__(app)\n    app._refresh_threshold_image()\n    assert app.threshold_preview.shape == (1, 4, 4)\n    assert app.threshold_preview[0, 0, 0] == 0\n    assert app.threshold_preview[0, 1, 0] == 0\n    assert app.threshold_preview[0, 2, 0] == 0\n    assert app.threshold_preview[0, 3, 0] == 255\n    assert np.all(app.threshold_preview[:, :, 3] == 255)\n''', encoding="utf-8")

assert text != original
SOURCE.write_text(text, encoding="utf-8")
print("Applied approved threshold sanitation stage 2")
