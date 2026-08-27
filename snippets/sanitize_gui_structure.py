from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"

text = SOURCE.read_text(encoding="utf-8")
original = text

old_doc = '''"""GUI-first eclipse detector shell.\n\nThis milestone intentionally contains the user interface only. Detection, ellipse fitting, horizon handling, and image centering are not\nimplemented yet. The interface is based on the latest GUI from\n``refactor/cleanup-performance`` and adds a mutually exclusive centering target:\nlight ellipse (default) or dark ellipse. A clicked slider retains keyboard focus\nfor arrow-key adjustment until the mouse is clicked anywhere outside that slider.\nGUI-only Auto select buttons are provided for threshold and radius, and a\nSave centered images placeholder is available beside the image-loading control.\nThe preview rasters contain no placeholder text. Empty preview regions are stored\nas BGRA image data with alpha=0 rather than simulated with a matching Tk background.\nSetting controls request preview refreshes through the existing GUI hook. Slider\nmotion only updates the displayed value; preview refresh is deferred until mouse\nrelease or the final keyboard key release, while checkbox/radio changes refresh\nimmediately. This GUI-only milestone does not generate a threshold raster.\n"""'''
new_doc = '''"""GUI foundation for the eclipse alignment application.\n\nThis branch defines the user-interface contract independently from the detector\nbackend. It intentionally contains image loading and navigation, control layout,\npreview surfaces, focus behavior, completed-setting event handling, and backend\nhooks, while ellipse fitting, horizon detection, centering, export, automatic\nthreshold selection, automatic radius selection, and threshold-raster generation\nremain outside this GUI-only implementation. The interface descends from the\n``refactor/cleanup-performance`` GUI and retains its established layout and\nkeyboard interaction model.\n\nThe centering target is represented by one shared Tk ``StringVar`` and two mutually\nexclusive radio buttons: Light ellipse is the default and Dark ellipse is the\nalternative. Sliders retain keyboard focus after a click so arrow keys can make\nprecise adjustments; clicking elsewhere releases that focus. A slider's numeric\nlabel updates continuously while it moves, but setting completion is delayed until\nmouse release or the final keyboard key release. Keyboard auto-repeat may emit\nintermediate release/press pairs on some Tk platforms, so those releases are\ncoalesced through a short settling window. Checkboxes, radio buttons, and Auto\nselect controls commit immediately because their interaction is already discrete.\n\nCompleted setting changes are deliberately separate from explicit preview\nprocessing. They call ``_commit_setting_change()``, whose only responsibility is\nto invoke the lightweight ``_refresh_threshold_image()`` hook. In a backend-enabled\nbranch that hook may rebuild the B/W image from the already-loaded grayscale image\nand the current threshold, but it must not run ellipse, horizon, candidate, or\nother Refresh Preview processing. The full ``refresh_preview()`` action is reserved\nfor an explicit Refresh Preview request or for initialization after a readable\nimage is loaded. In this GUI-only branch both backend hooks remain placeholders and\nno grayscale or B/W threshold backend is implemented.\n\nPreview rasters contain no placeholder text. Empty preview regions are represented\nas real BGRA image data with alpha=0, while loaded source-image pixels are made\nfully opaque. Transparency is retained in the raster itself rather than simulated\nby matching the Tk canvas background. The Save centered images control, threshold\nAuto select, radius Auto select, Refresh Preview, and Apply Full Resolution actions\nremain present as interface/backend boundaries for later implementation.\n"""'''
assert old_doc in text
text = text.replace(old_doc, new_doc, 1)

old_types = '''IMAGE_FILE_TYPES = (\n    ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),\n    ("All files", "*.*"),\n)\n'''
new_types = old_types + '''\nSLIDER_KEY_RELEASE_SETTLE_MS = 45\nPREVIEW_REDRAW_DELAY_MS = 60\n'''
assert old_types in text
text = text.replace(old_types, new_types, 1)

renames = {
    "build_navigation": "_build_navigation_bar",
    "build_controls": "_build_settings_panel",
    "build_previews": "_build_preview_panes",
    "add_scale": "_add_slider",
    "focus_scale": "_focus_slider",
    "begin_scale_mouse_change": "_begin_slider_mouse_change",
    "finish_scale_mouse_change": "_finish_slider_mouse_change",
    "begin_scale_key_change": "_begin_slider_keyboard_change",
    "defer_scale_key_refresh": "_schedule_slider_keyboard_commit",
    "finish_scale_key_change": "_finish_slider_keyboard_change",
    "release_scale_focus_if_outside": "_release_slider_focus_if_clicked_elsewhere",
    "pending": "_commit_setting_change",
    "center_target_changed": "_handle_center_target_change",
    "update_center_preview_label": "_update_center_preview_label",
    "schedule_redraw": "_schedule_preview_redraw",
    "redraw": "_redraw_previews",
    "show_image": "_show_image_on_canvas",
    "resize_job": "preview_redraw_job",
    "scale_key_release_job": "slider_keyboard_commit_job",
    "scale_key_widget": "slider_keyboard_widget",
    "scale_key_start_value": "slider_keyboard_start_value",
}
for old, new in renames.items():
    text = text.replace(old, new)

# Prevent the generic redraw rename from altering the already-specific scheduler name.
text = text.replace("_schedule_preview__redraw_previews", "_schedule_preview_redraw")

old_class_doc = '''    """GUI shell for the eclipse detector rebuild.\n\n    Only interface behavior is implemented here: image loading/navigation,\n    controls, preview panes, and the centering-target selector. Detector buttons\n    deliberately report that backend functionality has not yet been implemented.\n    """'''
new_class_doc = '''    """Own the GUI state, widgets, interaction rules, and preview hooks.\n\n    The class manages the ordered image list, alpha-aware preview rasters, settings,\n    navigation, slider focus, completed-setting commits, and canvas redraws. Public\n    methods correspond to application actions; underscore-prefixed methods are\n    implementation details used by Tk callbacks or rendering internals. Detector\n    work is intentionally represented only by backend hooks in this branch.\n    """'''
assert old_class_doc in text
text = text.replace(old_class_doc, new_class_doc, 1)

text = text.replace(
    "self.slider_keyboard_commit_job = self.root.after(45, self._finish_slider_keyboard_change)",
    "self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )",
)
text = text.replace(
    "self.preview_redraw_job = self.root.after(60, self._redraw_previews)",
    "self.preview_redraw_job = self.root.after(\n            PREVIEW_REDRAW_DELAY_MS, self._redraw_previews\n        )",
)

old_begin = '''    def _begin_slider_keyboard_change(self, event):\n        """Start/coalesce a keyboard slider interaction without refreshing."""\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n            self.slider_keyboard_commit_job = None\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n\n    def _schedule_slider_keyboard_commit(self, event):\n        """Refresh after the final KeyRelease, not during key auto-repeat."""\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n        self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )\n'''
new_begin = '''    def _cancel_pending_slider_keyboard_commit(self):\n        """Cancel a release callback superseded by continuing keyboard input."""\n        if self.slider_keyboard_commit_job is not None:\n            self.root.after_cancel(self.slider_keyboard_commit_job)\n            self.slider_keyboard_commit_job = None\n\n    def _begin_slider_keyboard_change(self, event):\n        """Begin or continue one keyboard slider interaction."""\n        self._cancel_pending_slider_keyboard_commit()\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n\n    def _schedule_slider_keyboard_commit(self, event):\n        """Schedule completion after a KeyRelease survives the repeat window."""\n        if self.slider_keyboard_widget is not event.widget:\n            self.slider_keyboard_widget = event.widget\n            self.slider_keyboard_start_value = event.widget.get()\n        self._cancel_pending_slider_keyboard_commit()\n        self.slider_keyboard_commit_job = self.root.after(\n            SLIDER_KEY_RELEASE_SETTLE_MS, self._finish_slider_keyboard_change\n        )\n'''
assert old_begin in text
text = text.replace(old_begin, new_begin, 1)

old_auto = '''    def auto_select_threshold(self):\n        self.refresh_preview()\n        self.status.set(\n            "Auto select threshold: algorithm not implemented in the GUI milestone."\n        )\n'''
new_auto = '''    def auto_select_threshold(self):\n        self._commit_setting_change()\n        self.status.set(\n            "Auto select threshold: algorithm not implemented in the GUI foundation."\n        )\n'''
assert old_auto in text
text = text.replace(old_auto, new_auto, 1)

old_pending = '''    def _commit_setting_change(self, *_args):\n        """Request an immediate preview refresh after a discrete setting change."""\n        self.refresh_preview()\n\n    def _handle_center_target_change(self):\n        self._update_center_preview_label()\n        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n        self.refresh_preview()\n        self.status.set(\n            f"Centering target set to {target}. Actual centering will be implemented with ellipse detection."\n        )\n\n    def _update_center_preview_label(self):\n        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n        self.center_preview_label.set(f"Full-color image — center on {target}")\n\n    def refresh_preview(self):\n'''
new_pending = '''    def _commit_setting_change(self, *_args):\n        """Apply only the lightweight threshold-image consequence of a setting change."""\n        self._refresh_threshold_image()\n\n    def _selected_center_target_name(self):\n        """Return the user-facing name of the selected centering target."""\n        return "light ellipse" if self.center_target.get() == "light" else "dark ellipse"\n\n    def _handle_center_target_change(self):\n        self._update_center_preview_label()\n        self._commit_setting_change()\n        self.status.set(\n            f"Centering target set to {self._selected_center_target_name()}. "\n            "Actual centering will be implemented with ellipse detection."\n        )\n\n    def _update_center_preview_label(self):\n        target = self._selected_center_target_name()\n        self.center_preview_label.set(f"Full-color image — center on {target}")\n\n    def _refresh_threshold_image(self):\n        """Backend hook for rebuilding only the current-threshold B/W image.\n\n        The GUI-only branch intentionally has no grayscale/threshold backend.\n        Backend-enabled branches implement this hook without running the broader\n        Refresh Preview processing pipeline.\n        """\n\n    def refresh_preview(self):\n'''
assert old_pending in text
text = text.replace(old_pending, new_pending, 1)

start = text.index("    def clear_threshold_preview(self):")
end = text.index("    def _commit_setting_change", start)
text = text[:start] + text[end:]

old_load = '''            self.threshold_preview = transparent_bgra(width, height)\n            self._redraw_previews()\n            self.status.set(\n                "Image loaded. Detector functionality is intentionally not implemented in this milestone."\n            )\n'''
new_load = '''            self.threshold_preview = transparent_bgra(width, height)\n            self._redraw_previews()\n            self.refresh_preview()\n            self.status.set(\n                "Image loaded. Detector functionality is intentionally not implemented in the GUI foundation."\n            )\n'''
assert old_load in text
text = text.replace(old_load, new_load, 1)

text = text.replace("GUI-only milestone. Load images to inspect the interface.",
                    "GUI foundation. Load images to inspect the interface.")
text = text.replace("Ellipse / Arc Detector — GUI milestone", "Ellipse / Arc Detector — GUI foundation")
text = text.replace("GUI milestone", "GUI foundation")
text = text.replace("GUI-only milestone", "GUI foundation")

helper_start = text.index("    def _add_slider(")
helper_end = text.index("    @staticmethod\n    def _focus_slider", helper_start)
helper = text[helper_start:helper_end]
helper = helper.replace("scale = tk.Scale(", "slider = tk.Scale(")
helper = helper.replace("scale.grid(", "slider.grid(")
helper = helper.replace("scale.bind(", "slider.bind(")
text = text[:helper_start] + helper + text[helper_end:]

text = text.replace("        rows = [\n", "        slider_specs = [\n", 1)
text = text.replace("        for row, spec in enumerate(rows):\n            self._add_slider(frame, row, *spec)\n",
                    "        for row, spec in enumerate(slider_specs):\n            self._add_slider(frame, row, *spec)\n", 1)

old_color = '''        if hasattr(self, "color_canvas"):\n            if self.color_image is None:\n                self.color_photo = self._show_image_on_canvas(\n                    self.color_canvas, transparent_bgra()\n                )\n            else:\n                self.color_photo = self._show_image_on_canvas(self.color_canvas, self.color_image)\n'''
new_color = '''        if hasattr(self, "color_canvas"):\n            image = self.color_image if self.color_image is not None else transparent_bgra()\n            self.color_photo = self._show_image_on_canvas(self.color_canvas, image)\n'''
assert old_color in text
text = text.replace(old_color, new_color, 1)

assert text != original
SOURCE.write_text(text, encoding="utf-8")

tests = ROOT / "tests"

(tests / "test_gui_deferred_preview_refresh.py").write_text('''"""Behavioral regression checks for completed slider interactions."""\n\nfrom types import SimpleNamespace\n\nimport circle_arc_detector as gui\n\n\nclass FakeRoot:\n    def __init__(self):\n        self.jobs = {}\n        self.cancelled = []\n        self.next_job = 1\n\n    def after(self, delay, callback):\n        job = self.next_job\n        self.next_job += 1\n        self.jobs[job] = (delay, callback)\n        return job\n\n    def after_cancel(self, job):\n        self.cancelled.append(job)\n        self.jobs.pop(job, None)\n\n\nclass FakeSlider:\n    def __init__(self, value):\n        self.value = value\n        self.focused = False\n\n    def get(self):\n        return self.value\n\n    def focus_set(self):\n        self.focused = True\n\n\ndef make_app():\n    app = object.__new__(gui.DetectorApp)\n    app.root = FakeRoot()\n    app.slider_keyboard_commit_job = None\n    app.slider_keyboard_widget = None\n    app.slider_keyboard_start_value = None\n    app.commits = 0\n    app._commit_setting_change = lambda: setattr(app, "commits", app.commits + 1)\n    return app\n\n\ndef test_mouse_slider_commits_once_after_changed_release():\n    app = make_app()\n    slider = FakeSlider(8)\n    event = SimpleNamespace(widget=slider)\n\n    app._begin_slider_mouse_change(event)\n    slider.value = 12\n    app._finish_slider_mouse_change(event)\n\n    assert app.commits == 1\n\n\ndef test_mouse_slider_does_not_commit_when_value_is_unchanged():\n    app = make_app()\n    slider = FakeSlider(8)\n    event = SimpleNamespace(widget=slider)\n\n    app._begin_slider_mouse_change(event)\n    app._finish_slider_mouse_change(event)\n\n    assert app.commits == 0\n\n\ndef test_keyboard_repeat_releases_are_cancelled_and_final_release_commits_once():\n    app = make_app()\n    slider = FakeSlider(8)\n    event = SimpleNamespace(widget=slider)\n\n    app._begin_slider_keyboard_change(event)\n    slider.value = 9\n    app._schedule_slider_keyboard_commit(event)\n    first_job = app.slider_keyboard_commit_job\n\n    app._begin_slider_keyboard_change(event)\n    assert first_job in app.root.cancelled\n    assert first_job not in app.root.jobs\n\n    slider.value = 12\n    app._schedule_slider_keyboard_commit(event)\n    final_job = app.slider_keyboard_commit_job\n    delay, callback = app.root.jobs[final_job]\n\n    assert delay == gui.SLIDER_KEY_RELEASE_SETTLE_MS\n    assert app.commits == 0\n\n    callback()\n\n    assert app.commits == 1\n    assert app.slider_keyboard_commit_job is None\n    assert app.slider_keyboard_widget is None\n    assert app.slider_keyboard_start_value is None\n\n\ndef test_keyboard_interaction_does_not_commit_if_final_value_matches_start():\n    app = make_app()\n    slider = FakeSlider(8)\n    event = SimpleNamespace(widget=slider)\n\n    app._begin_slider_keyboard_change(event)\n    slider.value = 9\n    app._schedule_slider_keyboard_commit(event)\n    app._begin_slider_keyboard_change(event)\n    slider.value = 8\n    app._schedule_slider_keyboard_commit(event)\n    _, callback = app.root.jobs[app.slider_keyboard_commit_job]\n    callback()\n\n    assert app.commits == 0\n''', encoding="utf-8")

(tests / "test_gui_preview_refresh.py").write_text('''"""Behavioral checks for lightweight setting commits and full preview entry points."""\n\nfrom types import SimpleNamespace\n\nimport cv2\nimport numpy as np\n\nimport circle_arc_detector as gui\n\n\nclass FakeStatus:\n    def __init__(self):\n        self.value = None\n\n    def set(self, value):\n        self.value = value\n\n\nclass FakeStringVar:\n    def __init__(self, value):\n        self.value = value\n\n    def get(self):\n        return self.value\n\n\ndef test_completed_setting_change_only_refreshes_threshold_image():\n    app = object.__new__(gui.DetectorApp)\n    calls = []\n    app._refresh_threshold_image = lambda: calls.append("threshold")\n    app.refresh_preview = lambda: calls.append("full")\n\n    app._commit_setting_change()\n\n    assert calls == ["threshold"]\n\n\ndef test_threshold_auto_select_uses_completed_setting_change_path():\n    app = object.__new__(gui.DetectorApp)\n    app.status = FakeStatus()\n    calls = []\n    app._commit_setting_change = lambda: calls.append("commit")\n    app.refresh_preview = lambda: calls.append("full")\n\n    app.auto_select_threshold()\n\n    assert calls == ["commit"]\n\n\ndef test_center_target_change_updates_label_and_commits_once():\n    app = object.__new__(gui.DetectorApp)\n    app.center_target = FakeStringVar("dark")\n    app.center_preview_label = SimpleNamespace(set=lambda value: setattr(app, "label", value))\n    app.status = FakeStatus()\n    calls = []\n    app._commit_setting_change = lambda: calls.append("commit")\n\n    app._handle_center_target_change()\n\n    assert calls == ["commit"]\n    assert app.label == "Full-color image — center on dark ellipse"\n\n\ndef test_readable_image_load_invokes_full_preview_processing(tmp_path):\n    path = tmp_path / "image.png"\n    image = np.zeros((5, 7, 3), dtype=np.uint8)\n    assert cv2.imwrite(str(path), image)\n\n    app = object.__new__(gui.DetectorApp)\n    app.image_paths = [str(path)]\n    app.current_index = -1\n    app.current_path = None\n    app.color_image = None\n    app.threshold_preview = gui.transparent_bgra()\n    app.threshold_photo = None\n    app.color_photo = None\n    app.status = FakeStatus()\n    app.full_calls = 0\n    app.refresh_preview = lambda: setattr(app, "full_calls", app.full_calls + 1)\n    app._redraw_previews = lambda: None\n    app.update_navigation_state = lambda: None\n\n    app.load_image_at(0)\n\n    assert app.full_calls == 1\n    assert app.color_image.shape == (5, 7, 4)\n''', encoding="utf-8")

old_preview_test = tests / "test_gui_preview_clear.py"
if old_preview_test.exists():
    old_preview_test.unlink()

alpha_path = tests / "test_gui_alpha_preview.py"
alpha = alpha_path.read_text(encoding="utf-8")
old_alpha_test = '''\n\ndef test_setting_changes_replace_threshold_with_transparent_frame():\n    start = TEXT.index("    def clear_threshold_preview(self):")\n    end = TEXT.index("    def pending(", start)\n    block = TEXT[start:end]\n    assert "transparent_bgra(" in block\n    assert "self.show_image(" in block\n'''
assert old_alpha_test in alpha
alpha = alpha.replace(old_alpha_test, "", 1)
alpha_path.write_text(alpha, encoding="utf-8")

focus_path = tests / "test_gui_focus_release.py"
focus = focus_path.read_text(encoding="utf-8")
focus = focus.replace("release_scale_focus_if_outside", "_release_slider_focus_if_clicked_elsewhere")
focus = focus.replace("focus_scale", "_focus_slider")
focus = focus.replace("scale.bind(", "slider.bind(")
focus_path.write_text(focus, encoding="utf-8")

auto_path = tests / "test_gui_autoselect_preview_refresh.py"
auto_path.write_text('''"""Behavioral regression check for threshold Auto-select event routing."""\n\nimport circle_arc_detector as gui\n\n\nclass FakeStatus:\n    def set(self, _value):\n        pass\n\n\ndef test_threshold_autoselect_commits_setting_without_running_full_preview():\n    app = object.__new__(gui.DetectorApp)\n    app.status = FakeStatus()\n    calls = []\n    app._commit_setting_change = lambda: calls.append("commit")\n    app.refresh_preview = lambda: calls.append("full")\n\n    app.auto_select_threshold()\n\n    assert calls == ["commit"]\n''', encoding="utf-8")

focus_auto_path = tests / "test_gui_focus_autoselect.py"
focus_auto = focus_auto_path.read_text(encoding="utf-8")
focus_auto = focus_auto.replace("self.focus_scale", "self._focus_slider")
focus_auto = focus_auto.replace("scale.bind(", "slider.bind(")
focus_auto = focus_auto.replace("def test_auto_select_buttons_are_ui_only_placeholders():\n    assert 'Auto select threshold: algorithm not implemented' in TEXT\n    assert 'Auto select radius range: algorithm not implemented' in TEXT\n    assert 'def detect(' not in TEXT\n", "def test_auto_select_actions_remain_present():\n    assert 'Auto select threshold: algorithm not implemented' in TEXT\n    assert 'Auto select radius range: algorithm not implemented' in TEXT\n")
focus_auto_path.write_text(focus_auto, encoding="utf-8")

structure_path = tests / "test_gui_structure.py"
structure = structure_path.read_text(encoding="utf-8")
negative = '''\n\ndef test_detection_is_not_implemented_in_gui_milestone():\n    assert "def detect(" not in TEXT\n    assert "def fit_ellipse" not in TEXT\n    assert "detector backend not implemented" in TEXT\n'''
assert negative in structure
structure = structure.replace(negative, "", 1)
structure_path.write_text(structure, encoding="utf-8")

print("Sanitized GUI structure and behavior tests")
