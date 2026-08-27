"""Make threshold Auto select regenerate the preview like other completed controls.

The automatic threshold algorithm is unchanged. After it computes/stores T and sets
the slider variable, this patch calls the existing ``refresh_preview`` path so the
actual black/white threshold raster is regenerated immediately.
"""
from pathlib import Path

SOURCE = Path("circle_arc_detector.py")
INTEGRATION_TEST = Path("tests/test_gui_threshold_integration.py")
DEFERRED_TEST = Path("tests/test_threshold_gui_deferred_preview_refresh.py")
FOCUS_TEST = Path("tests/test_gui_focus_autoselect.py")

text = SOURCE.read_text(encoding="utf-8")

old_doc = '''Setting controls regenerate the threshold preview after discrete changes. Slider
motion only updates the displayed value; preview refresh is deferred until mouse
release or the final keyboard key release, while checkbox/radio changes refresh
immediately. Threshold Auto select remains an explicit no-preview-regeneration action.
'''
new_doc = '''Setting controls regenerate the threshold preview after discrete changes. Slider
motion only updates the displayed value; preview refresh is deferred until mouse
release or the final keyboard key release, while checkbox/radio and threshold
Auto select changes refresh immediately.
'''
assert old_doc in text
text = text.replace(old_doc, new_doc, 1)

old_auto = '''    def auto_select_threshold(self):
        """Rerun image-only automatic T selection without regenerating the preview."""
        if self.gray_image is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        result = auto_threshold_from_gray(self.gray_image)
        selected_threshold = int(result.threshold)
        if self.current_path is not None:
            self.image_thresholds[self.current_path] = selected_threshold
            self.image_auto_results[self.current_path] = result

        # The threshold variable trace updates only the displayed slider value.
        # Per explicit user requirement, Auto select itself DOES NOT regenerate the
        # preview; Refresh Preview / Apply Full Resolution remain available actions.
        self.threshold.set(selected_threshold)
        if result.resolved:
            self.status.set(
                "Automatic grayscale threshold selected: "
                f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                f"histogram start={result.histogram_left_edge}). "
                "Preview not regenerated."
            )
        else:
            self.status.set(
                "Automatic component tracking unresolved; "
                f"using rightmost-histogram left edge T={selected_threshold}. "
                "Preview not regenerated."
            )
'''
new_auto = '''    def auto_select_threshold(self):
        """Rerun image-only automatic T selection and regenerate the preview."""
        if self.gray_image is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        result = auto_threshold_from_gray(self.gray_image)
        selected_threshold = int(result.threshold)
        if self.current_path is not None:
            self.image_thresholds[self.current_path] = selected_threshold
            self.image_auto_results[self.current_path] = result

        # The threshold variable trace updates only the displayed slider value.
        # Auto select is a completed setting change, so regenerate through the same
        # refresh path used by the other controls after storing the new T.
        self.threshold.set(selected_threshold)
        self.refresh_preview()
        if result.resolved:
            self.status.set(
                "Automatic grayscale threshold selected: "
                f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                f"histogram start={result.histogram_left_edge})."
            )
        else:
            self.status.set(
                "Automatic component tracking unresolved; "
                f"using rightmost-histogram left edge T={selected_threshold}."
            )
'''
assert old_auto in text
text = text.replace(old_auto, new_auto, 1)
SOURCE.write_text(text, encoding="utf-8")

# Update the integration regression to require the shared refresh path.
it = INTEGRATION_TEST.read_text(encoding="utf-8")
it = it.replace(
    "def test_auto_select_recomputes_threshold_without_regenerating_preview():",
    "def test_auto_select_recomputes_threshold_and_regenerates_preview():",
)
it = it.replace(
    '''    assert "self.render_threshold_preview()" not in block
    assert "Preview not regenerated." in block
''',
    '''    assert "self.refresh_preview()" in block
    assert "Preview not regenerated." not in block
''',
)
INTEGRATION_TEST.write_text(it, encoding="utf-8")

# Update the deferred-event regression: Auto select is now an immediate completed
# setting change, while slider mouse/key behavior remains deferred/coalesced.
dt = DEFERRED_TEST.read_text(encoding="utf-8")
old_test = '''def test_auto_select_keeps_explicit_no_preview_regeneration_rule():
    auto = block("    def auto_select_threshold(self):", "    def auto_select_radius")
    assert "auto_threshold_from_gray(self.gray_image)" in auto
    assert "self.threshold.set(selected_threshold)" in auto
    assert "self.render_threshold_preview()" not in auto
    assert "self.refresh_preview()" not in auto
    assert "Preview not regenerated." in auto
'''
new_test = '''def test_auto_select_refreshes_preview_immediately_after_selecting_t():
    auto = block("    def auto_select_threshold(self):", "    def auto_select_radius")
    assert "auto_threshold_from_gray(self.gray_image)" in auto
    assert "self.threshold.set(selected_threshold)" in auto
    assert "self.refresh_preview()" in auto
    assert auto.index("self.threshold.set(selected_threshold)") < auto.index("self.refresh_preview()")
    assert "Preview not regenerated." not in auto
'''
assert old_test in dt
dt = dt.replace(old_test, new_test, 1)
DEFERRED_TEST.write_text(dt, encoding="utf-8")

# Update the older Auto-select GUI regression that encoded the same superseded rule.
ft = FOCUS_TEST.read_text(encoding="utf-8")
old_focus = '''def test_threshold_auto_select_is_implemented_without_regenerating_preview():
    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius", 1
    )[0]
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.threshold.set(selected_threshold)" in block
    assert "self.render_threshold_preview()" not in block
    assert "Preview not regenerated." in block
'''
new_focus = '''def test_threshold_auto_select_is_implemented_and_refreshes_preview():
    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius", 1
    )[0]
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.threshold.set(selected_threshold)" in block
    assert "self.refresh_preview()" in block
    assert "Preview not regenerated." not in block
'''
assert old_focus in ft
ft = ft.replace(old_focus, new_focus, 1)
FOCUS_TEST.write_text(ft, encoding="utf-8")

print("Applied immediate preview refresh after threshold Auto select")
