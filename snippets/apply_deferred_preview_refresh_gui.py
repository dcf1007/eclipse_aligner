"""Apply deferred preview-refresh event semantics to the GUI-only milestone.

This is intentionally GUI-only. It does not add grayscale conversion, threshold
raster generation, detection, or any other image-processing backend. It changes
only when the existing ``refresh_preview`` hook is called:

* slider variable traces update only their numeric labels;
* mouse slider changes refresh once after ButtonRelease;
* keyboard slider changes coalesce auto-repeat and refresh after final KeyRelease;
* checkbox/radio setting changes refresh immediately.
"""
from pathlib import Path

SOURCE = Path("circle_arc_detector.py")
TEST = Path("tests/test_gui_preview_clear.py")
text = SOURCE.read_text(encoding="utf-8")

old_doc = '''The preview rasters contain no placeholder text. Empty preview regions are stored
as BGRA image data with alpha=0 rather than simulated with a matching Tk background.
Any setting change replaces the threshold preview with a fully transparent BGRA
frame until the next explicit Refresh Preview or Apply Full Resolution action.
'''
new_doc = '''The preview rasters contain no placeholder text. Empty preview regions are stored
as BGRA image data with alpha=0 rather than simulated with a matching Tk background.
Setting controls request preview refreshes through the existing GUI hook. Slider
motion only updates the displayed value; preview refresh is deferred until mouse
release or the final keyboard key release, while checkbox/radio changes refresh
immediately. This GUI-only milestone does not generate a threshold raster.
'''
assert old_doc in text
text = text.replace(old_doc, new_doc, 1)

old_state = '''        self.threshold_photo = None
        self.color_photo = None
        self.resize_job = None
'''
new_state = '''        self.threshold_photo = None
        self.color_photo = None
        self.resize_job = None

        # Keyboard auto-repeat can emit intermediate release/press pairs on some
        # Tk platforms. Keep one deferred refresh job so only the final key-up
        # commits a slider-driven preview refresh.
        self.scale_key_release_job = None
        self.scale_key_widget = None
        self.scale_key_start_value = None
'''
assert old_state in text
text = text.replace(old_state, new_state, 1)

old_scale = '''        # Tk Scale supports precise arrow-key adjustment while it owns keyboard
        # focus. Explicitly focus the clicked scale and leave focus there after the
        # mouse interaction, instead of relying on platform-specific focus policy.
        scale.bind("<ButtonPress-1>", self.focus_scale, add="+")
        scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")
        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            value_label.config(text=formatter(variable.get()))
            self.pending()

        variable.trace_add("write", update_value)
        update_value()
'''
new_scale = '''        # Tk Scale supports precise arrow-key adjustment while it owns keyboard
        # focus. Value traces update only the label. Preview refresh is deliberately
        # deferred until the user finishes the mouse or keyboard interaction.
        scale.bind("<ButtonPress-1>", self.focus_scale, add="+")
        scale.bind("<ButtonPress-1>", self.begin_scale_mouse_change, add="+")
        scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")
        scale.bind("<ButtonRelease-1>", self.finish_scale_mouse_change, add="+")
        scale.bind("<KeyPress>", self.begin_scale_key_change, add="+")
        scale.bind("<KeyRelease>", self.defer_scale_key_refresh, add="+")
        value_label = tk.Label(parent, width=18, anchor="e")
        value_label.grid(row=row, column=2, pady=2)

        def update_value(*_args):
            # Do not refresh here: this trace fires continuously while the Scale is
            # dragged or while an arrow key auto-repeats.
            value_label.config(text=formatter(variable.get()))

        variable.trace_add("write", update_value)
        update_value()
'''
assert old_scale in text
text = text.replace(old_scale, new_scale, 1)

old_focus = '''    @staticmethod
    def focus_scale(event):
        """Keep a clicked slider focused so arrow keys continue to adjust it."""
        event.widget.focus_set()

'''
new_focus = '''    @staticmethod
    def focus_scale(event):
        """Keep a clicked slider focused so arrow keys continue to adjust it."""
        event.widget.focus_set()

    @staticmethod
    def begin_scale_mouse_change(event):
        """Remember the value before a mouse slider interaction begins."""
        event.widget._preview_mouse_start_value = event.widget.get()

    def finish_scale_mouse_change(self, event):
        """Refresh once after a mouse slider change has actually finished."""
        start_value = getattr(event.widget, "_preview_mouse_start_value", None)
        if start_value is not None and event.widget.get() != start_value:
            self.pending()

    def begin_scale_key_change(self, event):
        """Start/coalesce a keyboard slider interaction without refreshing."""
        if self.scale_key_release_job is not None:
            self.root.after_cancel(self.scale_key_release_job)
            self.scale_key_release_job = None
        if self.scale_key_widget is not event.widget:
            self.scale_key_widget = event.widget
            self.scale_key_start_value = event.widget.get()

    def defer_scale_key_refresh(self, event):
        """Refresh after the final KeyRelease, not during key auto-repeat."""
        if self.scale_key_widget is not event.widget:
            self.scale_key_widget = event.widget
            self.scale_key_start_value = event.widget.get()
        if self.scale_key_release_job is not None:
            self.root.after_cancel(self.scale_key_release_job)
        self.scale_key_release_job = self.root.after(45, self.finish_scale_key_change)

    def finish_scale_key_change(self):
        """Commit one preview refresh after keyboard slider input becomes idle."""
        self.scale_key_release_job = None
        widget = self.scale_key_widget
        start_value = self.scale_key_start_value
        self.scale_key_widget = None
        self.scale_key_start_value = None
        if widget is not None and start_value is not None and widget.get() != start_value:
            self.pending()

'''
assert old_focus in text
text = text.replace(old_focus, new_focus, 1)

old_pending = '''    def pending(self, *_args):
        self.clear_threshold_preview()
        self.status.set(
            "Settings changed. Threshold preview cleared; Refresh Preview or Apply Full Resolution to recompute."
        )

'''
new_pending = '''    def pending(self, *_args):
        """Request an immediate preview refresh after a discrete setting change."""
        self.refresh_preview()

'''
assert old_pending in text
text = text.replace(old_pending, new_pending, 1)

old_center = '''    def center_target_changed(self):
        self.clear_threshold_preview()
        self.update_center_preview_label()
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.status.set(
            f"Centering target set to {target}. Actual centering will be implemented with ellipse detection."
        )
'''
new_center = '''    def center_target_changed(self):
        self.update_center_preview_label()
        target = "light ellipse" if self.center_target.get() == "light" else "dark ellipse"
        self.refresh_preview()
        self.status.set(
            f"Centering target set to {target}. Actual centering will be implemented with ellipse detection."
        )
'''
assert old_center in text
text = text.replace(old_center, new_center, 1)

# Guard the user's explicit constraint: this GUI-only change must not introduce a
# grayscale/BW thresholding backend.
assert "cv2.COLOR_BGR2GRAY" not in text
assert "cv2.COLOR_BGRA2GRAY" not in text
assert "cv2.threshold(" not in text

SOURCE.write_text(text, encoding="utf-8")

# Replace the stale-preview-invalidating expectations with the new approved event
# contract. This remains a headless/static regression test.
TEST.write_text('''"""Regression checks for alpha-backed previews and refresh event semantics."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_preview_frames_have_no_placeholder_text():
    assert "def placeholder(" not in TEXT
    assert "create_text(" not in TEXT


def test_empty_preview_is_real_bgra_alpha_zero():
    assert "def transparent_bgra(" in TEXT
    assert "np.zeros((height, width, 4)" in TEXT
    assert "alpha = 0" in TEXT or "alpha=0" in TEXT


def test_transparency_is_not_faked_by_tk_background_matching():
    assert 'preview_background = frame.cget("background")' not in TEXT
    assert "Transparency is retained in the" in TEXT


def test_discrete_setting_changes_request_preview_refresh():
    pending = TEXT.split("def pending(self, *_args):", 1)[1].split("def center_target_changed", 1)[0]
    assert "self.refresh_preview()" in pending
    assert "self.clear_threshold_preview()" not in pending


def test_center_target_change_requests_preview_refresh():
    block = TEXT.split("def center_target_changed(self):", 1)[1].split("def update_center_preview_label", 1)[0]
    assert "self.refresh_preview()" in block
    assert "self.clear_threshold_preview()" not in block


def test_refresh_and_apply_actions_remain_present():
    assert "def refresh_preview(self):" in TEXT
    assert "def apply_full_resolution(self):" in TEXT
''', encoding="utf-8")

print("Applied deferred preview-refresh GUI event semantics")
