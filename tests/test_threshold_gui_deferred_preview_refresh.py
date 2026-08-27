"""Threshold-branch regression checks for deferred slider refresh semantics."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def block(start_marker: str, end_marker: str) -> str:
    return TEXT.split(start_marker, 1)[1].split(end_marker, 1)[0]


def test_source_parses():
    ast.parse(TEXT)


def test_slider_trace_updates_label_only_during_motion():
    update = block("        def update_value(*_args):", '        variable.trace_add("write", update_value)')
    assert "value_label.config" in update
    assert "self.pending()" not in update
    assert "self.refresh_preview()" not in update
    assert "self.clear_threshold_preview()" not in update


def test_mouse_slider_refresh_is_deferred_until_release():
    assert 'scale.bind("<ButtonPress-1>", self.begin_scale_mouse_change, add="+")' in TEXT
    assert 'scale.bind("<ButtonRelease-1>", self.finish_scale_mouse_change, add="+")' in TEXT
    finish = block("    def finish_scale_mouse_change(self, event):", "    def begin_scale_key_change")
    assert "event.widget.get() != start_value" in finish
    assert "self.pending()" in finish


def test_keyboard_slider_refresh_is_deferred_and_auto_repeat_is_coalesced():
    assert 'scale.bind("<KeyPress>", self.begin_scale_key_change, add="+")' in TEXT
    assert 'scale.bind("<KeyRelease>", self.defer_scale_key_refresh, add="+")' in TEXT
    begin = block("    def begin_scale_key_change(self, event):", "    def defer_scale_key_refresh")
    assert "after_cancel" in begin
    deferred = block("    def defer_scale_key_refresh(self, event):", "    def finish_scale_key_change")
    assert "self.root.after(45, self.finish_scale_key_change)" in deferred
    finish = block("    def finish_scale_key_change(self):", "    def release_scale_focus_if_outside")
    assert "widget.get() != start_value" in finish
    assert "self.pending()" in finish


def test_discrete_settings_refresh_immediately_through_pending_hook():
    assert TEXT.count("command=self.pending") >= 3
    pending = block("    def pending(self, *_args):", "    def center_target_changed")
    assert "self.image_thresholds[self.current_path] = int(self.threshold.get())" in pending
    assert "self.refresh_preview()" in pending
    assert "self.clear_threshold_preview()" not in pending


def test_radio_setting_change_refreshes_immediately():
    center = block("    def center_target_changed(self):", "    def update_center_preview_label")
    assert "self.refresh_preview()" in center
    assert "self.clear_threshold_preview()" not in center


def test_auto_select_keeps_explicit_no_preview_regeneration_rule():
    auto = block("    def auto_select_threshold(self):", "    def auto_select_radius")
    assert "auto_threshold_from_gray(self.gray_image)" in auto
    assert "self.threshold.set(selected_threshold)" in auto
    assert "self.render_threshold_preview()" not in auto
    assert "self.refresh_preview()" not in auto
    assert "Preview not regenerated." in auto


def test_threshold_backend_invariants_are_preserved():
    assert "def auto_threshold_from_gray(" in TEXT
    assert "PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)" in TEXT
    assert "ROI_DILATION_FRACTION = 0.065" in TEXT
    assert "GUARD_DILATION_FRACTION = 0.195" in TEXT
    renderer = block("    def render_threshold_preview(self):", "    def clear_threshold_preview")
    assert "cv2.CMP_GT" in renderer
