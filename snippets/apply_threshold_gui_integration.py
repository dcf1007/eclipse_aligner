"""Apply the tested threshold_finder module to the rebuilt GUI shell.

This integration script uses exact, asserted source replacements so the tested
threshold algorithm is imported rather than copied/reimplemented in the GUI.
It is intentionally preserved under snippets/ as the transport/integration code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
FOCUS_TEST = ROOT / "tests" / "test_gui_focus_autoselect.py"
INTEGRATION_TEST = ROOT / "tests" / "test_gui_threshold_integration.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one old block, found {count}")
    return text.replace(old, new, 1)


text = SOURCE.read_text(encoding="utf-8")

text = replace_once(
    text,
    '''This milestone intentionally contains the user interface only. Detection,
threshold candidate generation, ellipse fitting, horizon handling, and image
centering are not implemented yet. The interface is based on the latest GUI from
''',
    '''This milestone contains the rebuilt interface plus the grayscale-only automatic
threshold finder. Ellipse fitting, horizon handling, radius auto-selection, image
centering, and export are not implemented yet. The interface is based on the latest GUI from
''',
    "module docstring",
)

text = replace_once(
    text,
    '''import cv2
import numpy as np
''',
    '''import cv2
import numpy as np

from threshold_finder import auto_threshold_from_gray, to_gray
''',
    "threshold finder import",
)

text = replace_once(
    text,
    '''        self.current_path: str | None = None
        self.color_image = None
        self.threshold_preview = transparent_bgra()
''',
    '''        self.current_path: str | None = None
        self.color_image = None
        self.gray_image = None
        self.threshold_preview = transparent_bgra()

        # Threshold is stored per image so a manual override survives navigation.
        # Automatic selection runs on first load and only reruns when Auto select is
        # explicitly clicked for that image.
        self.image_thresholds: dict[str, int] = {}
        self.image_auto_results = {}
''',
    "per-image threshold state",
)

text = replace_once(
    text,
    '''        self.current_path = None
        self.color_image = None
        self.threshold_preview = transparent_bgra()
        self.load_image_at(0)
''',
    '''        self.current_path = None
        self.color_image = None
        self.gray_image = None
        self.threshold_preview = transparent_bgra()
        self.load_image_at(0)
''',
    "load-images reset",
)

old_load = '''        if image is None:
            self.color_image = None
            self.threshold_preview = transparent_bgra()
            self.threshold_photo = None
            self.color_photo = None
            self.redraw()
            self.status.set(f"Could not load image: {path}")
        else:
            # The preview pipeline is alpha-aware from the start. Source pixels are
            # opaque; future centered output may use alpha=0 outside image data.
            self.color_image = opaque_bgra(image)
            height, width = image.shape[:2]
            self.threshold_preview = transparent_bgra(width, height)
            self.redraw()
            self.status.set(
                "Image loaded. Detector functionality is intentionally not implemented in this milestone."
            )
'''
new_load = '''        if image is None:
            self.color_image = None
            self.gray_image = None
            self.threshold_preview = transparent_bgra()
            self.threshold_photo = None
            self.color_photo = None
            self.redraw()
            self.status.set(f"Could not load image: {path}")
        else:
            # Convert the original image to authoritative grayscale ONCE. The auto
            # threshold module derives its 1200px INTER_AREA working raster from
            # this grayscale image; no HSV/color threshold path exists.
            self.color_image = opaque_bgra(image)
            self.gray_image = to_gray(image)

            restored = path in self.image_thresholds
            if restored:
                selected_threshold = int(self.image_thresholds[path])
            else:
                result = auto_threshold_from_gray(self.gray_image)
                selected_threshold = int(result.threshold)
                self.image_thresholds[path] = selected_threshold
                self.image_auto_results[path] = result

            # Setting the Tk variable may clear an older preview through its trace;
            # image loading is one of the explicit operations that immediately
            # regenerates the black/white threshold raster afterward.
            self.threshold.set(selected_threshold)
            self.render_threshold_preview()
            self.redraw()

            if restored:
                self.status.set(
                    f"Image loaded. Restored stored threshold T={selected_threshold}."
                )
            elif result.resolved:
                self.status.set(
                    "Image loaded. Automatic grayscale threshold "
                    f"T={selected_threshold} (coarse T={result.coarse_threshold}, "
                    f"histogram start={result.histogram_left_edge})."
                )
            else:
                self.status.set(
                    "Image loaded. Automatic component tracking was unresolved; "
                    f"using rightmost-histogram left edge T={selected_threshold}."
                )
'''
text = replace_once(text, old_load, new_load, "image-load auto threshold")

text = replace_once(
    text,
    '''    def auto_select_threshold(self):
        self.status.set(
            "Auto select threshold: algorithm not implemented in the GUI milestone."
        )
''',
    '''    def auto_select_threshold(self):
        """Rerun image-only automatic T selection without regenerating the preview."""
        if self.gray_image is None:
            self.status.set("Auto select threshold: no readable image is loaded.")
            return

        result = auto_threshold_from_gray(self.gray_image)
        selected_threshold = int(result.threshold)
        if self.current_path is not None:
            self.image_thresholds[self.current_path] = selected_threshold
            self.image_auto_results[self.current_path] = result

        # The threshold variable trace deliberately clears any stale preview. Per
        # user requirement, Auto select itself DOES NOT regenerate that preview;
        # Refresh Preview / Apply Full Resolution remain explicit preview actions.
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
''',
    "auto-select threshold",
)

text = replace_once(
    text,
    '''    def clear_threshold_preview(self):
''',
    '''    def render_threshold_preview(self):
        """Render authoritative black/white mask for the currently selected T."""
        if self.gray_image is None:
            self.threshold_preview = transparent_bgra()
            self.threshold_photo = None
            return

        # Exact threshold semantics: dark = gray <= T, light = gray > T.
        light_mask = cv2.compare(
            self.gray_image,
            int(self.threshold.get()),
            cv2.CMP_GT,
        )
        preview = cv2.cvtColor(light_mask, cv2.COLOR_GRAY2BGRA)
        preview[:, :, 3] = 255
        self.threshold_preview = preview
        self.threshold_photo = None

    def clear_threshold_preview(self):
''',
    "threshold preview renderer",
)

text = replace_once(
    text,
    '''    def pending(self, *_args):
        self.clear_threshold_preview()
''',
    '''    def pending(self, *_args):
        if self.current_path is not None and self.gray_image is not None:
            self.image_thresholds[self.current_path] = int(self.threshold.get())
        self.clear_threshold_preview()
''',
    "manual threshold persistence",
)

text = replace_once(
    text,
    '''    def refresh_preview(self):
        self.status.set("Refresh Preview: detector backend not implemented in the GUI milestone.")

    def apply_full_resolution(self):
        self.status.set("Apply Full Resolution: detector backend not implemented in the GUI milestone.")
''',
    '''    def refresh_preview(self):
        if self.gray_image is None:
            self.status.set("Refresh Preview: no readable image is loaded.")
            return
        self.render_threshold_preview()
        self.redraw()
        self.status.set(
            f"Threshold preview regenerated at T={int(self.threshold.get())}."
        )

    def apply_full_resolution(self):
        if self.gray_image is None:
            self.status.set("Apply Full Resolution: no readable image is loaded.")
            return
        self.render_threshold_preview()
        self.redraw()
        self.status.set(
            "Full-resolution threshold preview applied at "
            f"T={int(self.threshold.get())}. Ellipse detector backend not implemented yet."
        )
''',
    "explicit preview actions",
)

SOURCE.write_text(text, encoding="utf-8")

FOCUS_TEST.write_text(
'''"""Static regression checks for slider focus and Auto select GUI controls."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_sliders_explicitly_keep_keyboard_focus_after_click():
    assert 'takefocus=True' in TEXT
    assert 'scale.bind("<ButtonPress-1>", self.focus_scale, add="+")' in TEXT
    assert 'scale.bind("<ButtonRelease-1>", self.focus_scale, add="+")' in TEXT
    assert 'event.widget.focus_set()' in TEXT


def test_threshold_has_auto_select_button_on_its_row():
    assert 'self.threshold_auto_button = tk.Button(' in TEXT
    assert 'command=self.auto_select_threshold' in TEXT
    assert 'row=0, column=3' in TEXT


def test_radius_auto_select_button_spans_min_and_max_rows():
    assert 'self.radius_auto_button = tk.Button(' in TEXT
    assert 'command=self.auto_select_radius' in TEXT
    assert 'row=1, column=3, rowspan=2' in TEXT


def test_threshold_auto_select_is_implemented_but_does_not_render_preview():
    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius", 1
    )[0]
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.threshold.set(selected_threshold)" in block
    assert "self.render_threshold_preview()" not in block
    assert "Preview not regenerated." in block


def test_radius_auto_select_remains_placeholder_for_later_branch_work():
    assert 'Auto select radius range: algorithm not implemented' in TEXT
    assert 'def detect(' not in TEXT
''',
encoding="utf-8",
)

INTEGRATION_TEST.write_text(
'''"""Static checks for automatic-threshold integration and preview timing."""

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_gui_source_parses():
    ast.parse(TEXT)


def test_gui_imports_tested_threshold_module_instead_of_copying_algorithm():
    assert "from threshold_finder import auto_threshold_from_gray, to_gray" in TEXT
    assert "def rightmost_histogram_peak(" not in TEXT


def test_first_image_load_runs_auto_and_generates_preview():
    block = TEXT.split("def load_image_at(self, index: int):", 1)[1].split(
        "def previous_image", 1
    )[0]
    assert "self.gray_image = to_gray(image)" in block
    assert "auto_threshold_from_gray(self.gray_image)" in block
    assert "self.render_threshold_preview()" in block


def test_per_image_manual_threshold_is_restored_on_navigation():
    assert "self.image_thresholds: dict[str, int] = {}" in TEXT
    pending = TEXT.split("def pending(self, *_args):", 1)[1].split(
        "def center_target_changed", 1
    )[0]
    assert "self.image_thresholds[self.current_path] = int(self.threshold.get())" in pending


def test_auto_select_does_not_regenerate_preview():
    block = TEXT.split("def auto_select_threshold(self):", 1)[1].split(
        "def auto_select_radius", 1
    )[0]
    assert "self.render_threshold_preview()" not in block


def test_only_image_load_and_explicit_preview_actions_render_threshold_output():
    refresh = TEXT.split("def refresh_preview(self):", 1)[1].split(
        "def apply_full_resolution", 1
    )[0]
    apply = TEXT.split("def apply_full_resolution(self):", 1)[1].split(
        "def schedule_redraw", 1
    )[0]
    assert "self.render_threshold_preview()" in refresh
    assert "self.render_threshold_preview()" in apply


def test_preview_uses_exact_threshold_semantics():
    renderer = TEXT.split("def render_threshold_preview(self):", 1)[1].split(
        "def clear_threshold_preview", 1
    )[0]
    assert "cv2.CMP_GT" in renderer
    assert "dark = gray <= T, light = gray > T" in renderer
''',
encoding="utf-8",
)

print("Applied threshold GUI integration and updated static regression tests.")
