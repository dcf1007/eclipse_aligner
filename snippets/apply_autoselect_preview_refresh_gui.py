"""Make threshold Auto select use the normal preview-refresh hook in GUI-only branch.

This remains GUI-only: it does not add threshold computation or grayscale/BW raster
code. It only makes the threshold Auto-select control follow the same completed-
control refresh contract as the other settings.
"""
from pathlib import Path

SOURCE = Path("circle_arc_detector.py")
text = SOURCE.read_text(encoding="utf-8")

old = '''    def auto_select_threshold(self):
        self.status.set(
            "Auto select threshold: algorithm not implemented in the GUI milestone."
        )
'''
new = '''    def auto_select_threshold(self):
        self.refresh_preview()
        self.status.set(
            "Auto select threshold: algorithm not implemented in the GUI milestone."
        )
'''
assert old in text
text = text.replace(old, new, 1)

# GUI-only invariant: this change must not introduce threshold/grayscale backend code.
assert "cv2.COLOR_BGR2GRAY" not in text
assert "cv2.COLOR_BGRA2GRAY" not in text
assert "cv2.threshold(" not in text

SOURCE.write_text(text, encoding="utf-8")
print("Applied GUI Auto-select preview-refresh hook")
