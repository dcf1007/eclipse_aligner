"""Route image-load threshold display through the same deferred preview pipeline."""

from pathlib import Path


path = Path("circle_arc_detector.py")
source = path.read_text(encoding="utf-8")

start_marker = "            self._update_center_preview_label()\n            selected_threshold = int(self.threshold.get())\n\n            # Stage 1: show the pure threshold result immediately.\n"
end_marker = "\n        self.update_navigation_state()\n"
start = source.find(start_marker)
if start < 0:
    raise RuntimeError("Could not find image-load threshold preview block")
end = source.find(end_marker, start)
if end < 0:
    raise RuntimeError("Could not find end of load_image_at preview block")

replacement = '''            self._update_center_preview_label()

            # Use the same explicit two-turn preview path as a completed control
            # change: raw gray > T is painted now, and finalized SolarData replaces
            # it only from the queued refinement callback. This keeps image loading
            # from being the one remaining path that hides the raw threshold frame.
            self.refresh_preview(changed_setting="image load")
'''

source = source[:start] + replacement + source[end:]
path.write_text(source, encoding="utf-8")
