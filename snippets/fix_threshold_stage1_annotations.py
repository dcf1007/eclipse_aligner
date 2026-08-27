from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
text = SOURCE.read_text(encoding="utf-8")
old = "from __future__ import annotations\n\n"
assert old in text
SOURCE.write_text(text.replace(old, "", 1), encoding="utf-8")
print("Removed unnecessary deferred annotations")
