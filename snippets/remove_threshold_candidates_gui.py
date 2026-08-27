"""Apply the approved GUI removal of the obsolete threshold-candidate palette.

This script is intentionally narrow and asserted: it removes only the unused
"Useful threshold candidates" placeholder block and closes the resulting row gap.
"""
from pathlib import Path

SOURCE = Path("circle_arc_detector.py")
text = SOURCE.read_text(encoding="utf-8")

old_doc = "Detection,\nthreshold candidate generation, ellipse fitting, horizon handling, and image\ncentering are not implemented yet."
new_doc = "Detection, ellipse fitting, horizon handling, and image centering are not\nimplemented yet."
assert old_doc in text
text = text.replace(old_doc, new_doc, 1)

old_block = '''        tk.Label(frame, text="Useful threshold candidates:").grid(\n            row=7, column=0, sticky="nw"\n        )\n        self.palette_frame = tk.Frame(frame)\n        self.palette_frame.grid(row=7, column=1, columnspan=3, sticky="w", pady=(0, 8))\n        tk.Label(\n            self.palette_frame,\n            text="Not implemented yet",\n            fg="#666666",\n        ).grid(row=0, column=0, sticky="w")\n\n'''
assert old_block in text
text = text.replace(old_block, "", 1)

assert 'button_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(2, 0))' in text
text = text.replace(
    'button_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(2, 0))',
    'button_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))',
    1,
)
assert ').grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))' in text
text = text.replace(
    ').grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))',
    ').grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))',
    1,
)

SOURCE.write_text(text, encoding="utf-8")
print("Removed obsolete Useful threshold candidates GUI block")
