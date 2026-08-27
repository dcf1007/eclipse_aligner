"""Remove the obsolete threshold-candidate palette from threshold-finder GUI.

The threshold branch already has the automatic finder integrated, so this asserted
patch removes only the unused palette block and closes the resulting GUI row gap.
"""
from pathlib import Path

SOURCE = Path("circle_arc_detector.py")
text = SOURCE.read_text(encoding="utf-8")

old_block = '''        tk.Label(frame, text="Useful threshold candidates:").grid(\n            row=7, column=0, sticky="nw"\n        )\n        self.palette_frame = tk.Frame(frame)\n        self.palette_frame.grid(row=7, column=1, columnspan=3, sticky="w", pady=(0, 8))\n        tk.Label(\n            self.palette_frame,\n            text="Not implemented yet",\n            fg="#666666",\n        ).grid(row=0, column=0, sticky="w")\n\n'''
assert old_block in text
text = text.replace(old_block, "", 1)

old_buttons = 'button_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(2, 0))'
new_buttons = 'button_frame.grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))'
assert old_buttons in text
text = text.replace(old_buttons, new_buttons, 1)

old_status = ').grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))'
new_status = ').grid(row=8, column=0, columnspan=4, sticky="ew", pady=(8, 0))'
assert old_status in text
text = text.replace(old_status, new_status, 1)

SOURCE.write_text(text, encoding="utf-8")
print("Removed obsolete Useful threshold candidates GUI block from threshold branch")
