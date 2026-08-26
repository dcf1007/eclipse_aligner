from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
text = SOURCE.read_text()

assert 'text="Save centered images"' in text
assert 'command=self.save_centered_images' in text
assert 'def save_centered_images(self):' in text
assert 'self.save_centered_button.grid(row=0, column=1' in text
assert 'frame.columnconfigure(4, weight=1)' in text
assert 'self.previous_button.grid(row=0, column=2' in text
assert 'self.next_button.grid(row=0, column=3' in text
assert 'row=0, column=4, sticky="ew"' in text

print("PASS: Save centered images button is beside Load images and remains GUI-only")
