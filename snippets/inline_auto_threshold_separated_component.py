from pathlib import Path

source_path = Path(__file__).resolve().parents[1] / "circle_arc_detector.py"
source = source_path.read_text()

old = '''        separated_component = solar_component_from_seed_at_threshold(
            gray,
            final_threshold,
            full_seed,
        )
'''
new = '''        # The Auto-T seed is already authoritative here. Flood its exact
        # full-resolution component directly for topology-knee optimization rather
        # than routing through the post-T SolarData establishment stage.
        separated_binary = cv2.compare(gray, int(final_threshold), cv2.CMP_GT)
        seed_x, seed_y = map(int, full_seed)
        if separated_binary[seed_y, seed_x] == 0:
            raise ThresholdResolutionError(
                f"Auto-T solar seed is not light at separated T={final_threshold}"
            )
        cv2.floodFill(separated_binary, None, (seed_x, seed_y), 128, flags=8)
        separated_component = separated_binary == 128
        if _touches_image_border(separated_component):
            raise ThresholdResolutionError(
                f"Auto-T solar component touches the image border at T={final_threshold}"
            )
'''

count = source.count(old)
if count != 1:
    raise RuntimeError(f"expected one Auto-T separated-component call, found {count}")

source_path.write_text(source.replace(old, new, 1))
print("Inlined Auto-T separated-component flood")
