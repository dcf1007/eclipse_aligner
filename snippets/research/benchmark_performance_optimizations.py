"""Microbenchmark exact-equivalent full-frame operations used by threshold finding.

Run from the repository root. The script deliberately compares the pre-optimization
expressions with the selected exact-equivalent implementations; it does not tune
algorithm parameters or alter production state.
"""
from __future__ import annotations

from statistics import median
from time import perf_counter

import cv2
import numpy as np

import circle_arc_detector as cad


HEIGHT = 4000
WIDTH = 6000
REPEATS = 5


def timed(function, repeats: int = REPEATS) -> float:
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append(perf_counter() - start)
    return median(samples)


def main() -> None:
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    center_y = HEIGHT // 2
    center_x = WIDTH // 2
    radius = min(HEIGHT, WIDTH) // 4
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2

    old_mask = lambda: np.where(mask, 255, 0).astype(np.uint8)
    new_mask = lambda: cad.bool_mask_to_uint8(mask)
    assert np.array_equal(old_mask(), new_mask())

    binary_u8 = new_mask()
    old_uint8_normalize = lambda: cad.bool_mask_to_uint8(binary_u8 != 0)
    new_uint8_normalize = lambda: cv2.compare(binary_u8, 0, cv2.CMP_NE)
    assert np.array_equal(old_uint8_normalize(), new_uint8_normalize())

    guard = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= (radius * 1.15) ** 2
    guard_u8 = cad.bool_mask_to_uint8(guard)
    binary = cad.bool_mask_to_uint8(mask)

    def old_clip() -> np.ndarray:
        result = binary.copy()
        result[~guard] = 0
        return result

    def new_clip() -> np.ndarray:
        result = binary.copy()
        cv2.bitwise_and(result, guard_u8, dst=result)
        return result

    assert np.array_equal(old_clip(), new_clip())

    boundary = cad.find_guard_boundary(guard)
    boundary_indices = np.flatnonzero(boundary)
    component = mask.copy()

    def old_boundary_contact() -> bool:
        return bool(np.any(component & boundary))

    def new_boundary_contact() -> bool:
        return bool(np.any(component.ravel()[boundary_indices]))

    assert old_boundary_contact() == new_boundary_contact()

    values = np.arange(HEIGHT * WIDTH, dtype=np.uint32).reshape(HEIGHT, WIDTH)
    uint16_image = (values % 65536).astype(np.uint16)

    def old_uint16_to_uint8() -> np.ndarray:
        return ((uint16_image.astype(np.uint32) + 128) // 257).astype(np.uint8)

    def new_uint16_to_uint8() -> np.ndarray:
        return cv2.convertScaleAbs(uint16_image, alpha=1.0 / 257.0)

    assert np.array_equal(old_uint16_to_uint8(), new_uint16_to_uint8())

    print(f"bool -> uint8 old: {timed(old_mask):.6f} s")
    print(f"bool -> uint8 new: {timed(new_mask):.6f} s")
    print(f"uint8 normalize old: {timed(old_uint8_normalize):.6f} s")
    print(f"uint8 normalize new: {timed(new_uint8_normalize):.6f} s")
    print(f"guard clip old:   {timed(old_clip):.6f} s")
    print(f"guard clip new:   {timed(new_clip):.6f} s")
    print(f"boundary old:     {timed(old_boundary_contact):.6f} s")
    print(f"boundary new:     {timed(new_boundary_contact):.6f} s")
    print(f"uint16 -> uint8 old: {timed(old_uint16_to_uint8):.6f} s")
    print(f"uint16 -> uint8 new: {timed(new_uint16_to_uint8):.6f} s")


if __name__ == "__main__":
    main()
