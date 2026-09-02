#!/usr/bin/env python3
"""Report the streamlined Auto-T stages for supplied image files.

Usage:
    python snippets/validate_streamlined_autot.py image1.jpg image2.jpg ...

The report intentionally exposes the work-resolution T and the base full-resolution
10%-guard separation T before Stage B so difficult-case regressions can be compared
without adding transient fields to AutoThresholdResult.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import cv2
import numpy as np

import circle_arc_detector as cad


def stages(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    full_res_gray = cad.to_gray(image)
    full_res_height, full_res_width = full_res_gray.shape
    if max(full_res_height, full_res_width) > cad.WORK_MAX_DIM:
        scale = cad.WORK_MAX_DIM / float(max(full_res_height, full_res_width))
        work_res_size = (
            max(1, round(full_res_width * scale)),
            max(1, round(full_res_height * scale)),
        )
    else:
        work_res_size = (full_res_width, full_res_height)
    work_res_gray = cad.resize_img(full_res_gray, work_res_size)
    start_T = cad.find_histogram_start_threshold(work_res_gray)
    work_kernel = cad.generate_kernel((cad.TRACKING_SEED_KERNEL_SIZE, cad.TRACKING_SEED_KERNEL_SIZE))
    work_res_T, work_res_component = cad.find_work_res_solar_component(
        work_res_gray, start_T, work_kernel
    )
    full_res_search_mask = cad.resize_img(
        work_res_component,
        (full_res_width, full_res_height),
    )

    mapped = (
        cad.TRACKING_SEED_KERNEL_SIZE
        * max(full_res_gray.shape)
        / float(max(work_res_gray.shape))
    )
    low = max(1, int(math.floor(mapped)))
    if low % 2 == 0:
        low -= 1
    high = low + 2
    kernel_size = low if abs(mapped - low) <= abs(high - mapped) else high
    full_kernel = cad.generate_kernel((kernel_size, kernel_size))
    full_res_seed = cad.brightest_supported_component_point(
        full_res_gray, full_res_search_mask, full_kernel
    )
    if full_res_seed is None:
        raise RuntimeError("no full-resolution supported seed")

    image_scale = math.sqrt(float(full_res_width) * float(full_res_height))
    guard = cad.dilate_component_mask(
        full_res_search_mask,
        cad.AUTO_T_GUARD_DILATION_FRACTION * image_scale,
    )
    full_res_T, full_res_component = cad.find_lowest_full_res_threshold(
        full_res_gray, work_res_T, full_res_seed, guard
    )
    selection = cad.optimize_separated_threshold(
        full_res_gray, full_res_T, full_res_seed, full_res_component
    )
    return start_T, work_res_T, kernel_size, full_res_seed, full_res_T, selection.threshold


def main():
    print("image,start_T,work_res_T,kernel,seed_x,seed_y,full_res_base_T,selected_T")
    for arg in sys.argv[1:]:
        path = Path(arg)
        start_T, work_T, kernel_size, seed, base_T, selected_T = stages(path)
        print(
            f"{path.name},{start_T},{work_T},{kernel_size},"
            f"{seed[0]},{seed[1]},{base_T},{selected_T}"
        )


if __name__ == "__main__":
    main()
