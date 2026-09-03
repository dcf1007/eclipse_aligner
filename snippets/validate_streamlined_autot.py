#!/usr/bin/env python3
"""Report current Stage-A and Stage-B Auto-T checkpoints for supplied images.

Usage:
    python snippets/validate_streamlined_autot.py image1.jpg image2.jpg ...

The report exposes the histogram start T, mature work-resolution T, mapped
source support size/seed, the Stage-A full-resolution separation T, and the
currently selected Stage-B T without adding transient fields to
AutoThresholdResult.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import cv2

import circle_arc_detector as cad


def stages(path: Path):
    # Match the production GUI's current canonical 8-bit BGR input path.
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    full_res_gray = cad.to_gray(image)
    full_res_height, full_res_width = full_res_gray.shape
    full_res_max_dim = max(full_res_height, full_res_width)

    # Build the explicit work-resolution raster size used by Stage A.
    if full_res_max_dim > cad.WORK_RES_MAX_DIM:
        scale = cad.WORK_RES_MAX_DIM / full_res_max_dim
        work_res_size = (
            round(full_res_width * scale),
            round(full_res_height * scale),
        )
    else:
        work_res_size = (full_res_width, full_res_height)
    work_res_gray = cad.resize_img(full_res_gray, work_res_size)

    # Establish and track the current 5x5-supported work-resolution identity.
    start_T = cad.find_histogram_start_threshold(work_res_gray)
    work_kernel = cad.generate_kernel((5, 5), round_kernel=False)
    work_res_T, work_res_component = cad.find_work_res_solar_component(
        work_res_gray,
        start_T,
        work_kernel,
    )

    # Transfer only mature component geometry onto the exact source grid.
    full_res_search_mask = cad.resize_img(
        work_res_component,
        (full_res_width, full_res_height),
    )

    # Map the work support footprint to the realized source scale.
    mapped_kernel_size = 5 * max(full_res_gray.shape) / max(work_res_gray.shape)
    kernel_size = cad.nearest_positive_odd(mapped_kernel_size)
    full_kernel = cad.generate_kernel(
        (kernel_size, kernel_size),
        round_kernel=False,
    )

    # Select the fixed supported source seed inside the transferred mature mask.
    full_res_seed = cad.brightest_supported_component_point(
        full_res_gray,
        full_res_search_mask,
        full_kernel,
    )
    if full_res_seed is None:
        raise RuntimeError("no full-resolution supported seed")

    # Build the fixed 10% L2 guard and find Stage A's source separation boundary.
    image_scale = math.sqrt(full_res_width * full_res_height)
    guard = cad.dilate_component_mask(
        full_res_search_mask,
        cad.AUTO_T_GUARD_DILATION_FRACTION * image_scale,
    )
    full_res_T, full_res_component = cad.find_lowest_full_res_threshold(
        full_res_gray,
        work_res_T,
        full_res_seed,
        guard,
    )

    # Apply the current Stage-B selector only after Stage A is fully resolved.
    selection = cad.optimize_separated_threshold(
        full_res_gray,
        full_res_T,
        full_res_seed,
        full_res_component,
    )
    return (
        start_T,
        work_res_T,
        kernel_size,
        full_res_seed,
        full_res_T,
        selection.threshold,
    )


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
