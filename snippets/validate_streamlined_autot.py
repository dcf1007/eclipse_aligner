#!/usr/bin/env python3
"""Report current Stage-A and Stage-B threshold checkpoints for supplied images."""
from __future__ import annotations
from pathlib import Path
import math
import sys
import cv2
import circle_arc_detector as cad

def stages(path: Path):
    # Load unchanged, normalize once to the production uint16 BGRA master,
    # then derive the same fixed-range uint8 grayscale used by the GUI.
    unchanged = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if unchanged is None:
        raise RuntimeError(f"could not read {path}")
    master = cad.normalize_master_bgra16(unchanged)
    full_res_gray = cad.master_bgra16_to_gray8(master)
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
        work_res_gray, start_T, work_kernel
    )

    # Transfer categorical work-component membership with exact nearest mapping.
    full_res_search_mask = cad.resize_img(
        work_res_component,
        (full_res_width, full_res_height),
        mask=True,
    )

    # Map the work support footprint to the realized source scale.
    mapped_kernel_size = max(work_kernel.shape) * max(full_res_gray.shape) / max(work_res_gray.shape)
    kernel_size = cad.nearest_positive_odd(mapped_kernel_size)
    full_kernel = cad.generate_kernel((kernel_size, kernel_size), round_kernel=False)

    # Select the fixed source seed and construct the fixed 10% Stage-A guard.
    full_res_seed = cad.brightest_supported_component_point(
        full_res_gray, full_res_search_mask, full_kernel
    )
    if full_res_seed is None:
        raise RuntimeError('no full-resolution supported seed')
    image_scale = math.sqrt(full_res_width * full_res_height)
    guard = cad.dilate_component_mask(
        full_res_search_mask,
        cad.AUTO_T_GUARD_DILATION_FRACTION * image_scale,
    )

    # Stage A returns only its minimum defensible D7-cleaned separation T.
    full_res_T = cad.find_lowest_full_res_threshold(
        full_res_gray, work_res_T, full_res_seed, guard
    )

    # Stage B receives only T, seed, and guard and reconstructs its own samples.
    selection = cad.optimize_separated_threshold(
        full_res_gray, full_res_T, full_res_seed, guard
    )
    return start_T, work_res_T, kernel_size, full_res_seed, full_res_T, selection.threshold

def main():
    print('image,start_T,work_res_T,kernel,seed_x,seed_y,full_res_base_T,selected_T')
    for arg in sys.argv[1:]:
        path = Path(arg)
        start_T, work_T, kernel_size, seed, base_T, selected_T = stages(path)
        print(
            f'{path.name},{start_T},{work_T},{kernel_size},'
            f'{seed[0]},{seed[1]},{base_T},{selected_T}'
        )

if __name__ == '__main__':
    main()
