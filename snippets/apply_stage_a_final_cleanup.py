#!/usr/bin/env python3
"""Apply the final Stage-A cleanup and source-layout boundary.

This transformation starts from the already streamlined/readability-cleaned
threshold-finder source. It preserves Stage-A behavior while removing the last
width-only resize assumption, making support-border semantics explicit, separating
Stage-A failures from Stage-B failures, and arranging the module by processing stage.
"""

from __future__ import annotations

from pathlib import Path
import sys


SETTINGS_MARKER = """# ---------------------------------------------------------------------------\n# Per-image processing settings\n# ---------------------------------------------------------------------------\n"""
OLD_AUTO_MARKER = """# ---------------------------------------------------------------------------\n# Grayscale automatic threshold finder\n# ---------------------------------------------------------------------------\n"""
CLEANUP_MARKER = """# ---------------------------------------------------------------------------\n# Cleanup morphology candidates\n# ---------------------------------------------------------------------------\n"""
FINAL_MARKER = """# ---------------------------------------------------------------------------\n# Final-T full-resolution solar resolution and persistence\n# ---------------------------------------------------------------------------\n"""


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence, found {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


def finalize_stage_a_source(source: str) -> str:
    """Return the final Stage-A source without changing normal Stage-A decisions."""
    # Keep the module overview synchronized with the current Euclidean refinement
    # and fixed L2 guard geometry.
    source = _replace_once(
        source,
        "applies 7x7 elliptical OPEN/CLOSE,",
        "applies a 7x7 Euclidean OPEN/CLOSE,",
    )
    source = _replace_once(
        source,
        "full resolution only to delimit the full-resolution seed search and a rounded 10%\n"
        "guard.",
        "full resolution only to delimit the full-resolution seed search and the fixed 10% L2-distance\n"
        "guard.",
    )

    # A same-size request is already the exact desired raster; avoid an unnecessary
    # OpenCV pass. For non-binary images, any shrinking axis makes this a downscale.
    source = _replace_once(
        source,
        """    if width <= 0 or height <= 0:\n        raise ValueError(\"resize dimensions must be positive\")\n\n    is_binary = np.all((img == img.min()) | (img == img.max()))\n""",
        """    if width <= 0 or height <= 0:\n        raise ValueError(\"resize dimensions must be positive\")\n    if (width, height) == (original_width, original_height):\n        return img.copy()\n\n    is_binary = np.all((img == img.min()) | (img == img.max()))\n""",
    )
    source = _replace_once(
        source,
        "    elif width < original_width:\n",
        "    elif width < original_width or height < original_height:\n",
    )

    # Full support means the entire caller-supplied footprint must fit inside the
    # component; pixels beyond the raster are therefore explicitly background.
    source = _replace_once(
        source,
        "    supported = cv2.erode(source, support_kernel, iterations=1) != 0\n",
        """    supported = cv2.erode(\n        source,\n        support_kernel,\n        iterations=1,\n        borderType=cv2.BORDER_CONSTANT,\n        borderValue=0,\n    ) != 0\n""",
    )

    # The histogram is always at least 256 bins; no defensive length branch is needed.
    source = _replace_once(
        source,
        "    if len(signal) >= 2 and signal[-1] > signal[-2]:\n",
        "    if signal[-1] > signal[-2]:\n",
    )

    # Make 8-connectivity explicit and express the actual largest-area rule directly.
    old_component_function_start = source.index(
        "def largest_enclosed_bright_component(binary: np.ndarray) -> np.ndarray | None:\n"
    )
    old_component_function_end = source.index(
        "\n\ndef find_work_res_solar_component(",
        old_component_function_start,
    )
    new_component_function = '''def largest_enclosed_bright_component(binary: np.ndarray) -> np.ndarray | None:\n    """Return the largest 8-connected bright component enclosed by the raster."""\n    count, labels, stats, _ = cv2.connectedComponentsWithStats(\n        (binary != 0).astype(np.uint8),\n        connectivity=8,\n    )\n    height, width = binary.shape\n    best_label = None\n    best_area = None\n    for label in range(1, count):\n        x, y, component_width, component_height, area = stats[label]\n        if (\n            x == 0\n            or y == 0\n            or x + component_width >= width\n            or y + component_height >= height\n        ):\n            continue\n        if best_label is None or area > best_area:\n            best_area = area\n            best_label = label\n    if best_label is None:\n        return None\n    return labels == best_label\n'''
    source = (
        source[:old_component_function_start]
        + new_component_function
        + source[old_component_function_end:]
    )

    # Keep the configurable work-resolution support footprint in the Stage-A parent;
    # the child tracker should not own a hard-coded support size.
    source = _replace_once(
        source,
        "    # Generate the fixed square work-resolution support kernel once and reuse it through the search.\n",
        """    # Generate the current work-resolution support kernel in the parent so its\n    # footprint can be adjusted alongside WORK_RES_MAX_DIM without changing child tracking logic.\n""",
    )

    # Stage A owns only source-separation failures. Move the Stage-B call/result
    # construction outside that try/except so future Stage-B errors cannot be
    # misclassified as inability to establish a separated source component.
    stage_b_start = source.index(
        "        # Stage B may raise T to a topology knee but never lowers the proven separated threshold.\n"
    )
    except_start = source.index(
        "    except ThresholdResolutionError as exc:\n",
        stage_b_start,
    )
    stage_b_block = source[stage_b_start:except_start]
    stage_b_block = "".join(
        line[4:] if line.startswith("    ") else line
        for line in stage_b_block.splitlines(keepends=True)
    )
    stage_b_block = stage_b_block.replace(
        "    # Stage B may raise T to a topology knee but never lowers the proven separated threshold.\n",
        """    # Stage A is complete: these are the proven source-separation T, component, and seed.\n    # Stage B may raise T to an optimization knee but must never lower this Stage-A result.\n""",
        1,
    )
    source = source[:stage_b_start] + source[except_start:]
    final_store = (
        '    image_state["auto_threshold_result"] = result\n'
        '    return topology_selection.threshold\n'
    )
    final_store_index = source.index(
        final_store,
        source.index("def find_auto_threshold("),
    )
    source = (
        source[:final_store_index]
        + stage_b_block
        + "\n"
        + source[final_store_index:]
    )

    # Reassemble the module by ownership: generic utilities, settings, Stage A,
    # Stage B, final-T/SolarData, then GUI. This is intentionally source movement,
    # not a second implementation of any algorithm.
    prefix, after_settings_marker = source.split(SETTINGS_MARKER, 1)
    settings_content, after_auto_marker = after_settings_marker.split(OLD_AUTO_MARKER, 1)
    old_auto_content, after_cleanup_marker = after_auto_marker.split(CLEANUP_MARKER, 1)
    cleanup_content, after_final_marker = after_cleanup_marker.split(FINAL_MARKER, 1)
    final_content, gui_tail = after_final_marker.split("\n\nclass DetectorApp:", 1)

    stage_b_top_start = old_auto_content.index(
        "@dataclass(frozen=True)\nclass ThresholdTopology:"
    )
    stage_a_error_start = old_auto_content.index(
        "class ThresholdResolutionError(RuntimeError):"
    )
    generic_move_start = old_auto_content.index("def to_gray(")
    stage_a_functions_start = old_auto_content.index(
        "def find_histogram_start_threshold("
    )

    stage_b_top = old_auto_content[stage_b_top_start:stage_a_error_start]
    stage_a_error_result = old_auto_content[stage_a_error_start:generic_move_start]
    generic_moved = old_auto_content[generic_move_start:stage_a_functions_start]
    stage_a_functions = old_auto_content[stage_a_functions_start:]

    # Within Stage B, keep threshold-trajectory structures/functions together and
    # morphology-candidate measurement structures/functions together.
    cleanup_metrics_start = stage_b_top.index(
        "@dataclass(frozen=True)\nclass CleanupMetrics:"
    )
    cleanup_eval_start = stage_b_top.index(
        "@dataclass(frozen=True)\nclass CleanupCandidateEvaluation:"
    )
    topology_selection_start = stage_b_top.index(
        "@dataclass(frozen=True)\nclass ThresholdTopologySelection:"
    )
    descriptor_start = stage_b_top.index("def _component_descriptor(")

    threshold_topology = stage_b_top[:cleanup_metrics_start]
    cleanup_metrics = stage_b_top[cleanup_metrics_start:cleanup_eval_start]
    cleanup_evaluation = stage_b_top[cleanup_eval_start:topology_selection_start]
    topology_selection = stage_b_top[topology_selection_start:descriptor_start]
    topology_functions = stage_b_top[descriptor_start:]

    cleanup_function_start = cleanup_content.index("def open_close_component(")
    cleanup_preamble = cleanup_content[:cleanup_function_start]
    cleanup_functions = cleanup_content[cleanup_function_start:]

    # Label the generic utility section once, before the first raster helper.
    prefix = _replace_once(
        prefix,
        "\n\ndef transparent_bgra(",
        """\n\n# ---------------------------------------------------------------------------\n# Generic image and kernel utilities\n# ---------------------------------------------------------------------------\ndef transparent_bgra(""",
    )

    stage_a_section = f'''# ---------------------------------------------------------------------------\n# Auto-T Stage A: source separation\n# ---------------------------------------------------------------------------\n# Stage A establishes one authoritative full-resolution seed-connected component\n# and the lowest threshold at which that component remains separated inside one\n# fixed 10% L2 guard. Stage A does not perform topology/morphology optimization.\nWORK_RES_MAX_DIM = 1200\nPEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)\nAUTO_T_GUARD_DILATION_FRACTION = 0.10\n\n\n{stage_a_error_result.strip()}\n\n\n{stage_a_functions.strip()}\n'''

    stage_b_section = f'''# ---------------------------------------------------------------------------\n# Auto-T Stage B: threshold optimization\n# ---------------------------------------------------------------------------\n# Stage B receives Stage A's proven full-resolution T, component, and seed. It may\n# optimize upward from that boundary, but it must never redefine Stage-A identity\n# or lower the proven separation threshold. This block is the experimental redesign\n# boundary for the next phase of work.\nTOPOLOGY_OPTIMIZATION_STEPS = 5\n\n\n{threshold_topology.strip()}\n\n\n{topology_selection.strip()}\n\n\n{topology_functions.strip()}\n\n\n# Morphology candidate measurements retained for the Stage-B redesign.\n{cleanup_preamble.strip()}\n\n\n{cleanup_metrics.strip()}\n\n\n{cleanup_evaluation.strip()}\n\n\n{cleanup_functions.strip()}\n'''

    final_section = f'''# ---------------------------------------------------------------------------\n# Final-T full-resolution solar resolution and persistence\n# ---------------------------------------------------------------------------\n# These constants belong to downstream SolarData construction, not Auto-T Stage A.\nROI_DILATION_FRACTION = 0.065\nGUARD_DILATION_FRACTION = 0.195\n\n# Build the fixed final-T Euclidean cleanup kernel once and reuse it.\nSOLAR_COMPONENT_KERNEL = generate_kernel((7, 7), round_kernel=True)\n\n\n{final_content.strip()}\n'''

    rebuilt = (
        prefix.rstrip()
        + "\n\n\n"
        + generic_moved.strip()
        + "\n\n\n"
        + SETTINGS_MARKER
        + settings_content.strip()
        + "\n\n\n"
        + stage_a_section.strip()
        + "\n\n\n"
        + stage_b_section.strip()
        + "\n\n\n"
        + final_section.strip()
        + "\n\n\nclass DetectorApp:"
        + gui_tail
    )
    return rebuilt.rstrip() + "\n"


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")
    target.write_text(
        finalize_stage_a_source(target.read_text(encoding="utf-8")),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
