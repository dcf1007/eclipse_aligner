#!/usr/bin/env python3
"""Apply the final agreed Stage-A freeze cleanup to production source."""
from __future__ import annotations

from pathlib import Path
import sys


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label}, found {count}")
    return text.replace(old, new, 1)


def apply_freeze_cleanup(source: str) -> str:
    """Return production source with the final Stage-A invariants encoded."""
    text = source

    # The generic to_gray helper is obsolete now that unchanged native-depth loads
    # normalize to BGRA16 and derive authoritative gray8 through the explicit input path.
    text = replace_exact(
        text,
        '''\ndef to_gray(image: np.ndarray) -> np.ndarray:\n    """Convert once to authoritative 8-bit grayscale."""\n    if image.ndim == 2:\n        return image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)\n    if image.ndim == 3 and image.shape[2] == 3:\n        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n    if image.ndim == 3 and image.shape[2] == 4:\n        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)\n    raise ValueError(f"Unsupported image shape: {image.shape}")\n\n''',
        "\n",
        "dead to_gray helper",
    )

    # Document the cleanup that defines full-resolution Stage-A separation.
    text = replace_exact(
        text,
        '''Auto-T then starts from the work-resolution T and searches only in the\nmonotonic direction needed to find the lowest full-resolution threshold whose seeded\ncomponent stays inside that fixed guard.''',
        '''Auto-T then starts from the work-resolution T and searches only in the\nmonotonic direction needed to find the lowest full-resolution threshold whose\nD7-cleaned seeded component stays inside that fixed guard.''',
        "module Stage-A separation description",
    )

    # Report the actual work support geometry rather than a stale hard-coded 5x5 label.
    text = replace_exact(
        text,
        '''    if seed is None or work_res_T is None or work_res_component is None:\n        raise ThresholdResolutionError(\n            "No 5x5-supported enclosed bright component exists through T=0"\n        )\n''',
        '''    if seed is None or work_res_T is None or work_res_component is None:\n        kernel_height, kernel_width = work_res_seed_kernel.shape\n        raise ThresholdResolutionError(\n            f"No {kernel_width}x{kernel_height}-supported enclosed bright component "\n            "exists through T=0"\n        )\n''',
        "work-resolution support failure",
    )

    # Stage A is defined after D7 cleanup. Do not reject a raw threshold mask before
    # the agreed OPEN/CLOSE has had a chance to fill a small local dark defect.
    text = replace_exact(
        text,
        '''    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)\n    if binary[seed_y, seed_x] == 0:\n        raise ThresholdResolutionError(\n            f"Full-resolution tracking seed is not light at start T={start_T}"\n        )\n    binary = cv2.morphologyEx(\n''',
        '''    binary = cv2.compare(full_res_gray, start_T, cv2.CMP_GT)\n    binary = cv2.morphologyEx(\n''',
        "pre-D7 source seed rejection",
    )

    # Enforce the authoritative Stage-A raster contract at the public boundary.
    text = replace_exact(
        text,
        '''def find_auto_threshold(\n    full_res_gray: np.ndarray,\n    image_state: dict[str, object],\n) -> int:\n    """Determine Auto T and cache only search state; ``resolve_threshold`` owns SolarData."""\n    if full_res_gray.ndim != 2:\n        raise ValueError("grayscale image must be two-dimensional")\n''',
        '''def find_auto_threshold(\n    full_res_gray: np.ndarray,\n    image_state: dict[str, object],\n) -> int:\n    """Determine Auto T and cache only search state; ``resolve_threshold`` owns SolarData."""\n    if full_res_gray.ndim != 2:\n        raise ValueError("grayscale image must be two-dimensional")\n    if full_res_gray.dtype != np.uint8:\n        raise ValueError("automatic thresholding requires authoritative uint8 grayscale")\n''',
        "find_auto_threshold input contract",
    )

    # Map the actual parent-owned work support footprint instead of repeating the
    # current policy value. Changing the work kernel now automatically changes its
    # equivalent source support size.
    text = replace_exact(
        text,
        '''        # Preserve the 5-pixel work support footprint at the realized source scale.\n        mapped_kernel_size = 5 * max(full_res_gray.shape) / max(work_res_gray.shape)\n        full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)\n''',
        '''        # Preserve the actual work seed-support footprint at the realized source scale.\n        work_res_support_size = max(work_res_seed_kernel.shape)\n        mapped_kernel_size = (\n            work_res_support_size\n            * max(full_res_gray.shape)\n            / max(work_res_gray.shape)\n        )\n        full_res_kernel_size = nearest_positive_odd(mapped_kernel_size)\n''',
        "source support mapping",
    )

    # Stage B's boundary is T + fixed seed + fixed guard; no Stage-A component crosses it.
    text = replace_exact(
        text,
        '''# Stage B receives Stage A's proven full-resolution T, component, and seed. It may\n# optimize upward from that boundary, but it must never redefine Stage-A identity\n''',
        '''# Stage B receives Stage A's proven full-resolution T, fixed seed, and fixed guard. It may\n# optimize upward from that boundary, but it must never redefine Stage-A identity\n''',
        "Stage-B boundary comment",
    )

    if "def to_gray(" in text:
        raise RuntimeError("dead to_gray helper remains")
    if "def opaque_bgra(" not in text:
        raise RuntimeError("GUI opaque BGRA helper was removed unexpectedly")
    if "mapped_kernel_size = 5 *" in text[text.index("# Auto-T Stage A"):text.index("# Auto-T Stage B")]:
        raise RuntimeError("Stage-A source support still hard-codes work kernel size")

    return text


def update_rebuild_chain(source: str) -> str:
    """Ensure the historical rebuild chain applies this final freeze cleanup last."""
    if "apply_stage_a_freeze_cleanup.py" in source:
        return source

    marker = '''    target.write_text(\n        prefreeze["apply_prefreeze_updates"](source),\n        encoding="utf-8",\n    )\n'''
    addition = marker + '''\n    # Apply the final Stage-A freeze cleanup after the pre-freeze input contract.\n    freezer_path = Path(__file__).with_name("apply_stage_a_freeze_cleanup.py")\n    freezer = runpy.run_path(str(freezer_path))\n    source = target.read_text(encoding="utf-8")\n    target.write_text(\n        freezer["apply_freeze_cleanup"](source),\n        encoding="utf-8",\n    )\n'''
    return replace_exact(source, marker, addition, "rebuild-chain prefreeze tail")


def update_validator(source: str) -> str:
    """Keep the retained real-image validator derived from actual work support."""
    return replace_exact(
        source,
        "    mapped_kernel_size = 5 * max(full_res_gray.shape) / max(work_res_gray.shape)\n",
        "    mapped_kernel_size = max(work_kernel.shape) * max(full_res_gray.shape) / max(work_res_gray.shape)\n",
        "validator source-support mapping",
    )


def main() -> None:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "circle_arc_detector.py")
    target.write_text(apply_freeze_cleanup(target.read_text(encoding="utf-8")), encoding="utf-8")

    if len(sys.argv) > 2:
        chain = Path(sys.argv[2])
        chain.write_text(update_rebuild_chain(chain.read_text(encoding="utf-8")), encoding="utf-8")

    if len(sys.argv) > 3:
        validator = Path(sys.argv[3])
        validator.write_text(update_validator(validator.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
