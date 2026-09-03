#!/usr/bin/env python3
"""Benchmark lossless in-memory compression of normalized uint16 BGRA masters.

Usage:
    python snippets/benchmark_master_image_compression.py image1 image2 ...

Reports source-file size, raw BGRA16 memory, zlib-level-1 payload size, one-time
compression time, and decompression time. Use genuine 16-bit TIFF inputs when
available; 8-bit JPEG-origin masters are unusually compressible after exact x257
expansion and therefore are not a reliable proxy for native 16-bit camera data.
"""

from __future__ import annotations

from pathlib import Path
import statistics
import sys
import time

import cv2

import circle_arc_detector as cad


def benchmark(path: Path, repeats: int = 3) -> tuple[object, ...]:
    unchanged = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if unchanged is None:
        raise RuntimeError(f"could not read {path}")

    master = cad.normalize_master_bgra16(unchanged)
    raw_bytes = master.nbytes

    compress_times = []
    payload = None
    for _ in range(repeats):
        start = time.perf_counter()
        payload = cad.compress_master_bgra16(master)
        compress_times.append(time.perf_counter() - start)
    assert payload is not None

    decompress_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        restored = cad.decompress_master_bgra16(payload, master.shape)
        decompress_times.append(time.perf_counter() - start)
        if not (restored == master).all():
            raise RuntimeError("master compression round trip changed pixels")

    return (
        path.name,
        path.stat().st_size,
        raw_bytes,
        len(payload),
        statistics.median(compress_times),
        statistics.median(decompress_times),
    )


def main() -> None:
    print("image,source_MB,raw_BGRA16_MB,zlib1_MB,memory_saved_pct,compress_s,decompress_s")
    for arg in sys.argv[1:]:
        name, source_bytes, raw_bytes, compressed_bytes, compress_s, decompress_s = benchmark(Path(arg))
        saved = 100.0 * (1.0 - compressed_bytes / raw_bytes)
        print(
            f"{name},{source_bytes / 1e6:.2f},{raw_bytes / 1e6:.2f},"
            f"{compressed_bytes / 1e6:.2f},{saved:.1f},{compress_s:.3f},{decompress_s:.3f}"
        )


if __name__ == "__main__":
    main()
