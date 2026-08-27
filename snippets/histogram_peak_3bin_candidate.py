"""Candidate replacement for broad Gaussian histogram smoothing.

Only immediate neighboring grayscale bins contribute to peak detection. The raw
histogram remains authoritative for pixel counts; this signal is used only to find
the rightmost local peak and its preceding valley.
"""
from __future__ import annotations

import numpy as np

PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)


def local_peak_signal(histogram: np.ndarray) -> np.ndarray:
    """Return a 3-bin [1,2,1]/4 signal for local peak/valley detection."""
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.shape != (256,):
        raise ValueError(f"Expected 256-bin histogram, got shape {hist.shape}")
    return np.convolve(hist, PEAK_KERNEL, mode="same")
