"""Regression tests for the validated 3-bin histogram peak signal."""

from pathlib import Path
import ast

import numpy as np

import circle_arc_detector as cad

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"
TEXT = SOURCE.read_text(encoding="utf-8")


def test_source_parses():
    ast.parse(TEXT)


def test_production_uses_tested_3bin_kernel_and_no_gaussian_sigma():
    assert "PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)" in TEXT
    assert "HISTOGRAM_SIGMA" not in TEXT
    assert "def _gaussian_kernel_1d(" not in TEXT


def test_local_peak_signal_only_uses_immediate_neighbor_bins():
    hist = np.zeros(256, dtype=np.float64)
    hist[100] = 4.0
    signal = cad.local_peak_signal(hist)

    expected = np.zeros(256, dtype=np.float64)
    expected[99] = 1.0
    expected[100] = 2.0
    expected[101] = 1.0
    np.testing.assert_array_equal(signal, expected)


def test_rightmost_peak_consumes_local_signal_not_raw_or_gaussian_signal():
    block = TEXT.split("def rightmost_histogram_peak", 1)[1].split(
        "def deepest_component_point", 1
    )[0]
    assert "histogram_with_peak_signal(gray)" in block
    assert "_local_peaks(signal)" in block
    assert "_preceding_valley(signal, peak)" in block
