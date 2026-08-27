"""Apply the validated 3-bin histogram peak signal to production.

This script changes only histogram peak/valley detection. The raw 256-bin
histogram and every downstream solar-lineage/connectivity step remain unchanged.
The 3-bin [1,2,1]/4 helper is copied exactly from
``snippets/histogram_peak_3bin_candidate.py``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "circle_arc_detector.py"

text = SOURCE.read_text(encoding="utf-8")

old_constant = "HISTOGRAM_SIGMA = 3.0\n"
new_constant = "PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)\n"
if text.count(old_constant) != 1:
    raise RuntimeError(f"expected exactly one HISTOGRAM_SIGMA constant, found {text.count(old_constant)}")
text = text.replace(old_constant, new_constant, 1)

old_histogram_block = '''def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smoothed_histogram(gray: np.ndarray, sigma: float = HISTOGRAM_SIGMA):
    """Return exact 256-bin histogram plus a peak-finding-only smoothed copy."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    smooth = np.convolve(hist, _gaussian_kernel_1d(sigma), mode="same")
    return hist, smooth
'''

new_histogram_block = '''def local_peak_signal(histogram: np.ndarray) -> np.ndarray:
    """Return a 3-bin [1,2,1]/4 signal for local peak/valley detection."""
    hist = np.asarray(histogram, dtype=np.float64)
    if hist.shape != (256,):
        raise ValueError(f"Expected 256-bin histogram, got shape {hist.shape}")
    return np.convolve(hist, PEAK_KERNEL, mode="same")


def histogram_with_peak_signal(gray: np.ndarray):
    """Return exact histogram plus the local 3-bin peak/valley signal."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    return hist, local_peak_signal(hist)
'''

if text.count(old_histogram_block) != 1:
    raise RuntimeError("expected Gaussian histogram block was not found exactly once")
text = text.replace(old_histogram_block, new_histogram_block, 1)

old_rightmost = '''def rightmost_histogram_peak(gray: np.ndarray) -> tuple[int, int]:
    """Rightmost smoothed mode and the valley defining its left edge."""
    _hist, smooth = smoothed_histogram(gray)
    peak = max(_local_peaks(smooth))
    return int(peak), int(_preceding_valley(smooth, peak))
'''

new_rightmost = '''def rightmost_histogram_peak(gray: np.ndarray) -> tuple[int, int]:
    """Rightmost locally smoothed mode and the valley defining its left edge."""
    _hist, signal = histogram_with_peak_signal(gray)
    peak = max(_local_peaks(signal))
    return int(peak), int(_preceding_valley(signal, peak))
'''

if text.count(old_rightmost) != 1:
    raise RuntimeError("expected rightmost_histogram_peak block was not found exactly once")
text = text.replace(old_rightmost, new_rightmost, 1)

# Guard against accidentally retaining the broad Gaussian path.
assert "HISTOGRAM_SIGMA" not in text
assert "_gaussian_kernel_1d" not in text
assert "PEAK_KERNEL = np.array([0.25, 0.50, 0.25], dtype=np.float64)" in text
assert "def local_peak_signal(" in text

SOURCE.write_text(text, encoding="utf-8")
print("Applied tested 3-bin histogram peak signal to circle_arc_detector.py")
