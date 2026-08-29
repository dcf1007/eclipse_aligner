"""Generic post-separation topology optimizer for automatic threshold selection.

The existing threshold finder establishes the lowest full-resolution threshold T
at which the persistent solar component is separated from the background. This
module does not replace that separation search. It examines only the nested seeded
components at T..T+5 and selects the earliest topology knee: the point where most
boundary cleanup has already been gained and later threshold increases primarily
continue eroding the component.

No eclipse phase, frame number, radius, ellipse, EXIF, neighboring image, or
horizon information participates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

TOPOLOGY_OPTIMIZATION_STEPS = 5


@dataclass(frozen=True)
class ThresholdTopology:
    threshold: int
    area: int
    contour_n: int
    perimeter: float
    roughness: float
    solidity: float


@dataclass(frozen=True)
class ThresholdTopologySelection:
    threshold: int
    base_threshold: int
    delta: int
    trajectory: tuple[ThresholdTopology, ...]
    net_quality: tuple[float, ...]
    knee_curve: tuple[float, ...]


def _component_descriptor(component: np.ndarray, threshold: int) -> ThresholdTopology:
    component = np.asarray(component, dtype=bool)
    area = int(np.count_nonzero(component))
    if area <= 0:
        raise ValueError("empty component")
    u8 = np.where(component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("component has no external contour")
    contour = max(contours, key=cv2.contourArea)
    contour_n = int(len(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0.0 else 0.0
    roughness = float(perimeter / max(2.0 * math.sqrt(math.pi * area), 1e-9))
    return ThresholdTopology(
        threshold=int(threshold),
        area=area,
        contour_n=contour_n,
        perimeter=perimeter,
        roughness=roughness,
        solidity=solidity,
    )


def topology_trajectory_from_separated_component(
    full_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> tuple[ThresholdTopology, ...]:
    """Measure exact seeded topology for T..T+max_delta in the T-component crop.

    Because higher thresholds can only remove light pixels, a seeded component at
    T+k cannot gain pixels that were outside the seeded component at T. Restricting
    the scan to the base-component crop is therefore topology-equivalent to repeated
    full-frame floods and substantially cheaper on full-resolution photographs.
    """
    base_component = np.asarray(base_component, dtype=bool)
    ys, xs = np.nonzero(base_component)
    if len(xs) == 0:
        raise ValueError("base component is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    base_crop = base_component[y0:y1, x0:x1]
    gray_crop = full_gray[y0:y1, x0:x1]
    sx = int(seed_point[0]) - x0
    sy = int(seed_point[1]) - y0
    if not (0 <= sx < gray_crop.shape[1] and 0 <= sy < gray_crop.shape[0]):
        raise ValueError("seed lies outside base component crop")

    trajectory: list[ThresholdTopology] = []
    for delta in range(max(0, int(max_delta)) + 1):
        threshold = min(255, int(base_threshold) + delta)
        light = base_crop & (gray_crop > threshold)
        if not bool(light[sy, sx]):
            break
        flood = np.where(light, 255, 0).astype(np.uint8)
        cv2.floodFill(flood, None, (sx, sy), 128, flags=8)
        component = flood == 128
        if not np.any(component):
            break
        trajectory.append(_component_descriptor(component, threshold))
        if threshold >= 255:
            break
    if not trajectory:
        raise ValueError("no valid topology samples")
    return tuple(trajectory)


def select_topology_knee(
    trajectory: tuple[ThresholdTopology, ...] | list[ThresholdTopology],
) -> ThresholdTopologySelection:
    """Select the first knee of cleanup benefit versus component erosion.

    All quantities are dimensionless relative changes from the separated component.

    Benefit terms (equal weight):
      * external contour-count reduction,
      * perimeter-normalized roughness reduction,
      * positive solidity gain normalized by the base solidity headroom to 1.

    Cost terms (equal weight):
      * solar-component area loss,
      * solidity loss normalized by the base solidity.

    The resulting net-quality trajectory is replaced by its best-so-far envelope;
    a higher T that is already worse cannot become the selected knee. The envelope
    is normalized to its observed maximum and compared with uniform threshold
    progress (0..1). Maximizing ``quality_progress - threshold_progress`` is the
    standard discrete elbow construction and deliberately chooses the *first* point
    where most available topology improvement has already been realized.
    """
    rows = tuple(trajectory)
    if not rows:
        raise ValueError("empty topology trajectory")
    if len(rows) == 1:
        return ThresholdTopologySelection(
            threshold=rows[0].threshold,
            base_threshold=rows[0].threshold,
            delta=0,
            trajectory=rows,
            net_quality=(0.0,),
            knee_curve=(0.0,),
        )

    base = rows[0]
    base_area = max(float(base.area), 1.0)
    base_contour = max(float(base.contour_n), 1.0)
    base_roughness = max(float(base.roughness), 1e-12)
    base_solidity = max(float(base.solidity), 1e-12)
    solidity_headroom = max(1.0 - float(base.solidity), 1e-12)

    net: list[float] = []
    for row in rows:
        contour_cleanup = max(0.0, 1.0 - float(row.contour_n) / base_contour)
        roughness_cleanup = max(0.0, 1.0 - float(row.roughness) / base_roughness)
        solidity_gain = max(
            0.0,
            (float(row.solidity) - float(base.solidity)) / solidity_headroom,
        )
        benefit = (contour_cleanup + roughness_cleanup + solidity_gain) / 3.0

        area_loss = max(0.0, 1.0 - float(row.area) / base_area)
        solidity_loss = max(
            0.0,
            (float(base.solidity) - float(row.solidity)) / base_solidity,
        )
        cost = (area_loss + solidity_loss) / 2.0
        net.append(float(benefit - cost))

    best_so_far = np.maximum.accumulate(np.asarray(net, dtype=np.float64))
    best_value = float(np.max(best_so_far))
    if not math.isfinite(best_value) or best_value <= 0.0:
        selected_index = 0
        knee = np.zeros(len(rows), dtype=np.float64)
    else:
        quality_progress = best_so_far / best_value
        threshold_progress = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
        knee = quality_progress - threshold_progress
        selected_index = int(np.argmax(knee))  # np.argmax gives earliest exact tie.

    selected = rows[selected_index]
    return ThresholdTopologySelection(
        threshold=int(selected.threshold),
        base_threshold=int(base.threshold),
        delta=int(selected.threshold - base.threshold),
        trajectory=rows,
        net_quality=tuple(float(v) for v in net),
        knee_curve=tuple(float(v) for v in knee),
    )


def optimize_separated_threshold(
    full_gray: np.ndarray,
    base_threshold: int,
    seed_point: tuple[int, int],
    base_component: np.ndarray,
    max_delta: int = TOPOLOGY_OPTIMIZATION_STEPS,
) -> ThresholdTopologySelection:
    """Optimize a valid separated T without ever lowering it or invalidating it.

    ``base_threshold`` has already been proven to separate the persistent solar
    component from the background. Topology optimization is therefore a refinement,
    not another resolution stage. If the optional descriptor scan cannot be formed,
    return the proven base threshold as a one-sample selection rather than turning
    a resolved automatic threshold into an unresolved one.
    """
    try:
        trajectory = topology_trajectory_from_separated_component(
            full_gray,
            int(base_threshold),
            tuple(map(int, seed_point)),
            base_component,
            max_delta=max_delta,
        )
        return select_topology_knee(trajectory)
    except (ValueError, cv2.error):
        try:
            base = _component_descriptor(base_component, int(base_threshold))
            trajectory = (base,)
        except (ValueError, cv2.error):
            base = ThresholdTopology(
                threshold=int(base_threshold),
                area=int(np.count_nonzero(base_component)),
                contour_n=0,
                perimeter=0.0,
                roughness=0.0,
                solidity=0.0,
            )
            trajectory = (base,)
        return ThresholdTopologySelection(
            threshold=int(base_threshold),
            base_threshold=int(base_threshold),
            delta=0,
            trajectory=trajectory,
            net_quality=(0.0,),
            knee_curve=(0.0,),
        )
