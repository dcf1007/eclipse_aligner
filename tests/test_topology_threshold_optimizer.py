import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
P = HERE.parents[1] / 'snippets' / 'topology_threshold_optimizer.py'
spec = importlib.util.spec_from_file_location('topopt', P)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

T = m.ThresholdTopology


def row(t, area, n, rough, sol):
    return T(t, area, n, float(n), rough, sol)


def test_strong_first_cleanup_selects_t_plus_one():
    trajectory = (
        row(10,10000,10000,10.0,.80),
        row(11,9400,3000,3.2,.88),
        row(12,9300,2200,2.5,.90),
        row(13,9250,1900,2.3,.905),
        row(14,9200,1800,2.2,.907),
        row(15,9150,1750,2.15,.908),
    )
    result = m.select_topology_knee(trajectory)
    assert result.threshold == 11
    assert result.delta == 1


def test_no_positive_topology_improvement_keeps_separation_t():
    trajectory = (
        row(20,10000,1000,2.0,.90),
        row(21,9500,1050,2.1,.89),
        row(22,9000,1100,2.2,.88),
        row(23,8500,1150,2.3,.87),
    )
    result = m.select_topology_knee(trajectory)
    assert result.threshold == 20
    assert result.delta == 0


def test_later_real_cleanup_can_select_second_step():
    trajectory = (
        row(30,10000,10000,10.0,.80),
        row(31,9800,8500,8.5,.805),
        row(32,9500,3500,3.8,.86),
        row(33,9400,3000,3.4,.87),
        row(34,9350,2850,3.3,.872),
        row(35,9300,2800,3.25,.873),
    )
    result = m.select_topology_knee(trajectory)
    assert result.threshold == 32
    assert result.delta == 2


def test_nested_component_scan_matches_direct_full_frame_area():
    import cv2
    import numpy as np
    gray = np.zeros((120,180),np.uint8)
    cv2.circle(gray,(95,60),34,30,-1)
    gray[55:66,20:95] = 9
    cv2.circle(gray,(100,58),7,80,-1)
    seed=(100,58); base_t=9
    direct=cv2.compare(gray,base_t,cv2.CMP_GT); cv2.floodFill(direct,None,seed,128,flags=8)
    base=direct==128
    traj=m.topology_trajectory_from_separated_component(gray,base_t,seed,base,5)
    for r in traj:
        direct=cv2.compare(gray,r.threshold,cv2.CMP_GT); cv2.floodFill(direct,None,seed,128,flags=8)
        assert r.area == int(np.count_nonzero(direct==128))


def test_optimize_separated_threshold_never_lowers_valid_base_t():
    import cv2
    import numpy as np
    gray=np.zeros((120,180),np.uint8)
    cv2.circle(gray,(95,60),34,30,-1)
    gray[55:66,20:95]=9
    cv2.circle(gray,(100,58),7,80,-1)
    seed=(100,58); base_t=9
    direct=cv2.compare(gray,base_t,cv2.CMP_GT); cv2.floodFill(direct,None,seed,128,flags=8)
    base=direct==128
    result=m.optimize_separated_threshold(gray,base_t,seed,base,5)
    assert result.threshold >= base_t
    assert result.threshold <= base_t + 5


def test_optimize_separated_threshold_falls_back_to_base_on_bad_scan():
    import numpy as np
    gray=np.zeros((10,10),np.uint8)
    component=np.zeros_like(gray,dtype=bool)
    result=m.optimize_separated_threshold(gray,7,(5,5),component,5)
    assert result.threshold == 7
    assert result.delta == 0
