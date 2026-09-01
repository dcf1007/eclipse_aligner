import cv2
import numpy as np
import pytest
import circle_arc_detector as cad


def oc(mask,kernel):
    u8=np.where(mask,255,0).astype(np.uint8)
    a=cv2.morphologyEx(u8,cv2.MORPH_OPEN,kernel,iterations=1)
    b=cv2.morphologyEx(a,cv2.MORPH_CLOSE,kernel,iterations=1)
    return b!=0


def test_euclidean_disk_footprints_are_exact():
    assert cad.euclidean_disk_kernel(3).sum() == 5
    assert cad.euclidean_disk_kernel(5).sum() == 13
    assert cad.euclidean_disk_kernel(7).sum() == 29
    for size in (3,5,7):
        k=cad.euclidean_disk_kernel(size)
        assert np.array_equal(k,np.rot90(k))
    with pytest.raises(ValueError): cad.euclidean_disk_kernel(4)


def test_candidate_paths_are_direct_and_progressive_open_close():
    raw=np.zeros((61,61),bool)
    cv2.circle(raw.view(np.uint8),(30,30),18,1,-1)
    raw[30,49:55]=True
    candidates=cad.cleanup_morphology_candidates(raw)
    assert tuple(candidates) == cad.CLEANUP_CANDIDATE_ORDER
    k3,k5,k7=(cad.euclidean_disk_kernel(s) for s in (3,5,7))
    assert np.array_equal(candidates['raw'],raw)
    assert np.array_equal(candidates['D3'],oc(raw,k3))
    assert np.array_equal(candidates['D5'],oc(raw,k5))
    assert np.array_equal(candidates['D7'],oc(raw,k7))
    assert np.array_equal(candidates['P35'],oc(oc(raw,k3),k5))
    assert np.array_equal(candidates['P357'],oc(oc(oc(raw,k3),k5),k7))


def test_cleanup_generation_does_not_replace_production_refiner_yet():
    assert cad.SOLAR_COMPONENT_KERNEL.shape == (7,7)
    assert cad.CLEANUP_CANDIDATE_ORDER[0] == 'raw'
