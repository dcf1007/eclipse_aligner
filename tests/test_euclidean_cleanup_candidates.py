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
    assert cad.generate_kernel((3, 3), round_kernel=True).sum() == 5
    assert cad.generate_kernel((5, 5), round_kernel=True).sum() == 13
    assert cad.generate_kernel((7, 7), round_kernel=True).sum() == 29
    for size in (3,5,7):
        k=cad.generate_kernel((size, size), round_kernel=True)
        assert np.array_equal(k,np.rot90(k))
    with pytest.raises(ValueError): cad.generate_kernel((4, 4), round_kernel=True)




def test_generate_kernel_supports_rectangles_and_ellipses():
    rectangle = cad.generate_kernel((5, 3))
    assert rectangle.shape == (3, 5)
    assert np.all(rectangle == 1)

    ellipse = cad.generate_kernel((5, 3), round_kernel=True)
    expected = np.array(
        [
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(ellipse, expected)

    # The new two-dimensional API must preserve the established disk exactly.
    assert np.array_equal(
        cad.generate_kernel((5, 5), round_kernel=True),
        cad.generate_kernel((5, 5), round_kernel=True),
    )

    # Degenerate one-pixel dimensions remain valid centered ellipses.
    assert np.all(cad.generate_kernel((1, 5), round_kernel=True) == 1)
    assert np.all(cad.generate_kernel((5, 1), round_kernel=True) == 1)

    with pytest.raises(ValueError):
        cad.generate_kernel((4, 3))
    with pytest.raises(ValueError):
        cad.generate_kernel((5, 2), round_kernel=True)

def test_candidate_paths_are_direct_and_progressive_open_close():
    raw=np.zeros((61,61),bool)
    cv2.circle(raw.view(np.uint8),(30,30),18,1,-1)
    raw[30,49:55]=True
    candidates=cad.cleanup_morphology_candidates(raw)
    assert tuple(candidates) == cad.CLEANUP_CANDIDATE_ORDER
    k3,k5,k7=(cad.generate_kernel((s, s), round_kernel=True) for s in (3,5,7))
    assert np.array_equal(candidates['raw'],raw)
    assert np.array_equal(candidates['D3'],oc(raw,k3))
    assert np.array_equal(candidates['D5'],oc(raw,k5))
    assert np.array_equal(candidates['D7'],oc(raw,k7))
    assert np.array_equal(candidates['P35'],oc(oc(raw,k3),k5))
    assert np.array_equal(candidates['P357'],oc(oc(oc(raw,k3),k5),k7))


def test_cleanup_generation_does_not_replace_production_refiner_yet():
    assert cad.SOLAR_COMPONENT_KERNEL.shape == (7,7)
    assert cad.CLEANUP_CANDIDATE_ORDER[0] == 'raw'
