import cv2
import numpy as np
import circle_arc_detector as cad


def test_filled_component_has_zero_internal_dark_fraction():
    mask=np.zeros((51,51),bool)
    cv2.circle(mask.view(np.uint8),(25,25),15,1,-1)
    d=cad._component_descriptor(mask,7)
    assert d.internal_dark_fraction == 0.0


def test_internal_dark_fraction_counts_binary_pixels_inside_external_fill():
    mask=np.zeros((51,51),bool)
    cv2.rectangle(mask.view(np.uint8),(10,10),(40,40),1,-1)
    mask[20:25,22:29]=False
    d=cad._component_descriptor(mask,9)
    filled=np.zeros(mask.shape,np.uint8)
    contours,_=cv2.findContours(np.where(mask,255,0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea)
    cv2.drawContours(filled,[contour],-1,1,cv2.FILLED)
    f=filled!=0
    n1=np.count_nonzero(mask & f); n0=np.count_nonzero(f & ~mask)
    assert d.internal_dark_fraction == n0/(n0+n1)
    assert n0 == 35


def test_existing_topology_construction_remains_backward_compatible():
    row=cad.ThresholdTopology(10,100,20,25.0,1.2,0.9)
    assert row.internal_dark_fraction == 0.0
