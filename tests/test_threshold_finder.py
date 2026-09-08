import math
import cv2
import numpy as np
import circle_arc_detector as tf

def synthetic_bridge_image():
    g=np.zeros((220,320),np.uint8); cv2.circle(g,(180,110),46,30,-1); g[106:115,0:180]=8; cv2.circle(g,(190,105),12,70,-1); return g

def run_auto(gray):
    state={'settings':tf.ImageSettings(),'auto_threshold_result':None,'solar_data':None}; t=tf.find_auto_threshold(gray,state); return t,state['auto_threshold_result']

def test_threshold_semantics_and_lowest_separation_refinement():
    t,r=run_auto(synthetic_bridge_image()); assert t==r.full_res_refined_threshold==8 and r.full_res_separation_threshold==8

def test_histogram_start_threshold_is_left_of_rightmost_mode(): assert 0<=tf.find_histogram_start_threshold(synthetic_bridge_image())<70

def test_unresolved_auto_returns_none_and_records_failure_not_histogram_fallback():
    g=np.full((80,120),100,np.uint8); start=tf.find_histogram_start_threshold(g); t,r=run_auto(g); assert t is None and r.failure_reason is not None and r.histogram_start_threshold==start

def test_sqrt_pixel_scale_preserves_linear_scaling_and_10_percent_guard():
    base=math.sqrt(6016*4000); half=math.sqrt(3008*2000); assert abs(half/base-.5)<1e-12; assert round(tf.AUTO_T_GUARD_DILATION_FRACTION*base)==491

def test_working_resize_is_1200_max_and_grayscale_area():
    g=(np.arange(2000*1000,dtype=np.uint32).reshape(2000,1000)%256).astype(np.uint8); shape=tf.calculate_work_res_shape(g.shape); w=tf.resize_img(g,shape)
    assert shape==(1200,600) and w.shape==(1200,600) and w.ndim==2
