# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np 
cimport numpy as np 
import cv2 

cdef extern from "render_core.h":
	void _get_gaus(float* map, int sizeh, int sizew, float sigma, float* pts, int n_batch, int n_pts)

def render_heatmap(np.ndarray[float, ndim=3, mode='c'] pts, int sizeh, int sizew, float sigma):
	cdef int n_batch = pts.shape[0]
	cdef int n_pts = pts.shape[1]
	hmap = np.zeros([n_batch, n_pts, sizeh, sizew], dtype=np.float32)
	# for i in range(n_batch):
	# 	print(i)
	_get_gaus(<float*> np.PyArray_DATA(hmap), sizeh, sizew, sigma, <float*> np.PyArray_DATA(pts), n_batch, n_pts)
	return hmap

