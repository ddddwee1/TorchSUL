# FastGausMap
 Fast render Gaussian map!

## Install

python setup.py build_ext --inplace

## Usage 

render_heatmap(pts, size, sigma)

- pts: shape: [Num_pts, 3] 
- size: heatmap size 
- sigma: sigma
- Return: hmap: float32 [Num_pts, size, size]

For keypoint, 3 numbers are accepted (x,y,confidence). Only points with confidence > 0 will be rendered.

This will be 10x faster than python version.
