# FastGausMap
 Fast render Gaussian map!
 This gpu version only supports pts to be cuda tensors. I'm lazy, so don't want to write cpu tensor op for now. 

## Install

python setup.py build_ext --inplace

## Usage 

render_heatmap(pts, sizeh, sizew, sigma)

- pts: shape: [Num_batch, Num_pts, 3] 
- sizeh, sizew: heatmap size 
- sigma: sigma of Gaussian kernel
- Return: hmap: float32 [Num_batch, Num_pts, sizeh, sizew]

For each keypoint, 3 numbers are accepted (x,y,confidence). Only points with confidence > 0 will be rendered.

## Example

Please check "go.py"
