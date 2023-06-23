#include <torch/all.h>
#include <torch/python.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_SHAPE_PTS(x) AT_ASSERTM(torch::_shape_as_tensor(x).size(0)==3, #x "must be a 3-dim tensor")
#define CHECK_LAST_DIM_PTS(x) AT_ASSERTM(x.size(2)==3, #x "'s last dim should be 3")

torch::Tensor generate_hmap_cuda(torch::Tensor pts, int H, int W, float sigma);

torch::Tensor generate_hmap(torch::Tensor pts, int H, int W, float sigma){
	CHECK_INPUT(pts);
	CHECK_SHAPE_PTS(pts);
	CHECK_LAST_DIM_PTS(pts);
	return generate_hmap_cuda(pts, H, W, sigma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("render_heatmap", &generate_hmap, "Generate heatmap given points");
}
