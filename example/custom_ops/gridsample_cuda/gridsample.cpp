#include <torch/all.h>
#include <torch/python.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gridsample_cuda(torch::Tensor value, torch::Tensor grid);

torch::Tensor gridsample(torch::Tensor value, torch::Tensor grid){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	return gridsample_cuda(value, grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("grid_sample", &gridsample, "Compute grid sample given value and grid");
}
