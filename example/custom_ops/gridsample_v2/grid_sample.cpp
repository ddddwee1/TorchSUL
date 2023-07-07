#include <torch/all.h>
#include <torch/python.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gridsample_cuda(const torch::Tensor value, const torch::Tensor grid);

std::vector<torch::Tensor> gridsample_backward_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out);


torch::Tensor gridsample(const torch::Tensor value, const torch::Tensor grid){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	return gridsample_cuda(value, grid);
}

std::vector<torch::Tensor> gridsample_backward(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	CHECK_INPUT(grad_out);
	return gridsample_backward_cuda(value, grid, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("grid_sample", &gridsample, "Grid sample kernel.", py::arg("value"), py::arg("grid"));
	m.def("grid_sample_backward", &gridsample_backward, "Grid sample backward kernel", py::arg("value"), py::arg("grid"), py::arg("grad_out"));
}
