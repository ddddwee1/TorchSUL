#include <torch/all.h>
#include <torch/python.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcnv3_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight);

std::vector<torch::Tensor> dcnv3_backward_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight, const torch::Tensor grad_out);


torch::Tensor dcnv3(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	CHECK_INPUT(weight);
	return dcnv3_cuda(value, grid, weight);
}

std::vector<torch::Tensor> dcnv3_backward(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor weight, const torch::Tensor grad_out){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	CHECK_INPUT(weight);
	CHECK_INPUT(grad_out);
	return dcnv3_backward_cuda(value, grid, weight, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("grid_sample", &dcnv3, "Grid sample kernel.", py::arg("value"), py::arg("grid"), py::arg("weight"));
	m.def("grid_sample_backward", &dcnv3_backward, "Grid sample backward kernel", py::arg("value"), py::arg("grid"), py::arg("weight"), py::arg("grad_out"));
}
