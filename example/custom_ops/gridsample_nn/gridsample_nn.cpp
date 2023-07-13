#include <torch/all.h>
#include <torch/python.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be a cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gridsample_nn_cuda(const torch::Tensor value, const torch::Tensor grid);

std::vector<torch::Tensor> gridsample_nn_back_cuda(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out);

torch::Tensor gridsample_nn(const torch::Tensor value, const torch::Tensor grid){
	CHECK_INPUT(value);
	CHECK_CONTIGUOUS(grid);
	return gridsample_nn_cuda(value, grid);
}

std::vector<torch::Tensor> gridsample_nn_back(const torch::Tensor value, const torch::Tensor grid, const torch::Tensor grad_out){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	// CHECK_INPUT(grad_out);
	CHECK_CUDA(grad_out);
	const torch::Tensor grad_out_cont = grad_out.contiguous();
	return gridsample_nn_back_cuda(value, grid, grad_out_cont);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("grid_sample", &gridsample_nn, "Nearest grid sample with gradient", py::arg("value"), py::arg("grid"));
	m.def("grid_sample_back", &gridsample_nn_back, "Nearest grid sample backward", py::arg("value"), py::arg("grid"), py::arg("grad_out"));
}
