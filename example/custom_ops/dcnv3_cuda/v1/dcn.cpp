#include <torch/all.h>
#include <torch/python.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcn_cuda(torch::Tensor value, torch::Tensor grid, torch::Tensor mask);

torch::Tensor dcn(torch::Tensor value, torch::Tensor grid, torch::Tensor mask){
	CHECK_INPUT(value);
	CHECK_INPUT(grid);
	CHECK_INPUT(mask);
	return dcn_cuda(value, grid, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("dcn", &dcn, "Compute grid sample given value and grid (dcnv3 format)");
}
