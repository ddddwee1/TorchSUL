#include <torch/all.h>
#include <torch/python.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x "must be cuda tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dcn_cuda(const torch::Tensor value, const torch::Tensor offset, const torch::Tensor mask, const int P, const int G);

torch::Tensor dcn(const torch::Tensor value, const torch::Tensor offset, const torch::Tensor mask, const int P, const int G){
	CHECK_INPUT(value);
	CHECK_INPUT(offset);
	CHECK_INPUT(mask);
	return dcn_cuda(value, offset, mask, P, G);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("dcn", &dcn, "Compute grid sample given value and offset (dcnv3 format)", py::arg("value"), py::arg("offset"), py::arg("mask"), py::arg("P"), py::arg("G"));
	// m.def("dcn", &dcn, "Compute grid sample given value and offset (dcnv3 format)");
}
