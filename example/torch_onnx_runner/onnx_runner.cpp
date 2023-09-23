#include <torch/all.h>
#include <torch/python.h>
#include <vector>
#include "onnxruntime_cxx_api.h"
#include "cust_ops.cpp"

std::vector<torch::Tensor> run_onnx(const char* model_path, const std::vector<torch::Tensor> inputs){
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::SessionOptions session_option;

    GroupNormCustomOp custom_op;
    Ort::CustomOpDomain custom_op_domain("custom_ops");
    custom_op_domain.Add(&custom_op);
    session_option.Add(custom_op_domain);
    
    Ort::Session session(env, model_path, session_option);

    std::vector<const char*> input_names;
    std::vector<int64_t*> input_shapes;
    std::vector<size_t> input_dims;
    std::vector<Ort::AllocatedStringPtr> NodeNameAllocatedStrings_in;
    std::vector<std::vector<int64_t>> all_in_shapes;

    size_t inputCount = session.GetInputCount();
    for (int i=0; i<inputCount; i++){
        auto name = session.GetInputNameAllocated(i, allocator);
        NodeNameAllocatedStrings_in.push_back(std::move(name));
        input_names.push_back(NodeNameAllocatedStrings_in.back().get());
        std::vector<int64_t> shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        all_in_shapes.push_back(std::move(shape));
        input_shapes.push_back(all_in_shapes.back().data());
        input_dims.push_back(all_in_shapes.back().size());
    }
    for (int i=0; i<input_names.size(); i++){
        std::cout << "input name:" << input_names.data()[i] << std::endl;
    }

    std::vector<const char*> output_names;
    std::vector<int64_t*> output_shapes;
    std::vector<size_t> output_dims;
    std::vector<Ort::AllocatedStringPtr> NodeNameAllocatedStrings_out;
    std::vector<std::vector<int64_t>> all_out_shapes;

    size_t outputCount = session.GetOutputCount();
    for (int i=0; i<outputCount; i++){
        auto name = session.GetOutputNameAllocated(i, allocator);
        NodeNameAllocatedStrings_out.push_back(std::move(name));
        output_names.push_back(NodeNameAllocatedStrings_out.back().get());
        std::vector<int64_t> shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        all_out_shapes.push_back(std::move(shape));
        output_shapes.push_back(all_out_shapes.back().data());
        output_dims.push_back(all_out_shapes.back().size());
    }
    for (int i=0; i<output_names.size(); i++){
        std::cout << "output name:" << output_names.data()[i] << std::endl;
    }


    std::vector<Ort::Value> input_tensors;
    for (int i=0; i<input_names.size(); i++){
        Ort::Value v = Ort::Value::CreateTensor<float>(memory_info, inputs[i].data<float>(), inputs[i].numel(), input_shapes[i], input_dims[i]);
        input_tensors.push_back(std::move(v));
    }

    auto output_tensors = session.Run(Ort::RunOptions(nullptr), input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());

    std::vector<torch::Tensor> results;
    for (int cnt=0; cnt<output_names.size(); cnt++){
        float* output_matrix = output_tensors[cnt].GetTensorMutableData<float>();
        std::vector<int64_t> shape = all_out_shapes[cnt];
        auto shaperef = c10::makeArrayRef(shape);
        auto result = torch::zeros(shaperef);
        float* res_ptr = result.data<float>();
        for (int i=0;i<result.numel();i++){
            res_ptr[i] = output_matrix[i];
        }
        results.emplace_back(result);
    }
    return results;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("run_onnx", &run_onnx, "Run onnx with given input", py::arg("name"), py::arg("inputs"));
}
