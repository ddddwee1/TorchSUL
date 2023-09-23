#include <iostream>
#include "onnxruntime_cxx_api.h"


struct GroupNormKernel {
    GroupNormKernel(const OrtKernelInfo* info) {
        Ort::ConstKernelInfo ort_info{info};
        epsilon_ = ort_info.GetAttribute<float>("epsilon");
    }

    void Compute(OrtKernelContext* context){
        Ort::KernelContext ctx(context);
        auto input_X = ctx.GetInput(0);
        const float* X = input_X.GetTensorData<float>();
        auto dimensions = input_X.GetTensorTypeAndShapeInfo().GetShape();
        auto output = ctx.GetOutput(0, dimensions);
        float* out = output.GetTensorMutableData<float>();
        const int64_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

        for (int i=0;i<size;++i){
            out[i] = X[i] * 10;
        }
    }
    
    private:
        float epsilon_;
};


struct GroupNormCustomOp : Ort::CustomOpBase<GroupNormCustomOp, GroupNormKernel> {
//   void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) { return new GroupNormKernel(info); };
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new GroupNormKernel(info);
  }

  const char* GetName() const { return "testmul10"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};
