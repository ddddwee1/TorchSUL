import torch 
import onnx_runner 

x = torch.ones(1, 3, 10)
y = onnx_runner.run_onnx('dumb.onnx', [x])
print(y)
