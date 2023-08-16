# Model flag list 

Default flags in model

| Key | Value type | Description | Supported values |
|---|---|---|---|
| 'QActBit' | str | Bit type for quant activation | ['int8', 'uint8', 'int16'] |
| 'QActObserver' | str | Observer type for quant activation | ['minmax', 'omse', 'percentile'] |
| 'dump_onnx' | bool | Open when dumping to onnx which will remove certain duplicated layers | True / False |
| 'fc2conv' | bool | Automatically load the fc layers' weight in state_dict to 1x1 conv layers | True / False |
| 'save_tensor' | bool | Enable the function: Model.save_tensor | True / False | 
| 'conv_init_mode' | str | Use normal initializtion for conv layers with mean=0 and std=0.001 | ['normal', 'resnet', 'kaiming'] |
| 'fc_init_mode' | str | Similar to above, but for fc layers. | ['normal', 'kaiming'] |
| 'from_torch' | bool | Convert weights from standard pytorch to TorchSUL | True / False |
| 'loose_load' | bool | Whether ignore missing keys when loading from state dict| True / False |
