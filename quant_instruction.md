# Instruction to Quantization 

## How-To

Add M.QAct layer to places you want to quantize the data flow. 

Before initialization (run dumb data through network), start the quantization mode by calling:

```
# This is an example 

model = MyModel()
model.eval()
model.start_quant()
x = torch.zeros(1,3,224,224)
model(x)   # initialize the model with dumb input 

...

```

Then do calibrating 

```
model.start_calibrate()
for data in loader:
    model(data)
model.end_calibrate()
```

After that, your model will run in (fake) quantization mode. But more importantly, you get the quantization parameters for further works (like porting the models to low-end devices). The parameters are saved just by calling: 

```
M.Saver(model).save('./quant_model/quanted.pth')
```

Then, the network parameters and the quant params are saved to './quant_model/quanted.pth' as a state dictionary. 

## Things to notice 

1. Currently the calibration and fake quantization are only supported on CPU. 

2. Convolution input quantizatinon is fixed at layer-wise, and weight quantization is fixed for symmetric quantization. I'm considering to make it controllable by setting layer flags in the future. 

