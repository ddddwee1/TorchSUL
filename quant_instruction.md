# Instruction to Quantization 

## How-To

Add M.QAct layer to places you want to quantize the data flow. 

Before initialization (run dumb data through network), start the quantization mode by calling:

```python
# This is an example 

model = MyModel()
model.eval()
model.start_quant()
x = torch.zeros(1,3,224,224)
model(x)   # initialize the model with dumb input 

...

```

Then do calibrating 

```python
model.start_calibrate()
for data in loader:
    model(data)
model.end_calibrate()
```

After that, your model will run in quantization mode. The parameters are saved just by calling: 

```python
M.Saver(model).save('./quant_model/quanted.pth')
```

Then, the network parameters and the quant params are saved to './quant_model/quanted.pth' as a state dictionary. 

If you only want to save the quant parameters (zero-point and scale), you can use

```python 
quant_dict = M.inspect_quant_params(model)
torch.save(quant_dict, 'quant_params.pth')
```

#### Setting quant bit 

You can set the quantization bit of all Act layers using flags. For example, if you want to set the quant bit to int16:

```python
model.set_flag('QActBit', 'int16')
```

You probably want to use different types for specific layers. You can just create QAct layers by specifying bit type and this will override the QActBit flag for this layer: 

```python
act1 = M.QAct(bit_type='uint8')
```

#### Setting observer type 

You can set the observer type (available: minmax, percentile, omse) by:

```python
model.set_flag('QActObserver', 'omse')
```

or you can use different type of for specific layers:

```python
act1 = M.QAct(observer='percentile')
```

## Things to notice 

1. Convolution input quantizatinon is fixed at layer-wise, and weight quantization is fixed for symmetric quantization. I'm considering to make it controllable by setting layer flags in the future. 

