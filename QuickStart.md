# Quick Start

Take some time to know this SUL model.

## Creating a Model

Let's create a small pytorch model first:

```python
import torch 
import torch.nn as nn 

class TorchModule(nn.Module):
    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.c1 = nn.Conv2d(in_chn, out_chn, 5, padding=2)
        self.c2 = nn.Conv2d(out_chn, out_chn, 3, padding=1)
        self.c3 = nn.Conv2d(out_chn, out_chn, 3, padding=0)
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x 

x = torch.randn(1, 3, 32, 32)
torch_mod = TorchModule(3, 16)
y = torch_mod(x)
```

We can build a SUL model in a very similar way. 

```python
from TorchSUL import Model as M 

class SULModel(M.Model):
    def initialize(self, out_chn):
        self.c1 = M.ConvLayer(5, out_chn)
        self.c2 = M.ConvLayer(3, out_chn)
        self.c3 = M.ConvLayer(3, out_chn, pad='VALID')
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x 

sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod(x_dummy)    # This line is to initialize the mod
x_real = torch.randn(1, 3, 32, 32)
y = sul_mod(x_real)
```

In summary, there are 4 differences between nn.Module and SULModel

    1. Change the __init__() method to initialize() method, and remove the super().__init__() usage

    2. You don't need to specify the number of input channels for each layer.

    3. For same-padding layers (output preserves the same shape as input), you dont need to specify the padding pixels, as it will be automatically computed. For zero-padding layers, just specify pad='VALID'.

    4. Before running the model, you need to feed it with a dummy input first. In this dummy-forward, the model will infer the tensor shapes and build the parameters for each layer. 

## Loading from standard pytorch checkpoint 

Changing a model from a standard pytorch module to SULModel is simple as above, but the state dicts are different. SUL provides an option to load from standard pytorch checkpoints by specifying a flag:

```python
sul_mod.load_state_dict(torch_mod.state_dict())    # This will give errors. Param names are not compatible 
sul_mod.set_flag('from_torch')               
sul_mod.load_state_dict(torch_mod.state_dict())    # This will work properly by stating the "from_torch" flag
```

## Changing defualt initialization methods

Before initialization (feeding dummy into the network), you can specify the param initialization methods for conv or fc layers. 

```python
sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod.set_flag('conv_init_mode', 'normal')   # normal(mean=0, std=0.0001)
sul_mod.set_flag('fc_init_mode', 'kaiming')
sul_mod(x_dummy)    # This line is to initialize the mod
```

## Customize param initialization 

You may implement the init_params() method to customize the param initialization for the specific model 

```python
class SULModel(M.Model):
    def initialize(self, out_chn):
        self.c1 = M.ConvLayer(5, out_chn)
        self.c2 = M.ConvLayer(3, out_chn)
        self.c3 = M.ConvLayer(3, out_chn, pad='VALID')
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x 

    def init_params(self, x):
        nn.init.ones_(self.c1.conv.bias)
        nn.init.zeros_(self.c2.conv.weight)
        nn.init.kaiming_normal_(self.c3.conv.weight)
```

## Use flags to control the forward flow

Normally it's recommended to use keyword args to control the forward flow. But in some cases, setting a global configuration is more convenient. For example, when you have different strategies for inference and deployment (e.g. YOLO), you may need to omit some of blocks/layers/ops for exporting onnx/trt model. 

Following is a small example to achieve it with flags:

```python
class SULModel(M.Model):
    def initialize(self, out_chn):
        self.c1 = M.ConvLayer(5, out_chn)
        self.c2 = M.ConvLayer(3, out_chn)
        self.c3 = M.ConvLayer(3, out_chn, pad='VALID')
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        if not self.get_flag('deploy'):
            x = self.c3(x)
        return x 

sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod(x_dummy)                        # This line is to initialize the mod
x_real = torch.randn(1, 3, 32, 32)
y = sul_mod(x_real)

sul_mod.set_flag('deploy')              # setting this flag will disable forwarding through c3
torch.onnx.export(sul_mod, x_real, "sample.onnx", input_names=['input'], output_names=['output'])
```

## Inspecting intermediate layers 

SUL provides a way to inspect the intermediate values from model:

```python
class SULModel(M.Model):
    def initialize(self, out_chn):
        self.c1 = M.ConvLayer(5, out_chn)
        self.c2 = M.ConvLayer(3, out_chn)
        self.c3 = M.ConvLayer(3, out_chn, pad='VALID')
    
    def forward(self, x):
        x = self.c1(x)
        self.save_tensor(x, 'c1_output')         
        x = self.c2(x)
        self.save_tensor(x, 'c2_output')
        x = self.c3(x)
        self.save_tensor(x, 'c3_output')
        return x 

sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod(x_dummy)                        # This line is to initialize the mod

sul_mod.set_flag('save_tensor')         # set the "save_tensor" flag to enbale save tensor function 
x_real = torch.randn(1, 3, 32, 32)
y = sul_mod(x_real)
```

After this, you will see the saved tensors in "layer_dumps" folder. 

Together with the flag mechanism, you can further save tensors from different iterations. For example, if I would like to save intermediate results from odd iterations: 

```python
class SULModel(M.Model):
    def initialize(self, out_chn):
        self.c1 = M.ConvLayer(5, out_chn)
        self.c2 = M.ConvLayer(3, out_chn)
        self.c3 = M.ConvLayer(3, out_chn, pad='VALID')
    
    def forward(self, x):
        n_iter = self.get_flag('n_iter')
        x = self.c1(x)
        self.save_tensor(x, 'iter_%d_c1_output'%n_iter)         
        x = self.c2(x)
        self.save_tensor(x, 'iter_%d_c2_output'%n_iter)
        x = self.c3(x)
        self.save_tensor(x, 'iter_%d_c3_output'%n_iter)
        return x 

sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod(x_dummy)                                     # This line is to initialize the mod


for it in range(100):
    sul_mod.set_flag('save_tensor', it%2==1)         # save tensor for odd iterations 
    sul_mod.set_flag('n_iter', it)                   # passing n_iteration to flags 
    x_real = torch.randn(1, 3, 32, 32)
    y = sul_mod(x_real)
```

With this approach, the users can inspect & manipulate the forward flow with minimum hurt to the model structure and function definition. 

## Loosely load from checkpoint 

Sometimes, the weights in the checkpoint are not the same as your model definition. Then, you can specify the "loose_load" flag and strict=False to achieve this.

```python
sul_mod = SULModel(16)
x_dummy = torch.randn(1, 3, 32, 32)
sul_mod(x_dummy)                                     # This line is to initialize the mod

ckpt = torch.load('checkpoint.pth')
sul_mod.set_flag('loose_load')
sul_mod.load_state_dict(ckpt, strict=False)
```

## Using yaml config file 

SUL provides a lightweight config implementation. The values can be accessed as simple as using an attribute (X.Y.Z instead of X['Y']['Z'])

```python
from TorchSUL import Model as M 
from TorchSUL import Config 

class SULModel(M.Model):
    def initialize(self, cfg):
        self.cfg = cfg 
        self.c1 = M.ConvLayer(5, cfg.MODEL.OUT_CHN1)
        self.c2 = M.ConvLayer(3, cfg.MODEL.OUT_CHN2)
        self.c3 = M.ConvLayer(3, cfg.MODEL.OUT_CHN3, pad='VALID')
    
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        if self.cfg.FORWARD.DUPLICATE_C3:
            x = self.c3(x)
        return x 

cfg = Config.load_yaml('config.yaml')
sul_mod = SULModel(cfg)
```

Where the config.yaml is like

```yaml
MODEL:
  OUT_CHN1: 16
  OUT_CHN2: 32
  OUT_CHN3: 64
FORWARD:
  DUPLICATE_C3: true
```

## Model quantization

See [quant instruction](https://github.com/ddddwee1/TorchSUL/blob/master/QuantInstruction.md) for detailed information.

## Compatibility with Pytorch 

This SUL is just a modification to Pytorch nn.Modules. The training/testing pipeline is not different from standard [pytorch practice](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).

