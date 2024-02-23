import torch.nn as nn
from loguru import logger
from torch import Tensor

from .. import Layers as L
from ..Base import Model
from ..Consts.Activate import *
from ..Consts.Types import *
from ..Quant import QuantizerBase


class ConvLayer(Model):
    conv: L.conv2D
    batch_norm: bool 
    bn: L.BatchNorm
    act: L.Activation

    def __init__(self, size: TypeKSize2D, outchn:int, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, activation=activation,
                         batch_norm=batch_norm, affine=affine, usebias=usebias, groups=groups)
        
    def initialize(self, size: TypeKSize2D, outchn:int, stride:TypeKSize2D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        self.conv = L.conv2D(size, outchn, stride, pad, dilation_rate, usebias, groups)
        if batch_norm:
            self.bn = L.BatchNorm(affine=affine)
        self.batch_norm = batch_norm
        self.act = L.Activation(activation)

    def forward(self, x) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def to_torch_module(self):
        conv = self.conv.to_torch()
        setattr(self, 'conv', conv)
        if self.batch_norm:
            bn = self.bn.to_torch()
            setattr(self, 'bn', bn)
        if self.activation==PARAM_RELU:
            relu = nn.ReLU()
            setattr(self, 'act', relu)

    def get_input_quantizer(self) -> QuantizerBase:
        return self.conv.input_quantizer

    def get_weight_quantizer(self) -> QuantizerBase:
        return self.conv.w_quantizer

    def set_input_quantizer(self, quantizer: QuantizerBase):
        self.conv.set_input_quantizer(quantizer)

    def _load_from_state_dict2(self, state_dict, prefix):
        def _load_weight(k):
            if not k in state_dict:
                raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Cannot find weight in checkpoint for layer: [{prefix}]')
            return state_dict.pop(k)
        def _load_bias(k):
            if self.conv.usebias:
                try:
                    b = state_dict.pop(k)
                except:
                    raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Bias is set for layer [{prefix}] but not found in checkpoint.')
            else:
                b = None
            return b 
        
        if not self._is_built:
            logger.warning(f'Layer: [{prefix}] is not built. This layer was not used in dummy forward. Skipping loading state_dict.')
            return 

        # get names for params
        if prefix+'conv.weight' in state_dict:
            # normal load 
            w = prefix + 'conv.weight'
            b = prefix + 'conv.bias'
        elif self.get_flag('fc2conv') and ((prefix+'fc.weight') in state_dict):
            w = prefix + 'fc.weight'
            b = prefix + 'fc.bias'
        elif self.get_flag('from_torch'):
            w = prefix + 'weight'
            b = prefix + 'bias'
        else:
            raise KeyError(f'Cannot find weight in checkpoint for layer: [{prefix}]')

        # laod weight and bias 
        w = _load_weight(w)
        b = _load_bias(b)
        if self.get_flag('fc2conv') and len(w.shape)==2:
            w = w.unsqueeze(-1).unsqueeze(-1)

        # write processed params to state dict 
        state_dict[prefix+'conv.weight'] = w 
        if b is not None:
            state_dict[prefix+'conv.bias'] = b


class DeConvLayer(Model):
    conv: L.deconv2D
    batch_norm: bool 
    bn: L.BatchNorm
    act: L.Activation

    def __init__(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, activation=activation,
                         batch_norm=batch_norm, affine=affine, usebias=usebias, groups=groups)
        
    def initialize(self, size:int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        self.conv = L.deconv2D(size, outchn, stride, pad, dilation_rate, usebias, groups)
        if batch_norm:
            self.bn = L.BatchNorm(affine=affine)
        self.batch_norm = batch_norm
        self.act = L.Activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch_module(self):
        conv = self.conv.to_torch()
        setattr(self, 'conv', conv)
        if self.batch_norm:
            bn = self.bn.to_torch()
            setattr(self, 'bn', bn)
        if self.activation==PARAM_RELU:
            relu = nn.ReLU()
            setattr(self, 'act', relu)

    def get_input_quantizer(self) -> QuantizerBase:
        return self.conv.input_quantizer

    def get_weight_quantizer(self) -> QuantizerBase:
        return self.conv.w_quantizer

    def set_input_quantizer(self, quantizer: QuantizerBase):
        self.conv.set_input_quantizer(quantizer)

    def _load_from_state_dict2(self, state_dict, prefix):
        def _load_weight(k):
            if not k in state_dict:
                raise Exception(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}]. Cannot find weight in checkpoint for layer: [{prefix}]')
            return state_dict.pop(k)
        def _load_bias(k):
            if self.conv.usebias:
                try:
                    b = state_dict.pop(k)
                except:
                    raise Exception(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Bias is set for layer [{prefix}] but not found in checkpoint.')
            else:
                b = None
            return b 
        
        if not self._is_built:
            logger.warning(f'Layer: [{prefix}] is not built. This layer was not used in dummy forward. Skipping loading state_dict.')
            return 

        # get names for params
        if prefix+'conv.weight' in state_dict:
            # normal load 
            w = prefix + 'conv.weight'
            b = prefix + 'conv.bias'
        elif self.get_flag('from_torch'):
            w = prefix + 'weight'
            b = prefix + 'bias'
        else:
            raise KeyError(f'Cannot find weight in checkpoint for layer: [{prefix}]')

        # laod weight and bias 
        w = _load_weight(w)
        b = _load_bias(b)

        # write processed params to state dict 
        state_dict[prefix+'conv.weight'] = w 
        if b is not None:
            state_dict[prefix+'conv.bias'] = b


class ConvLayer1D(Model):
    conv: L.conv1D
    batch_norm: bool
    bn: L.BatchNorm
    act: L.Activation

    def __init__(self, size: int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, activation=activation,
                         batch_norm=batch_norm, affine=affine, usebias=usebias, groups=groups)
        
    def initialize(self, size: int, outchn:int, stride:int=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        self.conv = L.conv1D(size, outchn, stride, pad, dilation_rate, usebias, groups)
        if batch_norm:
            self.bn = L.BatchNorm(affine=affine)
        self.batch_norm = batch_norm
        self.act = L.Activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch_module(self):
        conv = self.conv.to_torch()
        setattr(self, 'conv', conv)
        if self.batch_norm:
            bn = self.bn.to_torch()
            setattr(self, 'bn', bn)
        if self.activation==PARAM_RELU:
            relu = nn.ReLU()
            setattr(self, 'act', relu)

    def get_input_quantizer(self) -> QuantizerBase:
        return self.conv.input_quantizer

    def get_weight_quantizer(self) -> QuantizerBase:
        return self.conv.w_quantizer

    def set_input_quantizer(self, quantizer: QuantizerBase):
        self.conv.set_input_quantizer(quantizer)
    
    def _load_from_state_dict2(self, state_dict, prefix):
        def _load_weight(k):
            if not k in state_dict:
                raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Cannot find weight in checkpoint for layer: [{prefix}]')
            return state_dict.pop(k)
        def _load_bias(k):
            if self.conv.usebias:
                try:
                    b = state_dict.pop(k)
                except:
                    raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Bias is set for layer [{prefix}] but not found in checkpoint.')
            else:
                b = None
            return b 
        
        if not self._is_built:
            logger.warning(f'Layer: [{prefix}] is not built. This layer was not used in dummy forward. Skipping loading state_dict.')
            return 

        # get names for params
        if prefix+'conv.weight' in state_dict:
            # normal load 
            w = prefix + 'conv.weight'
            b = prefix + 'conv.bias'
        elif self.get_flag('from_torch'):
            w = prefix + 'weight'
            b = prefix + 'bias'
        else:
            raise KeyError(f'Cannot find weight in checkpoint for layer: [{prefix}]')

        # laod weight and bias 
        w = _load_weight(w)
        b = _load_bias(b)

        # write processed params to state dict 
        state_dict[prefix+'conv.weight'] = w 
        if b is not None:
            state_dict[prefix+'conv.bias'] = b


class ConvLayer3D(Model):
    conv: L.conv3D
    batch_norm: bool
    bn: L.BatchNorm
    act: L.Activation

    def __init__(self, size: TypeKSize3D, outchn:int, stride:TypeKSize3D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        super().__init__(size=size, outchn=outchn, stride=stride, pad=pad, dilation_rate=dilation_rate, activation=activation,
                         batch_norm=batch_norm, affine=affine, usebias=usebias, groups=groups)
        
    def initialize(self, size: TypeKSize3D, outchn:int, stride:TypeKSize3D=1, pad:PadModes='SAME_LEFT', dilation_rate:int=1, activation:int=-1,
                 batch_norm:bool=False, affine:bool=True, usebias:bool=True, groups:int=1):
        self.conv = L.conv3D(size, outchn, stride, pad, dilation_rate, usebias, groups)
        if batch_norm:
            self.bn = L.BatchNorm(affine=affine)
        self.batch_norm = batch_norm
        self.act = L.Activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x 
    
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
    
    def to_torch_module(self):
        conv = self.conv.to_torch()
        setattr(self, 'conv', conv)
        if self.batch_norm:
            bn = self.bn.to_torch()
            setattr(self, 'bn', bn)
        if self.activation==PARAM_RELU:
            relu = nn.ReLU()
            setattr(self, 'act', relu)

    def get_input_quantizer(self) -> QuantizerBase:
        return self.conv.input_quantizer

    def get_weight_quantizer(self) -> QuantizerBase:
        return self.conv.w_quantizer

    def set_input_quantizer(self, quantizer: QuantizerBase):
        self.conv.set_input_quantizer(quantizer)
    
    def _load_from_state_dict2(self, state_dict, prefix):
        def _load_weight(k):
            if not k in state_dict:
                raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Cannot find weight in checkpoint for layer: [{prefix}]')
            return state_dict.pop(k)
        def _load_bias(k):
            if self.conv.usebias:
                try:
                    b = state_dict.pop(k)
                except:
                    raise KeyError(f'Attenpt to find [{k}] but only exist [{state_dict.keys()}] Bias is set for layer [{prefix}] but not found in checkpoint.')
            else:
                b = None
            return b 
        
        if not self._is_built:
            logger.warning(f'Layer: [{prefix}] is not built. This layer was not used in dummy forward. Skipping loading state_dict.')
            return 

        # get names for params
        if prefix+'conv.weight' in state_dict:
            # normal load 
            w = prefix + 'conv.weight'
            b = prefix + 'conv.bias'
        elif self.get_flag('from_torch'):
            w = prefix + 'weight'
            b = prefix + 'bias'
        else:
            raise KeyError(f'Cannot find weight in checkpoint for layer: [{prefix}]')
        
        # laod weight and bias 
        w = _load_weight(w)
        b = _load_bias(b)

        # write processed params to state dict 
        state_dict[prefix+'conv.weight'] = w 
        if b is not None:
            state_dict[prefix+'conv.bias'] = b

