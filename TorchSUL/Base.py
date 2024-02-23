import functools
import os
from typing import Any, Union

import torch
import torch.nn as nn
from torch import Tensor

from .Consts.Types import PRESET_FLAGS


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Model, self).__init__()
        self._quant_calibrating = False
        self._quant_calibrated = False
        self._quant = False
        self._is_built = False
        self._model_flags = {}
        self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs) -> None:
        pass 

    def _set_status(self) -> None:
        if self._quant:
            self.start_quant()
        for k in self._model_flags:
            self.set_flag(k, self._model_flags[k])
            
    def build(self, *inputs, **kwargs) -> None:
        pass

    def init_params(self, *inputs, **kwargs):
        pass 

    def __call__(self, *input, **kwargs):
        if not self._is_built:
            self.build(*input)
            self._set_status()
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result
        if torch._C._get_tracing_state():                            # type: ignore
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
            if not self._is_built:
                self.init_params(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        self._is_built = True
        return result

    def set_flag(self, k: Union[PRESET_FLAGS,Any], v: Any=True) -> None:
        def set_model_flag(obj):
            if hasattr(obj, '_model_flags'):
                obj._model_flags[k] = v 
        self.apply(set_model_flag)

    def get_flag(self, k):
        return self._model_flags.get(k, None)

    def start_calibrate(self):
        def set_calibarte(obj):
            if hasattr(obj, '_quant') and obj._quant:
                obj._quant_calibrating = True
        self.apply(set_calibarte)

    def end_calibrate(self):
        def unset_calibrate(obj):
            if hasattr(obj, '_quant') and obj._quant:
                obj._quant_calibrating = False
                if hasattr(obj, '_finish_calibrate'):
                    obj._finish_calibrate()
                obj._quant_calibrated = True
        self.apply(unset_calibrate)

    def start_quant(self):
        def set_quant(obj):
            obj._quant = True
        self.apply(set_quant)

    def end_quant(self):
        def unset_quant(obj):
            obj._quant = False
        self.apply(unset_quant)

    def save_tensor(self, out: Tensor, name: str):
        if self.get_flag('save_tensor'):
            os.makedirs('./layer_dumps/', exist_ok=True)
            torch.save(out, './layer_dumps/%s.pth'%name.replace('/','_'))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        try:
            self._load_from_state_dict2(state_dict, prefix)
        except Exception as e:
            if not self.get_flag('loose_load'):
                raise e 
            else:
                pass
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _load_from_state_dict2(self, state_dict, prefix):
        # Conveinient method. Omit infrequent arguments
        pass 
