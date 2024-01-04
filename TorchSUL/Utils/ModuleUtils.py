import copy
import inspect
from typing import Any, Dict

import torch
import torch.nn as nn
from loguru import logger

from ..Modules import ConvLayer, ConvLayer1D, ConvLayer3D, DeConvLayer, Dense
from ..Quant import QAct


def init_model(model, *args, **kwargs):
    # run one forward loop to initialize model 
    with torch.no_grad():
        return model(*args, **kwargs)


def to_standard_torch(model: nn.Module, inplace: bool=True) -> nn.Module:
    if not inplace:
        model = copy.deepcopy(model)
    if hasattr(model, 'to_torch_module'):
        model.to_torch_module()  # type: ignore
    else:
        for n,c in model.named_children():
            if hasattr(c, 'to_torch'):
                setattr(model, n.split('.')[-1], c.to_torch())  # type: ignore
            else:
                to_standard_torch(c)
    return model 


def inspect_quant_params(module: nn.Module, result_dict: Dict[str, Any]=dict(), prefix: str=''):
    if isinstance(module, QAct):
        try:
            zero_point = module.quantizer.observer.zero_point
            scale = module.quantizer.observer.scale 
            result_dict[prefix] = [scale, zero_point.round()] # type: ignore
        except:
            logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
        return result_dict
    if isinstance(module, (ConvLayer, ConvLayer1D, ConvLayer3D)):
        try:
            if hasattr(module.conv, 'input_quantizer'):
                scale = module.conv.input_quantizer.observer.scale
                zero_point = module.conv.input_quantizer.observer.zero_point
                result_dict[prefix+'/conv/Conv__input'] = [scale, zero_point.round()] # type: ignore
                scale = module.conv.w_quantizer.observer.scale
                zero_point = module.conv.w_quantizer.observer.zero_point
                result_dict[prefix+'/conv/Conv__weight'] = [scale, zero_point.round()] # type: ignore
        except:
            logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
        return result_dict
    if isinstance(module, DeConvLayer):
        try:
            if hasattr(module.conv, 'input_quantizer'):
                scale = module.conv.input_quantizer.observer.scale
                zero_point = module.conv.input_quantizer.observer.zero_point
                result_dict[prefix+'/conv/DeConv__input'] = [scale, zero_point.round()] # type: ignore
                scale = module.conv.w_quantizer.observer.scale
                zero_point = module.conv.w_quantizer.observer.zero_point
                result_dict[prefix+'/conv/DeConv__weight'] = [scale, zero_point.round()] # type: ignore
        except:
            logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
        return result_dict
    if isinstance(module, Dense):
        try:
            if hasattr(module.fc, 'input_quantizer'):
                scale = module.fc.input_quantizer.observer.scale
                zero_point = module.fc.input_quantizer.observer.zero_point
                result_dict[prefix+'/fc/Dense__input'] = [scale, zero_point.round()] # type: ignore
                scale = module.fc.w_quantizer.observer.scale
                zero_point = module.fc.w_quantizer.observer.zero_point
                result_dict[prefix+'/fc/Dense__weight'] = [scale, zero_point.round()] # type: ignore
        except:
            logger.warning(f'Quant params of layer: {prefix} cannot be properly retrieved. Maybe this layer is never called in calibration.')
        return result_dict
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        for i in range(len(module)):
            inspect_quant_params(module[i], result_dict=result_dict, prefix=prefix+'.%d'%i)
        return result_dict
    results = inspect.getmembers(module)
    for name, child_module in results:
        if isinstance(child_module, nn.Module):
            inspect_quant_params(child_module, result_dict=result_dict, prefix=prefix+'/'+name)
    return result_dict

