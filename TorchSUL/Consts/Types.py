from typing import Literal, Union

## Layer param type declaration
PadModes = Literal['SAME_LEFT', 'VALID']
TypeKSize = Union[int, list[int], tuple[int,...]]
TypeKSize2D = Union[int, list[int], tuple[int,int]]
TypeKSize3D = Union[int, list[int], tuple[int,int,int]]

## Quant type declarations
QuantModes = Literal['layer_wise', 'channel_wise']
QBitTypes = Literal['uint8', 'int8', 'int16']
QObserverTypes = Literal['minmax', 'percentile', 'omse']
QuantizerTypes = Literal['uniform']

## Image-video related 
BasicImageTypes = Literal['jpg', 'png', 'jpeg']

# These are pre-set flags for SULModel 
PRESET_FLAGS = Literal['save_tensor', 'loose_load', 'from_torch', 'fc2conv', 'QActBit', 'QActObserver',\
                        'conv_init_mode', 'fc_init_mode', 'bn_eps', 'ln_eps', 'dump_onnx']
