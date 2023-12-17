from dataclasses import dataclass
from typing import Type, Union

from ..Consts.Types import *


##### START: QuantType classes 
@dataclass 
class QTypeBase():
    max_val: int 
    min_val: int 
    signed: bool


class QUint8(QTypeBase):
    max_val = 2 ** 8 -1 
    min_val = 0 
    signed = False


class QInt8(QTypeBase):
    max_val = 2 ** 7 - 1 
    min_val = - 2 ** 7 
    signed = True 


class QInt16(QTypeBase):
    max_val = 2 ** 15 - 1 
    min_val = - 2 ** 15 
    signed = True 


QTYPES: dict[Union[QBitTypes, str], Type[QTypeBase]] = {"uint8": QUint8, "int8": QInt8, 'int16':QInt16}
##### END: QuantType classes 

