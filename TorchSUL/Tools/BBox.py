from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self


# bbox utils 
class BBoxes():
    format: Literal['xyxy','x1y1wh','xcycwh']
    conf: NDArray | None
    bbox: NDArray
    
    def __init__(self, bbox: NDArray, conf: Optional[NDArray] = None, format: Literal['xyxy','x1y1wh','xcycwh']='xyxy'):
        # TODO: Later can integrate NMS here 
        # TODO: Add thresholding, slicing
        assert format in ['xyxy','x1y1wh', 'xcycwh'], 'format must be one of ["xyxy", "x1y1wh", "xcycwh"]'
        assert (bbox.shape[1]==4) or (bbox.shape[1]==5), 'shape of each bbox must be 4 or 5'
        
        if conf is None:
            self.bbox = bbox[:, :4].copy()
            if bbox.shape[1]==4:
                self.conf = None 
            else:
                self.conf = bbox[:, 4].copy()
        else:
            self.bbox = bbox.copy()
            self.conf = conf.copy()
        self.format = format

    def get(self) -> NDArray:
        if self.conf is None:
            return self.bbox
        else:
            return np.concatenate([self.bbox, self.conf], axis=1)
    
    def get_box(self) -> NDArray:
        return self.bbox
    
    def get_conf(self) -> NDArray|None:
        return self.conf 

    def copy(self) -> Self:
        return type(self)(self.bbox, self.conf, self.format)

    def x1y1wh2xyxy(self) -> Self:
        x2 = self.bbox[:, 2] + self.bbox[:, 0]
        y2 = self.bbox[:, 3] + self.bbox[:, 1]
        self.bbox[:, 2] = x2 
        self.bbox[:, 3] = y2 
        return self

    def xyxy2x1y1wh(self) -> Self:
        w = self.bbox[:, 2] - self.bbox[:, 0]
        h = self.bbox[:, 3] - self.bbox[:, 1]
        self.bbox[:, 2] = w 
        self.bbox[:, 3] = h 
        return self 

    def x1y1wh2xcycwh(self) -> Self:
        xc = self.bbox[:, 0] + 0.5 * self.bbox[:, 2]
        yc = self.bbox[:, 1] + 0.5 * self.bbox[:, 3]
        self.bbox[:, 0] = xc 
        self.bbox[:, 1] = yc 
        return self 

    def xcycwh2x1y1wh(self) -> Self:
        x1 = self.bbox[:, 0] - 0.5 * self.bbox[:, 2]
        y1 = self.bbox[:, 1] - 0.5 * self.bbox[:, 3]
        self.bbox[:, 0] = x1 
        self.bbox[:, 1] = y1 
        return self 

    def xcycwh2xyxy(self) -> Self:
        self.xcycwh2x1y1wh()
        return self.x1y1wh2xyxy()

    def xyxy2xcycwh(self) -> Self:
        self.xyxy2x1y1wh()
        return self.x1y1wh2xcycwh()

    def convert(self, target_format: Literal['xyxy','x1y1wh','xcycwh'] = 'xyxy') -> Self:
        assert target_format in ['xyxy','x1y1wh', 'xcycwh']
        funcs = {('x1y1wh','xcycwh'): self.x1y1wh2xcycwh, 
                    ('xcycwh', 'x1y1wh'): self.xcycwh2x1y1wh,
                    ('x1y1wh', 'xyxy'): self.x1y1wh2xyxy,
                    ('xyxy', 'x1y1wh'): self.xyxy2x1y1wh,
                    ('xcycwh', 'x1y1wh'): self.xcycwh2x1y1wh,
                    ('x1y1wh', 'xcycwh'): self.x1y1wh2xcycwh}

        convert_tuple = (self.format, target_format)
        if convert_tuple in funcs:
            funcs[convert_tuple]()
        return self

    def iou(self, other: BBoxes) -> NDArray:
        box1 = self.copy().convert('xyxy').get_box()
        box2 = other.copy().convert('xyxy').get_box()
        x11, y11, x12, y12 = np.split(box1, 4, axis=1)
        x21, y21, x22, y22 = np.split(box2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou 

    def __xor__(self, other: BBoxes) -> NDArray:
        return self.iou(other)

