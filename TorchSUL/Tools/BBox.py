import torch 
import torchvision.ops as ops

from typing import Optional
from typing_extensions import Literal
from torch import Tensor 


# bbox utils 
class BBoxes():
    bbox: Tensor 
    conf: Optional[Tensor]
    box_format: Literal['xyxy', 'x1y1wh', 'xcycwh']

    def __init__(self, bbox: Tensor, box_format: Literal['xyxy', 'x1y1wh', 'xcycwh'], conf: Optional[Tensor]=None): 
        # dont use inplace
        assert (bbox.shape[-1] == 4) or (bbox.shape[-1] == 5)
        assert len(bbox.shape)==2
        self.bbox = bbox[:, :4].clone()
        if conf is None:
            if bbox.shape[-1] == 4:
                self.conf = None 
            else:
                self.conf = bbox[:, 4].clone()
        else:
            self.conf = conf.clone()
        self.box_format = box_format

    def copy(self):
        return type(self)(self.bbox, self.box_format, self.conf)

    def x1y1wh2xyxy(self) -> 'BBoxes':
        x2 = self.bbox[:, 0] + self.bbox[:, 2]
        y2 = self.bbox[:, 1] + self.bbox[:, 3]
        new_bbox = torch.stack([self.bbox[:, 0], self.bbox[:, 1], x2, y2], dim=-1)
        return type(self)(new_bbox, 'xyxy', self.conf)
    
    def xyxy2x1y1wh(self) -> 'BBoxes':
        w = self.bbox[:, 2] - self.bbox[:, 0]
        h = self.bbox[:, 3] - self.bbox[:, 1]
        new_bbox = torch.stack([self.bbox[:, 0], self.bbox[:, 1], w, h], dim=-1)
        return type(self)(new_bbox, 'x1y1wh', self.conf)
    
    def xyxy2xcycwh(self) -> 'BBoxes':
        xc = 0.5 * (self.bbox[:, 0] + self.bbox[:, 2])
        yc = 0.5 * (self.bbox[:, 1] + self.bbox[:, 3])
        w = self.bbox[:, 2] - self.bbox[:, 0]
        h = self.bbox[:, 3] - self.bbox[:, 1]
        new_bbox = torch.stack([xc,yc,w,h], dim=-1)
        return type(self)(new_bbox, 'xcycwh', self.conf)
    
    def xcycwh2xyxy(self) -> 'BBoxes':
        x1 = self.bbox[:, 0] - 0.5 * self.bbox[:, 2]
        y1 = self.bbox[:, 1] - 0.5 * self.bbox[:, 3]
        x2 = self.bbox[:, 0] + 0.5 * self.bbox[:, 2]
        y2 = self.bbox[:, 1] + 0.5 * self.bbox[:, 3]
        new_bbox = torch.stack([x1, y1, x2, y2], dim=-1)
        return type(self)(new_bbox, 'xyxy', self.conf)
    
    def x1y1wh2xcycwh(self) -> 'BBoxes':
        xc = self.bbox[:, 0] + 0.5 * self.bbox[:, 2]
        yc = self.bbox[:, 1] + 0.5 * self.bbox[:, 3]
        new_bbox = torch.stack([xc, yc, self.bbox[:, 2], self.bbox[:, 3]], dim=-1)
        return type(self)(new_bbox, 'xcycwh', self.conf)
    
    def xcycwh2x1y1wh(self) -> 'BBoxes':
        x1 = self.bbox[:, 0] - 0.5 * self.bbox[:, 2]
        y1 = self.bbox[:, 1] - 0.5 * self.bbox[:, 3] 
        new_bbox = torch.stack([x1, y1, self.bbox[:, 2], self.bbox[:, 3]], dim=-1)
        return type(self)(new_bbox, 'x1y1wh', self.conf)
    
    def convert(self, target_format: Literal['xyxy', 'x1y1wh', 'xcycwh']) -> 'BBoxes':
        funcs = {('x1y1wh', 'xcycwh'): self.x1y1wh2xcycwh, 
                 ('x1y1wh', 'xyxy'): self.x1y1wh2xyxy,
                 ('xcycwh', 'x1y1wh'): self.xcycwh2x1y1wh,
                 ('xcycwh', 'xyxy'): self.xcycwh2xyxy,
                 ('xyxy', 'x1y1wh'): self.xyxy2x1y1wh,
                 ('xyxy', 'xcycwh'): self.xyxy2xcycwh}
        
        convert_tuple = (self.box_format, target_format)
        if convert_tuple in funcs:
            new_obj = funcs[convert_tuple]()
        else:
            new_obj = self.copy()
        return new_obj

    def iou(self, other: 'BBoxes') -> Tensor:
        b1 = self.convert('xyxy')
        b2 = other.convert('xyxy')
        return ops.box_iou(b1.bbox, b2.bbox)

    def __xor__(self, other: 'BBoxes') -> Tensor:
        return self.iou(other)

    def distance(self, other: 'BBoxes') -> Tensor:
        b1 = self.convert('xcycwh')
        b2 = other.convert('xcycwh')
        return torch.cdist(b1.bbox[:, :2], b2.bbox[:, :2])

    def __or__(self, other: 'BBoxes') -> Tensor:
        return self.distance(other)

    def cat(self, other: 'BBoxes') -> 'BBoxes': 
        assert self.box_format==other.box_format
        new_box = torch.cat([self.bbox, other.bbox], dim=0)
        if (self.conf is not None) and (other.conf is not None):
            new_conf = torch.cat([self.conf, other.conf], dim=0)
        else:
            new_conf = None 
        return BBoxes(new_box, self.box_format, new_conf)

    def __matmul__(self, other: 'BBoxes') -> 'BBoxes':
        return self.cat(other)

    def __getitem__(self, idx: Tensor) -> 'BBoxes':
        new_box = self.bbox[idx]
        if self.conf is not None:
            new_conf = self.conf[idx]
        else:
            new_conf = None 
        return BBoxes(new_box, self.box_format, new_conf)

    def __len__(self):
        return len(self.bbox)

    def inside(self, other: 'BBoxes') -> Tensor:
        assert len(self)==len(other), f'Length should be the same for bbox __contain__, first: {len(other)}, second: {len(self)}'
        this_box = self.convert('xcycwh').bbox
        other_box = other.convert('xyxy').bbox
        xs = this_box[:, 0]
        ys = this_box[:, 1]
        result = ((xs > other_box[:, 0]) & (xs < other_box[:, 2]) & (ys > other_box[:, 1]) & (ys < other_box[:, 3]))
        return result
