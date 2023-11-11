from __future__ import annotations

import json 
import torch 
import numpy as np 
from numpy.typing import NDArray
from TorchSUL import Config


# I will try type hinting. Let's see if this will make life easier?

class PtsEncoder():
    ratios: list[float]
    strides: list[int]
    scale: float 
    base_whs: list[list[list[float]]]
    H: int 
    W: int 
    data: dict[str, list[list[float]]]
    fmap_size: list[list[int]]

    def __init__(self, cfg: Config.ConfigDict) -> None:
        self.H = cfg.DATA.H
        self.W = cfg.DATA.W 
        self.ratios = cfg.ANCHOR.RATIOS
        self.strides = cfg.ANCHOR.STRIDES 
        self.scale = cfg.ANCHOR.SCALE
        self.fmap_size = cfg.MODEL.FMAP_SIZE
        self.base_whs = []
        for st in self.strides:
            buff: list[list[float]] = []
            for r in self.ratios:
                w = st * self.scale / np.sqrt(r)
                h = st * self.scale * np.sqrt(r)
                buff.append([w,h])
            self.base_whs.append(buff)

    def parse_idx(self, box: list[float]) -> tuple[int, int, int, int]:
        def simple_iou(w1:float,h1:float,w2:float,h2:float) -> float:
            inter: float = min(w1,w2) * min(h1,h2)
            return inter / (w1*h1 + w2*h2)
        # box is in xywh format 
        best_iou: float = 0.0
        best_anchor: int = 0
        best_scale: int = 0
        for sc, bwh in enumerate(self.base_whs):
            for an, base_wh in enumerate(bwh):
                iou = simple_iou(base_wh[0], base_wh[1], box[0], box[1])
                if iou>best_iou:
                    best_iou = iou 
                    best_anchor = an 
                    best_scale = sc 
        if best_iou <= 0.0:
            return -1, -1, -1, -1
        else:
            x, y = box[:2]
            st = self.strides[best_scale]
            y_idx = int(round((y - st/2) / st))
            x_idx = int(round((x - st/2) / st))
            H, W = self.fmap_size[best_scale]
            if (x_idx<0) or (x_idx>=W) or (y_idx<0) or (y_idx>=H):
                return -1, -1, -1, -1
            return best_scale, best_anchor, y_idx, x_idx 

    def encode_box(self, box: list[float], scale_idx: int, anchor_idx: int, y_idx: int, x_idx: int) -> tuple[float, float, float, float]:
        x, y, w, h = box
        st = self.strides[scale_idx]
        base_w, base_h = self.base_whs[scale_idx][anchor_idx]
        base_x = x_idx * st 
        base_y = y_idx * st 
        dx = (x - base_x) / st
        dy = (y - base_y) / st 
        dw = np.log(w / base_w)
        dh = np.log(h / base_h)
        return dx,dy,dw,dh

    def parse_box(self, box: list[float], box_labels: list[NDArray], conf_labels: list[NDArray], mask: list[NDArray]) -> None:
        scale_idx, anchor_idx, y_idx, x_idx = self.parse_idx(box)
        if scale_idx==-1:
            return 
        dx,dy,dw,dh = self.encode_box(box, scale_idx, anchor_idx, y_idx, x_idx)
        conf_labels[scale_idx][anchor_idx, y_idx, x_idx] = 1 
        box_labels[scale_idx][anchor_idx*4: anchor_idx*4+4, y_idx, x_idx] = [dx,dy,dw,dh]
        mask[scale_idx][anchor_idx*4: anchor_idx:4+4, y_idx, x_idx] = 1 

    def generate_empty_labels(self) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
        boxes: list[NDArray] = []
        confs: list[NDArray] = []
        masks: list[NDArray] = []
        n_anchor = len(self.ratios)
        for h,w in self.fmap_size:
            b: NDArray = np.zeros([n_anchor*4, h, w], dtype=np.float32)
            c: NDArray = np.zeros([n_anchor, h, w], dtype=np.float32)
            m: NDArray = np.zeros([n_anchor*4, h, w], dtype=np.float32)
            boxes.append(b)
            confs.append(c)
            masks.append(m)
        return boxes, confs, masks
    
    def encode(self, boxes: list[list[float]]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        box_labels, conf_labels, mask_labels = self.generate_empty_labels()
        for box in boxes:
            self.parse_box(box,box_labels, conf_labels, mask_labels)
        box_labels = [torch.from_numpy(b) for b in box_labels]
        conf_labels = [torch.from_numpy(c) for c in conf_labels]
        mask_labels = [torch.from_numpy(m) for m in mask_labels]
        return box_labels, conf_labels, mask_labels


class PtsDecoder():
    ratios: list[float]
    strides: list[int]
    scale: float 
    base_whs: list[list[list[float]]]
    H: int 
    W: int 
    data: dict[str, list[list[float]]]
    fmap_size: list[list[int]]

    def __init__(self, cfg: Config.ConfigDict) -> None:
        self.H = cfg.DATA.H
        self.W = cfg.DATA.W 
        self.ratios = cfg.ANCHOR.RATIOS
        self.strides = cfg.ANCHOR.STRIDES 
        self.scale = cfg.ANCHOR.SCALE
        self.fmap_size = cfg.MODEL.FMAP_SIZE
        self.base_whs = []
        for st in self.strides:
            buff: list[list[float]] = []
            for r in self.ratios:
                w = st * self.scale / np.sqrt(r)
                h = st * self.scale * np.sqrt(r)
                buff.append([w,h])
            self.base_whs.append(buff)
    
    def decode(self, confs: list[torch.Tensor], boxes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        results_list: list[torch.Tensor] = []
        b_idx_all_list: list[torch.Tensor] = []
        for sc, conf in enumerate(confs):
            st = self.strides[sc]
            b_idx, a_idx, y_idx, x_idx = torch.where(conf>0)  # b, a, y, x
            xc = x_idx * st 
            yc = y_idx * st 
            bwh = torch.tensor(self.base_whs[sc])[a_idx]
            base_xywh = torch.cat([torch.stack([xc,yc], dim=-1), bwh], dim=-1)  # [N,4]
            box = boxes[sc].unflatten(1, (-1, 4))
            box = box[b_idx, a_idx, :, y_idx, x_idx]    # [N,4]
            res = torch.cat([base_xywh[:,:2]+box[:,:2]*st, base_xywh[:,2:]*torch.exp(box[:,2:])], dim=-1)  # [N,4]
            b_idx_all_list.append(b_idx)
            results_list.append(res)
        results = torch.cat(results_list, dim=0)
        b_idx_all = torch.cat(b_idx_all_list, dim=0)
        # TODO: add batched nms here
        return results, b_idx_all

if __name__=='__main__':
    cfg = Config.load_yaml('config.yaml')
    encoder = PtsEncoder(cfg)
    decoder = PtsDecoder(cfg)

    data = json.load(open(cfg.DATA.PATH))
    boxes: list[list[float]] = data['%08d'%1]
    scale = 0.35
    boxes = [[i*scale for i in j] for j in boxes]
    print(boxes)
    
    box_labels, conf_labels, mask_labels = encoder.encode(boxes)
    box_labels = [b.unsqueeze(0) for b in box_labels]
    conf_labels = [c.unsqueeze(0) for c in conf_labels]
    mask_labels = [m.unsqueeze(0) for m in mask_labels]
    results, b_idx_all = decoder.decode(conf_labels, box_labels)
    print(results)
