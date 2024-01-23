import cv2 
import random 
import numpy as np 
from numpy.typing import NDArray

from typing import Union, Optional


# type definition 
ScaleType2D =  Union[tuple[int,int], list[int]]
FloatArray = NDArray[np.float32]
MatArray = NDArray[np.uint8]


class ResizePad():
    def __init__(self, target_size: ScaleType2D):
        self.target_size = target_size
    
    def __call__(self, img: MatArray) -> tuple[MatArray, float]:
        h, w = img.shape[:2]
        scale = min(self.target_size[0]/h, self.target_size[1]/w)
        canvas = np.zeros([self.target_size[0], self.target_size[1], 3], dtype=np.uint8)
        img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h2, w2 = img_resized.shape[:2]
        canvas[:h2, :w2, :] = img_resized
        return canvas, scale


class RandomResizeRange():
    def __init__(self, scale_low: ScaleType2D, scale_high: ScaleType2D):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, img: MatArray, label: Optional[FloatArray]=None, mask:Optional[MatArray]=None
                 ) -> tuple[MatArray, Optional[FloatArray], Optional[MatArray]]:
        scale_h = random.random() * (self.scale_high[0] - self.scale_low[0]) + self.scale_low[0]
        scale_w = random.random() * (self.scale_high[1] - self.scale_low[1]) + self.scale_low[1]
        h,w = img.shape[:2]
        scale = min(scale_h / h, scale_w / w)
        h_2, w_2 = int(scale*h), int(scale*w)
        img_resized = cv2.resize(img, (w_2, h_2), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        # process label 
        if label is not None:
            scale_h = h_2 / h
            scale_w = w_2 / w 
            label_out = label.copy()
            label_out[:, 0] *= scale_w
            label_out[:, 1] *= scale_h
            label_out[:, 2] *= scale_w 
            label_out[:, 3] *= scale_h
        else:
            label_out = None 
        # process mask 
        if mask is not None:
            mask_out = cv2.resize(mask, (w_2, h_2), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        else:
            mask_out = None  
        return img_resized, label_out, mask_out
    

class Padding():
    def __init__(self, divisor:int):
        self.divisor = divisor
    
    def __call__(self, img: MatArray, mask: Optional[MatArray]=None) -> tuple[MatArray, Optional[MatArray]]:
        h,w = img.shape[:2]
        d = self.divisor
        h_pad = (d - (h % d)) % d 
        w_pad = (d - (w % d)) % d 
        canvas = np.zeros([h+h_pad, w+w_pad,3], dtype=np.uint8)
        canvas[:h, :w] = img 
        if mask is not None:
            mask_canvas = np.zeros([h+h_pad, w+w_pad,3], dtype=np.uint8)
            mask_canvas[:h, :w] = mask 
        else:
            mask_canvas = None 
        return canvas, mask_canvas


class ImageNetNormalize():
    def __init__(self, mean: list[float]=[0.485, 0.456, 0.406], std: list[float]=[0.229 , 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32) 
        self.std = np.array(std, dtype=np.float32) 

    def __call__(self, img: MatArray, mask: Optional[MatArray]=None) -> tuple[FloatArray, Optional[FloatArray]]:
        img_processed = img.astype(np.float32)
        img_processed = img_processed / 255.0
        img_processed = (img_processed - self.mean) / self.std 
        if mask is not None:
            mask_out = mask.astype(np.float32)
        else:
            mask_out = None
        return img_processed, mask_out


class BGR2RGB():
    def __call__(self, img: MatArray) -> MatArray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    

class RandomFlip():
    def __init__(self, prob: float=0.5):
        self.prob = prob 

    def __call__(self, x: MatArray, label: Optional[FloatArray], mask: Optional[MatArray]=None
                 ) -> tuple[MatArray, Optional[FloatArray], Optional[MatArray]]: 
        if random.random() < self.prob: 
            x = np.flip(x, axis=1)
            w = x.shape[1]
            # process label 
            if label is not None:
                label_out = label.copy()
                label_out[:, 0] = w - label_out[:, 0]
                label_out[:, 2] = w - label_out[:, 2]
                label_out = label_out[:, [2,1,0,3]]
            else:
                label_out = None 
            # process mask 
            if mask is not None:
                mask_out = np.flip(mask, axis=1)
            else:
                mask_out = None 
        else:
            label_out = label
            mask_out = mask 
        return x, label_out, mask_out



