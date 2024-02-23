import os 
import cv2 
import json 
import torch 
import numpy as np 

from numpy.typing import NDArray
from typing import Optional, TypedDict
from TorchSUL.Tools import Path
from loguru import logger 
from torch.utils.data import Dataset


# 包内dataset就写在这，其余的逻辑在外部继承/包含

# type definition 
BoxInfo = TypedDict('BoxInfo', {'category_id': int, 'bbox': list[float]})
MatArray = NDArray[np.uint8]


class CocoDataset(Dataset):
    def __init__(self, annot_path: str, root_path: str='./', buffer_path: Optional[str]=None):
        self.root_path = root_path
        # buffer path to save processed data, save data loading time 
        if buffer_path is not None:
            buffer = Path(buffer_path)
            if (not buffer.exists()) or (not (buffer + 'dataset.pth').exists()):
                buffer.makedirs()
                self.image_map, self.label_info = self._process_label(annot_path)
                self.image_id_list = list(self.image_map.keys())
                logger.info(f'Saving to buffer [{buffer_path}]')
                torch.save([self.image_map, self.label_info, self.image_id_list], str(buffer + 'dataset.pth'), _use_new_zipfile_serialization=False)
                logger.info('Saved.')
            else:
                logger.info(f'Loading parsed dataset metainfo from [{buffer_path}]...')
                self.image_map, self.label_info, self.image_id_list = torch.load(str(buffer + 'dataset.pth'))
                logger.info('Loaded.')
        else:
            self.image_map, self.label_info = self._process_label(annot_path)
            self.image_id_list = list(self.image_map.keys())

    def _process_label(self, annot_path: str) -> tuple[dict[int, str], dict[int, list[BoxInfo]]]:
        logger.info(f'Parsing dataset metainfo from [{annot_path}]...')
        data = json.load(open(annot_path))

        image_info = data['images']
        image_map: dict[int, str] = {}
        label_info: dict[int, list[BoxInfo]] = {}

        for i in image_info:
            idd: int = i['id']
            if 'file_name' in i:
                name: str = i['file_name'].split('/')[-1]
            else:
                name: str = i['coco_url'].split('/')[-1]
            image_map[idd] = name
            label_info[idd] = []

        annots = data['annotations']
        for annot in annots:
            imgid: int = annot['image_id']
            if not imgid in label_info:
                label_info[imgid] = []
            buff: BoxInfo = {'category_id': annot['category_id'], 'bbox': annot['bbox']}
            label_info[imgid].append(buff)
        logger.info('Parsed.')
        return image_map, label_info

    def __len__(self):
        return len(self.image_map)

    def __getitem__(self, idx: int) -> tuple[MatArray, list[BoxInfo], int]:
        image_id = self.image_id_list[idx]
        image_path = self.image_map[image_id]
        img = cv2.imread(os.path.join(self.root_path, image_path)).astype(np.uint8)
        annot = self.label_info[image_id]
        return img, annot, image_id



