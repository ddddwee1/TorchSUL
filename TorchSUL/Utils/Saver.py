import os
from distutils.version import LooseVersion

import torch
import torch.nn as nn
from loguru import logger

from ..Base import Model


class Saver():
    model: Model 

    def __init__(self, module: Model):
        self.model = module

    def _get_checkpoint(self, path: str) -> str:
        ckpt = os.path.join(path, 'checkpoint')
        if os.path.exists(ckpt):
            fname = open(ckpt).readline().strip()
            return os.path.join(path, fname)
        else:
            return ''

    def restore(self, path: str, strict: bool=True):
        logger.info('Trying to load from: %s'%path)
        device = torch.device('cpu')
        if path[-4:] == '.pth':
            if not os.path.exists(path):
                logger.warning('Path: %s does not exsist. No restoration will be performed.'%path)
            elif isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                state_dict = torch.load(path, map_location=device)
                self.model.module.load_state_dict(state_dict, strict=strict)
                logger.info('Model loaded from: %s'%path)
            else:
                state_dict = torch.load(path, map_location=device)
                self.model.load_state_dict(state_dict, strict=strict)
                logger.info('Model loaded from: %s'%path)
        else:
            path = self._get_checkpoint(path)
            if path:
                if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                    state_dict = torch.load(path, map_location=device)
                    self.model.module.load_state_dict(state_dict, strict=strict)
                else:
                    state_dict = torch.load(path, map_location=device)
                    self.model.load_state_dict(state_dict, strict=strict)
                logger.info('Model loaded from: %s'%path)
            else:
                logger.warning('No checkpoint found. No restoration will be performed.')

    def save(self, path: str):
        # To make it compatible with older pytorch 
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            if LooseVersion(torch.__version__)>=LooseVersion('1.6.0'):
                torch.save(self.model.module.state_dict(), path, _use_new_zipfile_serialization=False)
            else:
                torch.save(self.model.module.state_dict(), path)
        else:
            if LooseVersion(torch.__version__)>=LooseVersion('1.6.0'):
                torch.save(self.model.state_dict(), path, _use_new_zipfile_serialization=False)
            else:
                torch.save(self.model.state_dict(), path)
        logger.info('Model saved to: %s'%path)
        ckpt = open(directory + '/checkpoint', 'w')
        ckpt.write(os.path.basename(path))
        ckpt.close()

