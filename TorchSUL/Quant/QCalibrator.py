import torch 
import torch.nn.functional as F 
from ..Base import Model 
from typing import List 
from .Quantizers import QuantizerBase
from .Observers import PlaceholderObserver
from torch import Tensor 
from collections.abc import Callable
import itertools 
from loguru import logger 


def l2_score(a, b):
    return -torch.pow(a - b, 2).mean()


def cosine_score(a, b):
    a = a.flatten(1)
    b = b.flatten(1)
    dist = F.cosine_similarity(a, b, dim=1)
    return dist.mean()


class LayerCalibrator(Model):
    def initialize(self, quantizers: List[QuantizerBase], fwd_fn: Callable[..., Tensor]):
        self.observers = []
        for q in quantizers:
            self.observers.append(q.observer)
        for obs in self.observers:
            assert isinstance(obs, PlaceholderObserver), 'Observer for LayerCalibrator must be PlaceholderObserver'
        self.inputs = []
        self.outputs = []
        self.fwd_fn = fwd_fn

    def layer_hook(self, module, inputs, outputs):
        if self._quant and self._quant_calibrating:
            self.inputs.append(inputs)
            self.outputs.append(outputs)

    @torch.no_grad()
    def _finish_calibrate(self):
        scr_func_flag = self.get_flag('CalibScoreFunc')
        sim_func = cosine_score if scr_func_flag=='cosine' else l2_score
        logger.debug(f'LayerCalibrator post processing... Score function: {"L2" if scr_func_flag is None else scr_func_flag}')
        # combine inputs 
        logger.debug(f'Num inputs to calibrate: {len(self.inputs)}')
        inputs = list(zip(*self.inputs))
        inputs = [torch.cat(i, dim=0) for i in inputs]
        outputs = torch.cat(self.outputs, dim=0)

        # initialize observers
        for obs in self.observers:
            obs.init_quant_params()

        # phase 1: generate coarse grid to search best scales  
        scales_to_test = [[s*0.05+0.2 for s in range(17)],] * len(self.observers)

        best_scales = None 
        best_sim = None 

        for scales in itertools.product(*scales_to_test):
            for s,obs in zip(scales, self.observers):
                obs.resize_scale(s)
            quant_output = self.fwd_fn(*inputs)
            similarity = sim_func(quant_output, outputs)
            if best_sim is None or similarity>best_sim:
                best_scales = scales
                best_sim = similarity

        # phase 2: generate fine grid to search best scales 
        scales_to_test: List[List[float]] = [[s-0.05+0.002*i for i in range(51)] for s in best_scales]  # type: ignore
        
        best_scales = None 
        best_sim = None
        for scales in itertools.product(*scales_to_test):
            for s,obs in zip(scales, self.observers):
                obs.resize_scale(s)
            quant_output = self.fwd_fn(*inputs)
            similarity = sim_func(quant_output, outputs)
            if best_sim is None or similarity>best_sim:
                best_scales = scales
                best_sim = similarity

        for s,obs in zip(best_scales, self.observers):  # type: ignore
            obs.resize_scale(s)
        logger.debug(f'Find best scale: {best_scales}. Similarity: {best_sim.cpu().numpy()}')  # type: ignore

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        ...

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        ...

