import time 
import torch.distributed as dist 
import torch.multiprocessing as mp 

from abc import ABC, abstractmethod
from loguru import logger 
from typing import Optional, Type, Any
from numpy.typing import NDArray

from ..Evaluator import EvaluatorBase
from ..Logger import LoggerBase


class Runner(mp.Process, ABC):
    def __init__(self, local_rank: int, n_gpus: int, gpu_idx: Optional[int]=None, host: str='tcp://localhost', port: int=23456,  
                 evaluatorQ: Optional[mp.Queue]=None, loggerQ: Optional[mp.Queue]=None):
        super().__init__()
        # gpu_idx: specify the gpu to be used
        self.local_rank = local_rank
        self.host_addr = host + ':' + str(port)
        self.gpu_idx = local_rank if gpu_idx is None else gpu_idx
        self.n_gpus = n_gpus
        self.evaluatorQ = evaluatorQ
        self.loggerQ = loggerQ

    def _init_process_group(self):
        dist.init_process_group(backend='nccl', init_method=self.host_addr, world_size=self.n_gpus, rank=self.local_rank)
        logger.info(f'Initialized. Local rank: [{self.local_rank}]  Host: [{self.host_addr}]')

    def run(self):
        if self.n_gpus>1:
            self._init_process_group()
        else:
            logger.info('Use single GPU')
        self.procedure()

    @abstractmethod
    def procedure(self):
        ...

    def put_log_scalar(self, key: str, data: Any, n_iter: int, gpu_idx: int):
        # key, gpu_idx, n_iter, data, log_type
        if self.loggerQ is not None:
            self.loggerQ.put([key, gpu_idx, n_iter, data, 'scalar'])

    def put_eval_results(self, boxes: NDArray, category: NDArray, confidence: NDArray, image_id: int):
        # boxes, category, confidence, image_id, signal_type
        if self.evaluatorQ is not None:
            self.evaluatorQ.put([boxes, category, confidence, image_id, 'data'])

    def start_evaluation(self):
        if self.evaluatorQ is not None:
            self.evaluatorQ.put([None, None, None, None, 'start_eval'])

    def end_evaluation(self):
        if self.evaluatorQ is not None:
            self.evaluatorQ.put([None, None, None, None, 'end_eval'])


def start_training(runner: Type[Runner], n_gpus: int, device_ids: Optional[list[int]]=None,
                   evaluator: Optional[Type[EvaluatorBase]]=None, annot_path: Optional[str]=None,
                   logger: Optional[Type[LoggerBase]]=None,
                   host: str='tcp://localhost', port: int=23456):
    if device_ids is None:
        device_ids = list(range(n_gpus))
    else:
        assert len(device_ids)==n_gpus, f'Number of device ids must be equal to n_gpus, but got {len(device_ids)} and {n_gpus}'

    assert (evaluator is None)==(annot_path is None), 'annot_path should be given if the is not None' 

    if (evaluator is not None) and (annot_path is not None):
        evaluatorQ = mp.Queue()
        eval_proc = evaluator(evaluatorQ, n_gpus, annot_path)
        eval_proc.start()
    else:
        evaluatorQ = None 
        eval_proc = None 
    
    if logger is not None:
        loggerQ = mp.Queue()
        logger_proc = logger(loggerQ)
        logger_proc.start()
    else:
        loggerQ = None 
        logger_proc = None 

    train_procs: list[Runner] = []
    for i in range(n_gpus):
        train_proc = runner(i, n_gpus=n_gpus, gpu_idx=device_ids[i], host=host, port=port, evaluatorQ=evaluatorQ, loggerQ=loggerQ)
        train_proc.start()
        train_procs.append(train_proc)

    if eval_proc is None:
        while 1:
            is_running = True
            for p in train_procs:
                is_running = is_running * p.is_alive()
            if not is_running:
                for p in train_procs:
                    p.terminate()
                break
            time.sleep(0.5)
    else:
        for p in train_procs:
            p.join()

    if logger_proc is not None:
        logger_proc.stop()
    if eval_proc is not None:
        eval_proc.stop()

