import time 
import torch
import numpy as np 
import multiprocessing as mp 

from datetime import datetime
from multiprocessing import Value
from torch.utils.tensorboard.writer import SummaryWriter


class LoggerBase(mp.Process):
    def __init__(self, loggerQ: mp.Queue):
        super().__init__()
        self.loggerQ = loggerQ
        self.should_run = Value('i', 1)
        self.lock = mp.Lock()

    def stop(self):
        print('Trying to stop logger...')
        self.lock.acquire()
        self.should_run.value = 0  # type: ignore
        self.lock.release()

    def _add_scalar(self, key, gpu_idx, n_iter, data, writer):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if isinstance(data, np.ndarray):
            data = float(data)

        writer.add_scalar(f'gpu[{gpu_idx}]/{key}', data, n_iter)
        writer.flush()

    def run(self):
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        writer = SummaryWriter(log_dir='training_log/%s/'%stamp)
        while self.should_run.value:  # type: ignore
            self.lock.acquire()
            # currently only supports scalar 
            try:
                key, gpu_idx, n_iter, data, log_type = self.loggerQ.get(timeout=1)
                if log_type=='scalar':
                    self._add_scalar(key, gpu_idx, n_iter, data, writer)
                else:
                    raise NotImplementedError('Logger only supports scalar type for now')
            except KeyboardInterrupt:
                self.should_run.value = 0   # type: ignore
                self.lock.release()
                break  
            except:
                ...
            self.lock.release()
            time.sleep(0.05)

        writer.close()

