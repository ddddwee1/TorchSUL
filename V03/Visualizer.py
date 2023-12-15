from __future__ import annotations

import cv2 
import numpy as np 
import multiprocessing as mp 
from abc import ABC, abstractmethod
from .Tool import progress_bar

from typing import Any, Type


class VisualizerProc(mp.Process, ABC):
    def __init__(self, imgListQ: mp.Queue, progressQ: mp.Queue):
        super().__init__()
        self.imgListQ = imgListQ
        self.progressQ = progressQ
        self.initialize()
    
    def initialize(self):
        ...
    
    def run(self):
        while self.imgListQ.qsize() > 0:
            data_bundle = self.imgListQ.get()
            self.process_img(*data_bundle)
            self.progressQ.put(-1)

    @abstractmethod
    def process_img(self, *data_bundle):
        ...


class ProgressProc(mp.Process):
    def __init__(self, progressQ: mp.Queue):
        super().__init__()
        self.prog = progress_bar()
        self.progressQ = progressQ

    def run(self):
        self.prog.start()
        total = self.progressQ.get()
        task = self.prog.add_task('Visualizing', total=total)
        cnt = 0
        while 1:
            self.progressQ.get()
            self.prog.advance(task)
            cnt += 1 
            if cnt == total:
                break 
        self.prog.stop()


def visualize(data_bundle: list[Any], TVisProc: Type[VisualizerProc], n_procs=4):
    progressQ = mp.Queue()
    imageListQ = mp.Queue()

    for d in data_bundle:
        imageListQ.put(d)

    progressQ.put(len(data_bundle))

    progress_proc = ProgressProc(progressQ)
    progress_proc.start()

    procs = []
    for _ in range(n_procs):
        proc = TVisProc(imageListQ, progressQ)
        proc.start()
        procs.append(proc)
    
    for p in procs:
        p.join()

