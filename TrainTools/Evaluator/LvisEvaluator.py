import multiprocessing as mp
import time
from multiprocessing import Value

from loguru import logger
from lvis.eval import LVISEval
from .DetectionEvaluator import EvaluatorBase


class LvisEvaluator(EvaluatorBase):
    def __init__(self, evaluatorQ: mp.Queue, n_gpus: int, annot_path: str):
        super().__init__(evaluatorQ=evaluatorQ, n_gpus=n_gpus, annot_path=annot_path)

    def _summarize(self, results):
        lvis_eval = LVISEval(self.annot_path, results, 'bbox')
        lvis_eval.evaluate()
        lvis_eval.accumulate()
        lvis_eval.summarize()
        lvis_eval.print_results()

    def _do_eval(self):
        start_signal_received = 1 
        end_signal_received = 0
        results = []
        visited = {}
        while 1:
            boxes, category, confidence, image_id, signal_type = self.evaluatorQ.get()    # type: ignore
            if signal_type=='start_eval':
                start_signal_received += 1 
            elif signal_type=='end_eval':
                end_signal_received += 1 
                logger.info(f'End signal received. Count=[{end_signal_received}]')
            elif signal_type=='data':
                if image_id not in visited:
                    self._process_data(boxes, category, confidence, image_id, results)
                visited[image_id] = 1 
            else:
                raise Exception(f'Received signal [{signal_type}] in evaluator')

            if end_signal_received == self.n_gpus:
                break 
            if start_signal_received > self.n_gpus:
                logger.warning(f'Except {self.n_gpus}, but received {start_signal_received} [start_eval] signal. Please check your code.')

        logger.info('Evaluating...')
        # torch.save(results, 'results.pth')
        self._summarize(results)
        logger.info('End evaluating')
