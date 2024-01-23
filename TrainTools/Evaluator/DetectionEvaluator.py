import multiprocessing as mp
import time
from multiprocessing import Value

from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class EvaluatorBase(mp.Process):
    def __init__(self, evaluatorQ: mp.Queue, n_gpus: int, annot_path: str):
        super().__init__()
        self.evaluatorQ = evaluatorQ
        self.should_run = Value('i', 1)
        self.lock = mp.Lock()
        self.n_gpus = n_gpus
        self.annot_path = annot_path

    def stop(self):
        print('Trying to stop evaluator...')
        self.lock.acquire()
        self.should_run.value = 0   # type: ignore
        self.lock.release()

    def _process_data(self, boxes, category, confidence, image_id, results):
        for i in range(len(boxes)):
            b = [boxes[i,0], boxes[i,1], boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1]]
            b = [float(bb) for bb in b]
            c = int(category[i])
            scr = float(confidence[i])
            buff = {'bbox': b, 'category_id': c, 'score':scr, 'image_id': image_id}
            results.append(buff)

    def _summarize(self, coco: COCO, results):
        coco_dt = coco.loadRes(results)
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.params.useSegm = None 
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def _do_eval(self):
        coco = COCO(self.annot_path)
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
        self._summarize(coco, results)
        logger.info('End evaluating')

    def run(self):
        while self.should_run.value:          # type: ignore
            self.lock.acquire()
            try:
                boxes, category, confidence, image_id, signal_type = self.evaluatorQ.get(timeout=1)
            except KeyboardInterrupt:
                self.should_run.value = 0           # type: ignore
                break 
            except:
                self.lock.release()
                continue 

            if signal_type=='start_eval':
                try:
                    logger.info('Evaluator: [start_eval] signal received, starting evaluation...')
                    self._do_eval()
                except KeyboardInterrupt:
                    self.should_run.value = 0      # type: ignore
                except Exception as e:
                    print(e)
                    print()
                    logger.warning('Evaluation failed.')
                except:
                    ...

            self.lock.release()
            time.sleep(0.05)

