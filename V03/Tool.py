from __future__ import annotations 

import cv2
import os 
import glob
from loguru import logger
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, ProgressColumn
from rich.text import Text
from numpy.typing import NDArray
from typing import Optional, Literal, TypeVar
from typing_extensions import Self
import numpy as np 

# video utils 
class video_saver():
    def __init__(self,name,size, frame_rate=15.0):
        self.name = name
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.vidwriter = cv2.VideoWriter(name,fourcc,frame_rate,(size[1],size[0]))
    def write(self,img):
        self.vidwriter.write(img)
    def finish(self):
        self.vidwriter.release()


def check_frame_num(fname):
    video = cv2.VideoCapture(fname)
    framenum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return framenum


def check_fps(fname):
    video = cv2.VideoCapture(fname);
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps 


def combine_audio(vidname, audname, outname, fps=25):
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)


def extract_frames(fname, output_dir, ext='jpg', skip=1, frame_format='frame_%08d', return_images=False):
    def make_iterable(cap):
        while 1:
            ret, frame = cap.read()
            if ret:
                yield ret, frame
            else:
                return ret, frame
    
    assert isinstance(output_dir, str), 'output_dir must be string'
    assert isinstance(fname, str), 'file name must be string'
    assert ext.lower() in ['jpg', 'jpeg', 'png'], "extension must be one of ['jpg', 'jpeg', 'png']"

    logger.info(f'Extracting {fname}\tOutput_dir:{output_dir}\tFormat:{frame_format}\tExtension:{ext}\tSkip:{skip}')
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(fname)
    framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cnt = 0
    images = []
    with progress_bar() as prog:
        task = prog.add_task(description='Extracting...', total=framenum)
        while 1:
            ret, frame = cap.read()
            if not ret:
                break
            if cnt%skip==0:
                if return_images:
                    images.append(frame)
                else:
                    imgname = os.path.join(output_dir, frame_format%cnt + '.' + ext)
                    cv2.imwrite(imgname, frame)
            cnt += 1 
            prog.advance(task)

    logger.info('Extraction finished.')
    if return_images:
        return images


# progressbar utils 
class SpeedColumn(ProgressColumn):
    def render(self, task):
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        else:
            return Text(f"{speed:.2f} it/s", style="progress.data.speed")

def progress_bar(width=40):
    prog = Progress(TextColumn('[progress.description]{task.description}'), BarColumn(finished_style='green', bar_width=width), MofNCompleteColumn(), TimeRemainingColumn(elapsed_when_finished=True), SpeedColumn())
    return prog


# path utils 
class Path():
    path: str
    def __init__(self, path: str) -> None:
        self.path = path 

    def __add__(self, p2: Path | str) -> Path:
        if isinstance(p2, str):
            p2 = os.path.join(self.path, p2)
            return Path(p2)
        elif isinstance(p2, Path):
            p2 = os.path.join(self.path, p2.path)
            return Path(p2)

    def auto(self) -> Path:
        p2 = glob.glob(os.path.join(self.path, '*'))[0]
        return Path(p2)

    def find(self, prefix: str) -> Path:
        files = glob.glob(os.path.join(self.path, '*'))
        for f in files:
            if prefix in f.split('/')[-1]:
                return Path(f)
        raise FileNotFoundError(f'Cannot find [{prefix}] under path [{self.path}]')

    def __repr__(self) -> str:
        return self.path 

    def tostr(self) -> str:
        return self.path 

# bbox utils 
class BBoxes():
    format: Literal['xyxy','x1y1wh','xcycwh']
    conf: NDArray | None
    bbox: NDArray
    def __init__(self, bbox: NDArray, conf: Optional[NDArray] = None, format: Literal['xyxy','x1y1wh','xcycwh']='xyxy'):
        assert format in ['xyxy','x1y1wh', 'xcycwh']
        # record conf info. Later can integrate NMS here 
        assert bbox.shape[1]>=4, 'shape of bbox must >= 4'
        if conf is None:
            self.bbox = bbox[:, :4].copy()
            if bbox.shape[1]==4:
                self.conf = None 
            else:
                self.conf = bbox[:, 4].copy()
        else:
            self.bbox = bbox.copy()
            self.conf = conf.copy()
        self.format = format

    def get(self) -> NDArray:
        if self.conf is None:
            return self.bbox
        else:
            return np.concatenate([self.bbox, self.conf], axis=1)
    
    def get_box(self) -> NDArray:
        return self.bbox
    
    def get_conf(self) -> NDArray|None:
        return self.conf 

    # @classmethod
    def copy(self) -> Self:
        return type(self)(self.bbox, self.conf, self.format)

    def x1y1wh2xyxy(self) -> Self:
        x2 = self.bbox[:, 2] + self.bbox[:, 0]
        y2 = self.bbox[:, 3] + self.bbox[:, 1]
        self.bbox[:, 2] = x2 
        self.bbox[:, 3] = y2 
        return self

    def xyxy2x1y1wh(self) -> Self:
        w = self.bbox[:, 2] - self.bbox[:, 0]
        h = self.bbox[:, 3] - self.bbox[:, 1]
        self.bbox[:, 2] = w 
        self.bbox[:, 3] = h 
        return self 

    def x1y1wh2xcycwh(self) -> Self:
        xc = self.bbox[:, 0] + 0.5 * self.bbox[:, 2]
        yc = self.bbox[:, 1] + 0.5 * self.bbox[:, 3]
        self.bbox[:, 0] = xc 
        self.bbox[:, 1] = yc 
        return self 

    def xcycwh2x1y1wh(self) -> Self:
        x1 = self.bbox[:, 0] - 0.5 * self.bbox[:, 2]
        y1 = self.bbox[:, 1] - 0.5 * self.bbox[:, 3]
        self.bbox[:, 0] = x1 
        self.bbox[:, 1] = y1 
        return self 

    def xcycwh2xyxy(self) -> Self:
        self.xcycwh2x1y1wh()
        return self.x1y1wh2xyxy()

    def xyxy2xcycwh(self) -> Self:
        self.xyxy2x1y1wh()
        return self.x1y1wh2xcycwh()

    def convert(self, target_format: Literal['xyxy','x1y1wh','xcycwh'] = 'xyxy') -> Self:
        assert target_format in ['xyxy','x1y1wh', 'xcycwh']
        funcs = {('x1y1wh','xcycwh'): self.x1y1wh2xcycwh, 
                    ('xcycwh', 'x1y1wh'): self.xcycwh2x1y1wh,
                    ('x1y1wh', 'xyxy'): self.x1y1wh2xyxy,
                    ('xyxy', 'x1y1wh'): self.xyxy2x1y1wh,
                    ('xcycwh', 'x1y1wh'): self.xcycwh2x1y1wh,
                    ('x1y1wh', 'xcycwh'): self.x1y1wh2xcycwh}

        convert_tuple = (self.format, target_format)
        if convert_tuple in funcs:
            funcs[convert_tuple]()
        return self

    def iou(self, other: BBoxes) -> NDArray:
        box1 = self.copy().convert('xyxy').get_box()
        box2 = other.copy().convert('xyxy').get_box()
        x11, y11, x12, y12 = np.split(box1, 4, axis=1)
        x21, y21, x22, y22 = np.split(box2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou 

    def __xor__(self, other: BBoxes) -> NDArray:
        return self.iou(other)

