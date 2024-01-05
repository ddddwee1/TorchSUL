import os

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from typing import Tuple, List

from ..Consts.Types import *
from .Progress import progress_bar


# video utils 
class VideoSaver():
    def __init__(self, name: str, size: Union[Tuple[int,int], Literal['auto']] = 'auto', frame_rate: float = 15.0):
        self.name = name
        self.frame_rate = frame_rate
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
        self.size = size 

        if size=='auto':
            self.vidwriter = None 
        else:
            self.vidwriter = cv2.VideoWriter(self.name, self.fourcc, self.frame_rate, (self.size[1],self.size[0]))

    def write(self, img: NDArray):
        if self.vidwriter is None:
            self.size = [img.shape[0], img.shape[1]]
            self.vidwriter = cv2.VideoWriter(self.name, self.fourcc, self.frame_rate,(self.size[1],self.size[0]))
        assert (img.shape[0]==self.size[0]) and (img.shape[1]==self.size[1]), f'VideoSaver: Image shape is not consistent. Set {self.size}, but got {img.shape}'
        self.vidwriter.write(img)

    def finish(self):
        self.vidwriter.release()


def check_frame_num(fname: str) -> int:
    video = cv2.VideoCapture(fname)
    framenum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return framenum


def check_fps(fname: str) -> float:
    video = cv2.VideoCapture(fname)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps 


def combine_audio(vidname: str, audname: str, outname: str, fps: float=25) -> None:
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)


def compress_video(vidname: str, outname: str, fps: float=25) -> None:
    import moviepy.editor as mpe
    my_clip = mpe.VideoFileClip(vidname)
    my_clip.write_videofile(outname, fps=fps)


def extract_frames(fname: str, output_dir: str, ext: BasicImageTypes='jpg', skip: int=0, frame_format: str='frame_%08d', return_images: bool=False) -> List[NDArray[np.uint8]]:
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
            if cnt%(skip+1)==0:
                if return_images:
                    images.append(frame)
                else:
                    imgname = os.path.join(output_dir, frame_format%cnt + '.' + ext)
                    cv2.imwrite(imgname, frame)
            cnt += 1 
            prog.advance(task)

    logger.info('Extraction finished.')
    return images
