import cv2
import numpy as np 
import os 
import time 
import progressbar

def extract_frames(fname,prefix,skip=1):
	print('Extracting %s'%fname)
	t1 = time.time()
	cap = cv2.VideoCapture(fname)
	framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cnt = 0
	images = []
	with progressbar.ProgressBar(max_value=framenum) as bar:
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break
			if cnt%skip==0:
				if prefix is None:
					images.append(frame)
				else:
					imgname = prefix+'_%08d.jpg'%cnt
					if not os.path.exists(imgname):
						cv2.imwrite(imgname, frame)
			bar.update(cnt)
			cnt += 1
	t2 = time.time()
	print('Extraction finished. Time elapsed:', t2-t1)
	return images

class video_saver():
	def __init__(self,name,size, frame_rate=15.0):
		self.name = name
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.vidwriter = cv2.VideoWriter(name,fourcc,frame_rate,(size[1],size[0]))
	def write(self,img):
		self.vidwriter.write(img)
	def finish(self):
		self.vidwriter.release()

def computeIOU(box1, box2):
	x1, y1, x2, y2 = box1
	x3, y3, x4, y4 = box2 
	dx = max(0, min(x2,x4) - max(x1,x3))
	dy = max(0, min(y2,y4) - max(y1,y3))
	inter = dx*dy 
	union = (x4 - x3) * (y4 - y3) + (x2 - x1) * (y2 - y1) - inter
	iou = inter / (union + 1)
	return iou 

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

