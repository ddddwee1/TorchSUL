import cv2
import os 
from tqdm import tqdm

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

	print(f'Extracting {fname}\tOutput_dir:{output_dir}')
	os.makedirs(output_dir, exist_ok=True)
	cap = cv2.VideoCapture(fname)
	framenum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	cnt = 0
	images = []
	for ret, frame in tqdm(make_iterable(cap), total=framenum):
		if not ret:
			break
		if cnt%skip==0:
			if return_images:
				images.append(frame)
			else:
				imgname = os.path.join(output_dir, frame_format%cnt + '.' + ext)
				cv2.imwrite(imgname, frame)
		cnt += 1 

	print('Extraction finished.')
	if return_images:
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

