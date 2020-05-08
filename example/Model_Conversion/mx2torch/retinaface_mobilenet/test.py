import retina 
import cv2 
import numpy as np 

detector = retina.RetinaFace('./model/', True)

img = cv2.imread('rotated90.ppm')

thresh = 0.8
scales = [480, 640]
count = 1

im_shape = img.shape
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
im_scale = float(target_size) / float(im_size_min)
if np.round(im_scale * im_size_max) > max_size:
	im_scale = float(max_size) / float(im_size_max)

print('im_scale', im_scale)

scales2 = [im_scale]
flip = False

faces, landmarks = detector.detect(img, thresh, scales=scales2, do_flip=flip)

if faces is not None:
	print('find', faces.shape[0], 'faces')
	for i in range(faces.shape[0]):
		box = faces[i].astype(np.int)

	#color = (255,0,0)
	color = (0,255,0)
	print('BOX', box)
	print(landmarks)
	# print(box[3]-box[1])
	cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
	if landmarks is not None:
		landmark5 = landmarks[i].astype(np.int)
		for l in range(landmark5.shape[0]):
			color = (0,0,255)
			cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

cv2.imwrite('result.jpg', img)
