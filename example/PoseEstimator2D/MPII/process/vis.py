import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
import cv2 

data = pickle.load(open('mpii.pkl', 'rb'))

def plot(img, pts):
	img = './images/%s'%img
	img = cv2.imread(img)
	img = img[:,:,::-1]
	plt.imshow(img)
	for i in range(17):
		plt.plot(pts[i,0], pts[i,1], 'o')
		plt.text(pts[i,0], pts[i,1], '%d'%i)

for i in data:
	img = i[0]
	pts = i[1]
	# print(pts[:,2])
	# input()
	idx = np.where(pts[:,2]>0)[0]
	# print(len(idx))
	if len(idx) == 17:
		print(img)
		plot(img, pts)
		plt.show()
