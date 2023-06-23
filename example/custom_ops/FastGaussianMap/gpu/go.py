import torch 
import fast_guass 
import time 
import cv2 
import numpy as np 

pts = torch.ones(8,32,3).cuda() * 50
pts[2] *= 5
pts[5] *= 7
a = time.time()
for i in range(100):
	out = fast_guass.render_heatmap(pts, 512, 512, 4)
torch.cuda.synchronize()
b = time.time()
print('Time for 100 runs', b-a, 'pts shape:', pts.shape)
print(out.shape)
out = out.cpu().numpy()

out = np.uint8(out * 255)
cv2.imwrite('out.jpg', out[0,0])
cv2.imwrite('out2.jpg', out[2,0])
cv2.imwrite('out3.jpg', out[5,0])
