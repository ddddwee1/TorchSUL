import json 
import glob 
from tqdm import tqdm 

res = []
data = json.load(open('metadata.json'))
cnt = 0
for k in tqdm(data):
	# print(data[k])
	folder = k
	folder = folder.replace('.mp4','')
	label = data[k]['label']
	if label == 'REAL':
		label = 1
	else:
		label = 0
	# print('./dfdc_train_part_0/%s/*.jpg'%folder)
	imgs = glob.glob('./dfdc_train_part_0/%s/*.jpg'%folder)
	if len(imgs) == 300:
		for i in imgs:
			i = i.replace('\\','/')
			res.append([i, label, cnt])
		cnt += 1

f = open('imgslist.txt', 'w')
for i in tqdm(res):
	f.write(i[0] + '\t' + str(i[1]) + '\t' + str(i[2]) + '\n')
f.close()
