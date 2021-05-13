import numpy as np 

# size
inp_size = 224 
out_size = 56
base_sigma = 2.5
num_pts = 17
pairs = [[0,1], [1,2],[2,3], [0,4], [4,5],[5,6], [0,7],[7,8],[8,9],[9,10], [8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

# augmentation 
rotation = 0
min_scale = 1 # this controls largest size 
max_scale = 1 # this controls smallest sise 
max_translate = 0

blur_prob = 0.0
blur_size = [7, 11, 15, 21]
blur_type = ['vertical','horizontal','mean']

# training 
data_root = '/data/pose/mpii/images/'

max_epoch = 300
init_lr = 0.0005
decay = 0.0001
momentum = 0.9
lr_epoch = [150,250]
save_interval = 1

# extra 
distributed = True
scale_var = 19.2 
angle_var = np.pi 
