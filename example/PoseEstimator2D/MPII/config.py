import numpy as np 

# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

# size
inp_size = 384 
out_size = 192
base_sigma = 4.0
num_pts = 17
num_match_pts = 12

# augmentation 
rotation = 0
min_scale = 1 # this controls largest size 
max_scale = 1 # this controls smallest sise 
max_translate = 0

blur_prob = 0.0
blur_size = [7, 11, 15, 21]
blur_type = ['vertical','horizontal','mean']

# training 
data_root = './images/'

max_epoch = 300
init_lr = 0.001
decay = 0.0001
momentum = 0.9
lr_epoch = [200,250]
save_interval = 1

# extra 
distributed = True
