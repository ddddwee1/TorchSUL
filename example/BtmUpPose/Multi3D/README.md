# Bottom-up method for 3D multi-person pose estimation 

[Pre-trained models](https://www.dropbox.com/sh/f6rqqnkyox4c9p1/AABaHcTRXmRypIzkbacRpnQya?dl=0)

## How to use:

1. Process coco dataset with coco_process/dump.py and filt_coco.py. You will get 'filtered_coco_kpts.pkl' which contains the labels, and a folder 'masks' which contains the mask information for each image. 

2. Process muco dataset with muco_process/process.py. You will get 'augmented.pkl' and 'unaumgmented.pkl' that contain labels of augmented dataset and unugmented dataset accordingly. 

3. Change paths in train/config.py 

4. Train the network using 'cd train/ && distrib.py' ** remember to download the imagenet pre-trained model **

5. Test the network using test/run_img.py. Remember to copy the config.py 

