# ViT for 2D Pose

## Note

Must use batch-norm for conv layers. Not sure what is the mechanism behind, but I failed when trying to remove the bn.

## Pre-trained models 

Take it from [DINO](https://github.com/facebookresearch/dino)

Vit (Deit) small, patch size 8.  [Download](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth)

## Processed MPII keypoints 

Just for training a toy model, not for high performance. [Download](https://www.dropbox.com/s/3vabjyp39ol9ao0/mpii_3pts.pkl?dl=0)

## Gaussian heatmap renderer

Code from [here](https://github.com/ddddwee1/TorchSUL/tree/master/example/FastGaussianMap). Rename the folder name to "fastgaus" and compile according to the readme file. 

## Usage

### Training 

```
python distrib.py 
```

### Visualization

```
python test.py
```

A file named "hmap_attn.png" will be generated. It contains (n_head+2, n_keypoints) grids of images. First row is the heatmap and last row is the average of attention maps from all heads.
