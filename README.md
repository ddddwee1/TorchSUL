# TorchSUL

This package is created for better experience while using Pytorch. 

## Why making this

1. For fun.

2. Path-dependence. I am addicted to my own wrap-ups. 

3. Multi-platform. I have made the same APIs for pytorch, TF, MXNet, and a conversion tool to Caffe. 

4. Some strange reason. Frameworks like TF, MXNet, Caffe, Paddle do not need to claim input shape to initialize layers, but frameworks like pytorch, torch, chainer require this. I prefer not to claim since it will be more convenient when building models (Why I need to care about previous layers when I only want to write forward computation?), so I modified pytorch module to support this. Also, it inlines with my [TF wrap-ups](https://github.com/ddddwee1/sul) so I can move my old code easily to the current package.

## Installation

You need to install the latest version of pytorch.

Good, then just 

```
pip install --upgrade torchsul
```

## Patch Notes

#### 2023-08-05 (0.2.4)
1. Add "loose_load" flag. Working similarly to the "strict=False"
2. Fix a bug in sul config that would not support multi-processing
3. Add more initialization options for conv and fc layers

#### 2023-07-27 (0.2.2)
1. Add support for loading standard pytorch state dict. Users can set "from_torch" flag to load from standard pytorch state dict.


#### 2023-07-16:  Bugfixes (0.2.1)
1. Now *M.Model.\_laod_from_state_dict2* can normally manipulate state dictionary of its child modules 
2. Add quantization support for fc layers. 


#### 2023-07-15:  Upgrade to 0.2.0
1. This is acually a pruning of previous version, where many redundant and outdated modules/functions are removed. You may find some layers are not supported anymore because there should be a convenient pytorch equivalance to be used. 
2. This package is reformed since version 0.2.0. The submodule "DataReader" and "sulio" are removed, and all codes which utilizes these submodules will *no longer be supported*. It is recommended to use pytorch's dataloader for reading data. 
3. Add config module to handle configs from yaml files. One the one hand, it's easier to control experiments from outside; on the other hand, build-in configs will prettify the codes.
4. Remove caffe conversion support in package as it's redundant. One can use external method to build caffe converter (e.g, forward hooks) to convert models. As an alternative, the caffe conversion codes (modified Models & layers.py) still remain in examples. 




## Projects 

You can find some examples in the "example" folder.

- ArcFace (Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018))

- HR Net (Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." arXiv preprint arXiv:1902.09212 (2019))

- AutoDeepLab (Liu, Chenxi, et al. "Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019)

- Knowledge distillation (Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015))

- 3DCNN (Ji, Shuiwang, et al. "3D convolutional neural networks for human action recognition." IEEE transactions on pattern analysis and machine intelligence 35.1 (2012): 221-231)

- Temporal Convolutional Network (Not the same) (Pavllo, Dario, et al. "3D human pose estimation in video with temporal convolutions and semi-supervised training." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019)

- RetinaFace for face detection (Deng, Jiankang, et al. "Retinaface: Single-stage dense face localisation in the wild." arXiv preprint arXiv:1905.00641 (2019))

- Fractal Net (Larsson, Gustav, Michael Maire, and Gregory Shakhnarovich. "Fractalnet: Ultra-deep neural networks without residuals." arXiv preprint arXiv:1605.07648 (2016))

- Polarized Self Attention (Liu, Huajun, et al. "Polarized self-attention: Towards high-quality pixel-wise regression." arXiv preprint arXiv:2107.00782 (2021))

- Model conversions 

- Batch_norm compression to speed-up models 

