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

- Model conversions 

- Batch_norm compression to speed-up models 

