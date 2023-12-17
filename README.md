# TorchSUL

This package is created for a better experience with Pytorch. 


## Quick start

[Quick start](https://github.com/ddddwee1/TorchSUL/blob/master/QuickStart.md)


## References

[Quantization](https://github.com/ddddwee1/TorchSUL/blob/master/QuantInstruction.md)

[Model Flags](https://github.com/ddddwee1/TorchSUL/blob/master/FlagList.md)


## Why making this

- I dont want to write in_channels when building models. It's wired that I should care about input when I only want to write *forward* flows.

- It inlines with my [TF wrap-ups](https://github.com/ddddwee1/sul) so I can easily move my old code and model structures to the current package.


## Installation

You need to install the latest version of pytorch, and python>=3.9

Good, then just 

```
pip install --upgrade torchsul
```


## Patch Notes


#### 2023-xx-xx:  Upgrade to 0.3.0

1. Refactor the codes. Now the structure becomes clearer and more extendable

2. Almost supports type hinting 

3. (Base) build_forward is removed 

4. (Tools) Add more tools


## Projects 

You can find some examples in the "example" folder. It's almost like a trash bin, and some of the functions & modules may be no longer supported in the current version.

- ArcFace (Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018))

- HR Net (Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." arXiv preprint arXiv:1902.09212 (2019))

- AutoDeepLab (Liu, Chenxi, et al. "Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019)

- Knowledge distillation (Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015))

- 3DCNN (Ji, Shuiwang, et al. "3D convolutional neural networks for human action recognition." IEEE transactions on pattern analysis and machine intelligence 35.1 (2012): 221-231)

- Temporal Convolutional Network (Not the same) (Pavllo, Dario, et al. "3D human pose estimation in video with temporal convolutions and semi-supervised training." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019)

- RetinaFace for face detection (Deng, Jiankang, et al. "Retinaface: Single-stage dense face localisation in the wild." arXiv preprint arXiv:1905.00641 (2019))

- Fractal Net (Larsson, Gustav, Michael Maire, and Gregory Shakhnarovich. "Fractalnet: Ultra-deep neural networks without residuals." arXiv preprint arXiv:1605.07648 (2016))

- Polarized Self Attention (Liu, Huajun, et al. "Polarized self-attention: Towards high-quality pixel-wise regression." arXiv preprint arXiv:2107.00782 (2021))

- Some 2D/3D pose estimation works

- Some network structures

- Some cuda kernels
