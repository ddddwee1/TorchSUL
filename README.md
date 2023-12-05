# TorchSUL

This package is created for a better experience while using Pytorch. 

## Quick start

[Quick start](https://github.com/ddddwee1/TorchSUL/blob/master/QuickStart.md)

## References

[Quantization](https://github.com/ddddwee1/TorchSUL/blob/master/QuantInstruction.md)

[Model Flags](https://github.com/ddddwee1/TorchSUL/blob/master/FlagList.md)

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

#### 2023-11-13 (0.2.10)
1. (Layer) Add warning when loading an un-initialized convLayer, deconvLayer or Dense, instead of raising KeyError.

#### 2023-09-30 (0.2.9)
1. (Model) Fix a bug that nn.Sequential is ignored when inspecting quant params
2. (Layers) Now conv layers support non-squared kernels

#### 2023-09-15 (0.2.8)
1. (Base) Using config inside the base module is not a wise option. So we remove it from base module.
2. (Model) Fix a bug of loading state_dict for dense layer when in "from_torch" mode.
3. (tool) Fall back to XVID codec.

#### 2023-09-11 (0.2.7)
1. (Base) We deprecate the build_forward functions. Reason is that if we use this function for parameter initialization purpose, it will be more readable and user-friendly to use it after parameters are initialized (after forward function). Therefore, this functionality will be achieved by another post-forward function "init_params", and other functionalities would be achieved by "*if self._is_built*" statement during forward.
2. (Quant) Non-calibrated layers will be auto-omitted and skipped in forward loops.
3. (General) Switch from print to loguru, a simple but powerful logging package.
4. (General) Switch from tqdm to rich progress

#### 2023-08-18 (0.2.6)
1. (Quant) Non-existing quant params will no longer trigger Exceptions when inspecting, triggering warnings instead.

#### 2023-08-16 (0.2.5)
1. (Layers) Add quant support for deconv layer.

#### 2023-08-05 (0.2.4)
1. (Layers) Add "loose_load" flag. Working similarly to the "strict=False".
2. (Config) Fix a bug in sul config that would not support multi-processing.
3. (Layers) Add more initialization options for conv and fc layers.

#### 2023-07-27 (0.2.2)
1. (Layers) Add support for loading standard pytorch state dict. Users can set "from_torch" flag to load from standard pytorch state dict.


#### 2023-07-16:  Bugfixes (0.2.1)
1. (Base) Now *M.Model.\_laod_from_state_dict2* can normally manipulate state dictionary of its child modules.
2. (Layers) Add quantization support for fc layers. 


#### 2023-07-15:  Upgrade to 0.2.0
1. This is acually a pruning of previous version, where many redundant and outdated modules/functions are removed. You may find some layers are not supported anymore because there should be a convenient pytorch equivalance to be used. 
2. This package is reformed since version 0.2.0. The submodule "DataReader" and "sulio" are removed, and all codes which utilizes these submodules will *no longer be supported*. It is recommended to use pytorch's dataloader for reading data. 
3. Add config module to handle configs from yaml files. One the one hand, it's easier to control experiments from outside; on the other hand, build-in configs will prettify the codes.
4. Remove caffe conversion support in package as it's redundant. One can use external method to build caffe converter (e.g, forward hooks) to convert models. As an alternative, the caffe conversion codes (modified Models & layers.py) still remain in examples. 




## Projects 

You can find some examples in the "example" folder. It's almost like a trash bin.

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
