from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gridsample_nn',
    ext_modules=[
        CUDAExtension('gridsample_nn', [
            'gridsample_nn.cpp',
            'gridsample_nn_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
