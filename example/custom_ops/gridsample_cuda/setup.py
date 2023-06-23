from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gridsample',
    ext_modules=[
        CUDAExtension('gridsample', [
            'gridsample.cpp',
            'gridsample_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
