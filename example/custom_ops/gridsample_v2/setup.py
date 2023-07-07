from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gridsample',
    ext_modules=[
        CUDAExtension('gridsample', [
            'grid_sample.cpp',
            'grid_sample_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
