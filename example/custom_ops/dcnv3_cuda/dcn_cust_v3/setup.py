from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dcnv3',
    ext_modules=[
        CUDAExtension('dcnv3', [
            'dcnv3.cpp',
            'dcnv3_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
