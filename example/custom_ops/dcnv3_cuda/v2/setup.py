from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dcn',
    ext_modules=[
        CUDAExtension('dcn', [
            'dcn.cpp',
            'dcn_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
