from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_guass',
    ext_modules=[
        CUDAExtension('fast_guass', [
            'fast_gauss.cpp',
            'fast_guass_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
