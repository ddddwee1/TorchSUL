from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='onnx_runner',
    ext_modules=[
        CUDAExtension('onnx_runner', [
            'onnx_runner.cpp', 'cust_ops.cpp'
        ], include_dirs=['/mnt/c/Users/cy960/Desktop/onnxrt/onnxruntime/include/onnxruntime/core/session'],
            libraries=['onnxruntime'],
            library_dirs=['/mnt/c/Users/cy960/Desktop/onnxrt/onnxruntime/build/Linux/RelWithDebInfo'],
        ),
    ],
    
    cmdclass={
        'build_ext': BuildExtension
    })