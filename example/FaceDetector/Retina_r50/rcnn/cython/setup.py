# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


ext_modules = [
    Extension(
        "bbox",
        ["bbox.pyx"],
        extra_compile_args={},
        include_dirs=[np.get_include()]
    ),
    Extension(
        "anchors",
        ["anchors.pyx"],
        extra_compile_args={},
        include_dirs=[np.get_include()]
    ),
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={},
        include_dirs = [np.get_include()]
    ),
]

setup(
    name='frcnn_cython',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
