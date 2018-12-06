import torch
import tensorflow as tf
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension

sources = ['csrc/hough_voting_binds.cpp']
source_cuda = ['csrc/hough_voting_cuda.cu']
include_dirs = ['csrc', tf.sysconfig.get_include()]  # custom include dir here
define_macros = []
extra_compile_args = {"cxx": []}

extension = CUDAExtension
if torch.cuda.is_available() and CUDA_HOME is not None:
    # extension = CUDAExtension
    sources += source_cuda
    define_macros += [("WITH_CUDA", None)]
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        # "-G", "-g",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
else:
    raise NotImplementedError

module = extension('_C', 
			sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,)

setup(name='hough_voting',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension})
