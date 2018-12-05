import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension, BuildExtension

sources = ['csrc/ave_dist_loss_binds.cpp']
source_cuda = ['csrc/ave_dist_loss_cuda.cu']
include_dirs = ['csrc']  # custom include dir here
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

module = extension('ave_dist_loss._C', 
			sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,)

setup(name='ave_dist_loss',
      ext_modules=[module],
      cmdclass={'build_ext': BuildExtension})
