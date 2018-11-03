from __future__ import print_function
import os
import torch
from torch.utils.ffi import create_extension


sources = []#'src/hough_voting.c']
headers = []#'src/hough_voting.h']
defines = []
with_cuda = torch.cuda.is_available()

if with_cuda:
    print('Including CUDA code.')
    sources += ['src/hough_voting_cuda.c']
    headers += ['src/hough_voting_cuda.h']
    defines += [('WITH_CUDA', None)]

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/hough_voting_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.hough_voting',
    extra_compile_args=['-std=c99'],
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
