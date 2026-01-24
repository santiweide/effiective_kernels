import os
import torch
from setuptools import setup, Extension
from torch.utils import cpp_extension

# Get CUDA compute capability
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if cuda_arch_list is None:
    cuda_arch_list = ["7.0", "7.5", "8.0", "8.6"]
else:
    cuda_arch_list = cuda_arch_list.split(";")

extra_compile_args = {
    'cxx': ['-O3'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
    ] + [f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}' 
         for arch in cuda_arch_list]
}

setup(
    name='efficient_kernels',
    version='0.1.0',
    author='Your Name',
    description='Custom CUDA kernels for PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['efficient_kernels'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='efficient_kernels._C',
            sources=[
                'csrc/kernels/gemm.cu',
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
    ],
)
