import os
import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

# Get CUDA compute capability
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
if cuda_arch_list is None:
    # Default architectures: Volta, Turing, Ampere, Ada
    cuda_arch_list = ["7.0", "7.5", "8.0", "8.6", "8.9"]
else:
    cuda_arch_list = cuda_arch_list.split(";")

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
        '-std=c++17',
    ] + [f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}' 
         for arch in cuda_arch_list]
}

# Check if we're being built by pip (PEP 517) or directly
# This allows both `pip install -e .` and `python setup.py develop` to work
setup(
    name='efficient_kernels',
    version='0.1.0',
    author='Your Name',
    description='Custom CUDA kernels for PyTorch',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'examples', 'build']),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='efficient_kernels._C',
            sources=[
                'csrc/kernels/gemm.cu',
            ],
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CUDAExtension(
            name='efficient_kernels._C_optimized',
            sources=[
                'csrc/kernels/gemm_optimized.cu',
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=True)
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.8.0',
    ],
)
