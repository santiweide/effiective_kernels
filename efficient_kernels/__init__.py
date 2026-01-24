"""
Efficient CUDA Kernels for PyTorch

This package provides custom CUDA kernels that can be compiled independently 
from PyTorch, supporting both static compilation (.so files) and JIT dynamic 
compilation for development.
"""

import os
import torch
from pathlib import Path

__version__ = "0.1.0"

# Try to import the compiled extension
try:
    from . import _C
    _compiled_available = True
except ImportError:
    _compiled_available = False

# JIT compilation support
def _get_jit_extension():
    """Get JIT-compiled extension for development mode"""
    from torch.utils.cpp_extension import load
    
    package_dir = Path(__file__).parent.parent
    source_files = [
        str(package_dir / 'csrc' / 'kernels' / 'gemm.cu'),
    ]
    
    # Use consistent compile flags with setup.py
    extra_cflags = ['-O3']
    extra_cuda_cflags = ['-O3', '--use_fast_math', '-lineinfo']
    
    return load(
        name='efficient_kernels_jit',
        sources=source_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )

# Lazy load JIT extension
_jit_extension = None

def _get_extension():
    """Get the extension module (compiled or JIT)"""
    global _jit_extension
    
    if _compiled_available:
        return _C
    else:
        # Fall back to JIT compilation
        if _jit_extension is None:
            print("Compiled extension not found, using JIT compilation (development mode)...")
            _jit_extension = _get_jit_extension()
        return _jit_extension

def gemm(A: torch.Tensor, B: torch.Tensor, use_shared: bool = True) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using custom CUDA kernel
    
    Args:
        A: Input tensor of shape [M, K]
        B: Input tensor of shape [K, N]
        use_shared: If True, use shared memory optimization (default: True)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(128, 256).cuda()
        >>> B = torch.randn(256, 512).cuda()
        >>> C = ek.gemm(A, B)
    """
    ext = _get_extension()
    return ext.gemm(A, B, use_shared)

__all__ = ['gemm', '__version__']
