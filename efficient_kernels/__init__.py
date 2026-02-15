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

# Try to import the compiled extensions
try:
    from . import _C
    _compiled_available = True
except ImportError:
    _compiled_available = False

try:
    from . import _C_optimized
    _compiled_optimized_available = True
except ImportError:
    _compiled_optimized_available = False

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

def _get_jit_extension_optimized():
    """Get JIT-compiled optimized extension for development mode"""
    from torch.utils.cpp_extension import load
    
    package_dir = Path(__file__).parent.parent
    source_files = [
        str(package_dir / 'csrc' / 'kernels' / 'gemm_optimized.cu'),
    ]
    
    extra_cflags = ['-O3']
    extra_cuda_cflags = ['-O3', '--use_fast_math', '-lineinfo']
    
    return load(
        name='efficient_kernels_optimized_jit',
        sources=source_files,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True
    )

# Lazy load JIT extension
_jit_extension = None
_jit_extension_optimized = None

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

def _get_extension_optimized():
    """Get the optimized extension module (compiled or JIT)"""
    global _jit_extension_optimized
    
    if _compiled_optimized_available:
        return _C_optimized
    else:
        # Fall back to JIT compilation
        if _jit_extension_optimized is None:
            print("Compiled optimized extension not found, using JIT compilation...")
            _jit_extension_optimized = _get_jit_extension_optimized()
        return _jit_extension_optimized

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


def gemm_splitk(A: torch.Tensor, B: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using Split-K parallelization.
    
    Split-K divides the K dimension into multiple chunks, processes them in 
    parallel across different thread blocks, and reduces the partial results.
    
    This is beneficial when:
    - K is very large compared to M and N
    - The output tile count is small (not enough parallelism in M*N)
    
    Args:
        A: Input tensor of shape [M, K]
        B: Input tensor of shape [K, N]
        k_splits: Number of K-dimension splits (default: 4)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(64, 4096).cuda()  # Small M, large K
        >>> B = torch.randn(4096, 64).cuda()
        >>> C = ek.gemm_splitk(A, B, k_splits=8)
    """
    ext = _get_extension()
    return ext.gemm_splitk(A, B, k_splits)


def gemm_streamk(A: torch.Tensor, B: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using StreamK-style parallelization.
    
    StreamK-style approach:
    - Multiple blocks work on the same output tile along K dimension
    - Uses atomic operations to accumulate partial results
    - Better load balancing compared to traditional tiled GEMM
    
    Best for:
    - Irregular matrix sizes where traditional tiling leaves SMs underutilized
    - When K >> M*N (computation bound in K dimension)
    
    Args:
        A: Input tensor of shape [M, K]
        B: Input tensor of shape [K, N]
        k_splits: Number of K-dimension splits (default: 4)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Note:
        Uses atomicAdd which may have performance overhead on older GPUs.
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(32, 8192).cuda()
        >>> B = torch.randn(8192, 32).cuda()
        >>> C = ek.gemm_streamk(A, B, k_splits=4)
    """
    ext = _get_extension()
    return ext.gemm_streamk(A, B, k_splits)


def gemm_vectorized(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using vectorized memory access (float4).
    
    Uses float4 (128-bit) vectorized loads/stores to maximize memory bandwidth.
    This kernel uses larger tiles (128x128) and register blocking for better
    arithmetic intensity.
    
    Best for:
    - Large matrices where memory bandwidth is the bottleneck
    - Float32 tensors (only supported dtype)
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_vectorized(A, B)
    """
    ext = _get_extension()
    return ext.gemm_vectorized(A, B)


def gemm_splitk_vectorized(A: torch.Tensor, B: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using vectorized Split-K.
    
    Combines Split-K parallelization with float4 vectorized memory access.
    Each K-split uses larger tiles and vectorized loads for better bandwidth.
    
    Best for:
    - Large K dimension with moderate M and N
    - Float32 tensors (only supported dtype)
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        k_splits: Number of K-dimension splits (default: 4)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(256, 4096, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(4096, 256, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_splitk_vectorized(A, B, k_splits=4)
    """
    ext = _get_extension()
    return ext.gemm_splitk_vectorized(A, B, k_splits)


def gemm_streamk_vectorized(A: torch.Tensor, B: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """
    Perform matrix multiplication C = A @ B using vectorized StreamK.
    
    Combines StreamK-style parallelization with optimized memory access.
    Uses loop unrolling for better instruction-level parallelism.
    
    Best for:
    - Matrices with large K dimension
    - Float32 tensors (only supported dtype)
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        k_splits: Number of K-dimension splits (default: 4)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(128, 8192, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(8192, 128, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_streamk_vectorized(A, B, k_splits=4)
    """
    ext = _get_extension()
    return ext.gemm_streamk_vectorized(A, B, k_splits)


# =============================================================================
# Optimized Kernels (No Atomics, Reduced Register Pressure)
# =============================================================================

def gemm_optimized(A: torch.Tensor, B: torch.Tensor, version: int = 1) -> torch.Tensor:
    """
    Perform matrix multiplication using optimized GEMM kernel.
    
    Key optimizations over standard GEMM:
    - Smaller per-thread output tiles (2x2 or 4x4 vs 8x8)
    - Reduced register pressure for better occupancy
    - Shared memory padding to avoid bank conflicts
    - __launch_bounds__ for register control
    - Float32 only (no double support overhead)
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        version: 1 for 2x2 per thread (4 accumulators), 
                 2 for 4x4 per thread (16 accumulators)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_optimized(A, B, version=2)
    """
    ext = _get_extension_optimized()
    return ext.gemm_optimized(A, B, version)


def gemm_optimized_v1(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized GEMM v1: 16x16 threads, 2x2 output per thread.
    
    - TILE_SIZE = 32, each block computes 32x32 output
    - 256 threads (16x16), each thread computes 4 elements (2x2)
    - Only 4 accumulator registers per thread (vs 64 in naive 8x8)
    - __launch_bounds__(256, 4) for optimal register allocation
    
    Best for:
    - Matrices where high occupancy is critical
    - When register pressure is the bottleneck
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(512, 512, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(512, 512, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_optimized_v1(A, B)
    """
    ext = _get_extension_optimized()
    return ext.gemm_optimized_v1(A, B)


def gemm_optimized_v2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized GEMM v2: 16x16 threads, 4x4 output per thread.
    
    - TILE_M=64, TILE_N=64, TILE_K=16
    - 256 threads (16x16), each thread computes 16 elements (4x4)
    - 16 accumulator registers per thread
    - Uses register-level blocking for A and B fragments
    - __launch_bounds__(256, 2) for larger register budget
    
    Best for:
    - Larger matrices where arithmetic intensity matters
    - Balance between occupancy and per-thread work
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_optimized_v2(A, B)
    """
    ext = _get_extension_optimized()
    return ext.gemm_optimized_v2(A, B)


def gemm_splitk_v2(A: torch.Tensor, B: torch.Tensor, k_splits: int = 4) -> torch.Tensor:
    """
    Split-K GEMM v2: No atomics, tile-local partial sums + reduce.
    
    Key improvements over original Split-K:
    - NO atomicAdd (avoids write conflicts and memory ordering issues)
    - Tile-local workspace instead of [k_splits, M, N] global workspace
    - Two-phase approach: compute partial sums, then reduce
    - Much better memory access patterns
    
    Memory usage: O(num_tiles_m * num_tiles_n * k_splits * TILE_SIZE^2)
    vs original: O(k_splits * M * N)
    
    For a 4096x4096 matrix with k_splits=4:
    - Original: 4 * 4096 * 4096 * 4 = 256 MB
    - This version: 128 * 128 * 4 * 32^2 * 4 = 256 MB (same, but better locality)
    
    Args:
        A: Input tensor of shape [M, K] (float32 only)
        B: Input tensor of shape [K, N] (float32 only)
        k_splits: Number of K-dimension splits (1-16, default: 4)
        
    Returns:
        Output tensor C of shape [M, N]
        
    Example:
        >>> import torch
        >>> import efficient_kernels as ek
        >>> A = torch.randn(256, 8192, device='cuda', dtype=torch.float32)
        >>> B = torch.randn(8192, 256, device='cuda', dtype=torch.float32)
        >>> C = ek.gemm_splitk_v2(A, B, k_splits=8)
    """
    ext = _get_extension_optimized()
    return ext.gemm_splitk_v2(A, B, k_splits)


__all__ = [
    # Standard GEMM kernels
    'gemm', 
    'gemm_splitk', 
    'gemm_streamk', 
    'gemm_vectorized',
    'gemm_splitk_vectorized',
    'gemm_streamk_vectorized',
    # Optimized GEMM kernels (no atomics, reduced register pressure)
    'gemm_optimized',
    'gemm_optimized_v1',
    'gemm_optimized_v2',
    'gemm_splitk_v2',
    '__version__'
]
