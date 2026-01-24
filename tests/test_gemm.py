"""
Simple tests for the GEMM kernel
"""

import torch
import pytest

# Import will trigger JIT compilation if not already compiled
import efficient_kernels as ek


def test_gemm_basic():
    """Test basic GEMM functionality"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    M, K, N = 32, 64, 48
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    C = ek.gemm(A, B)
    C_ref = torch.matmul(A, B)
    
    assert C.shape == (M, N)
    assert torch.allclose(C, C_ref, rtol=1e-3, atol=1e-5)


def test_gemm_shared_vs_no_shared():
    """Test that both kernel versions produce same results"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    M, K, N = 128, 256, 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    C_shared = ek.gemm(A, B, use_shared=True)
    C_no_shared = ek.gemm(A, B, use_shared=False)
    
    assert torch.allclose(C_shared, C_no_shared, rtol=1e-4, atol=1e-6)


def test_gemm_different_sizes():
    """Test GEMM with different matrix sizes"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    test_cases = [
        (16, 32, 16),
        (100, 200, 150),
        (33, 65, 97),  # Non-power-of-2 sizes
    ]
    
    for M, K, N in test_cases:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        C = ek.gemm(A, B)
        C_ref = torch.matmul(A, B)
        
        assert torch.allclose(C, C_ref, rtol=1e-3, atol=1e-5), \
            f"Failed for size [{M}, {K}] x [{K}, {N}]"


def test_gemm_dtype_float():
    """Test GEMM with float32"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    C = ek.gemm(A, B)
    assert C.dtype == torch.float32


if __name__ == "__main__":
    print("Running tests...")
    test_gemm_basic()
    print("✓ test_gemm_basic passed")
    
    test_gemm_shared_vs_no_shared()
    print("✓ test_gemm_shared_vs_no_shared passed")
    
    test_gemm_different_sizes()
    print("✓ test_gemm_different_sizes passed")
    
    test_gemm_dtype_float()
    print("✓ test_gemm_dtype_float passed")
    
    print("\nAll tests passed!")
