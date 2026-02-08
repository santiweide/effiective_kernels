"""
Example demonstrating JIT compilation for development mode
"""

import torch

def main():
    print("=" * 60)
    print("JIT Compilation Example (Development Mode)")
    print("=" * 60)
    
    # This will trigger JIT compilation if the extension is not already compiled
    print("\nImporting efficient_kernels...")
    print("(This may take a moment if JIT compilation is triggered)")
    
    import efficient_kernels as ek
    
    print(f"\nSuccessfully loaded efficient_kernels version: {ek.__version__}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-enabled GPU.")
        return
    
    # Simple test
    print("\nRunning simple GEMM test...")
    A = torch.randn(64, 128, device='cuda')
    B = torch.randn(128, 256, device='cuda')
    
    C = ek.gemm(A, B)
    C_ref = torch.matmul(A, B)
    
    max_diff = torch.max(torch.abs(C - C_ref))
    print(f"Max difference from PyTorch: {max_diff.item():.6e}")
    
    if max_diff < 1e-3:
        print("\n✓ JIT compilation successful! Kernel works correctly.")
    else:
        print("\n✗ Large difference detected. Please check the kernel implementation.")
    
    print("\n" + "=" * 60)
    print("JIT compilation allows you to modify the kernel code")
    print("and test immediately without running 'pip install'!")
    print("=" * 60)

if __name__ == "__main__":
    main()
