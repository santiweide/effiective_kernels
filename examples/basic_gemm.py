"""
Basic example demonstrating the custom GEMM kernel
"""

import torch
import efficient_kernels as ek

def main():
    print("=" * 60)
    print("Efficient Kernels - GEMM Example")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-enabled GPU.")
        return
    
    print(f"\nUsing efficient_kernels version: {ek.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Setup test matrices
    M, K, N = 512, 1024, 256
    print(f"\nMatrix dimensions: A[{M}, {K}] @ B[{K}, {N}] = C[{M}, {N}]")
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warm up
    print("\nWarming up...")
    for _ in range(10):
        _ = ek.gemm(A, B)
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark custom kernel
    print("\nBenchmarking custom GEMM kernel...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        C_custom = ek.gemm(A, B, use_shared=True)
    end.record()
    torch.cuda.synchronize()
    custom_time = start.elapsed_time(end) / 100
    
    # Benchmark PyTorch matmul
    print("Benchmarking PyTorch matmul...")
    start.record()
    for _ in range(100):
        C_torch = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / 100
    
    # Verify correctness
    print("\nVerifying correctness...")
    max_diff = torch.max(torch.abs(C_custom - C_torch))
    mean_diff = torch.mean(torch.abs(C_custom - C_torch))
    print(f"Max difference: {max_diff.item():.6e}")
    print(f"Mean difference: {mean_diff.item():.6e}")
    
    # Print results
    print("\n" + "=" * 60)
    print("Performance Results:")
    print("=" * 60)
    print(f"Custom GEMM kernel: {custom_time:.4f} ms")
    print(f"PyTorch matmul:     {torch_time:.4f} ms")
    print(f"Speedup:            {torch_time / custom_time:.2f}x")
    print("=" * 60)
    
    # Test with different optimization
    print("\nTesting without shared memory optimization...")
    C_no_shared = ek.gemm(A, B, use_shared=False)
    max_diff = torch.max(torch.abs(C_no_shared - C_torch))
    print(f"Max difference (no shared): {max_diff.item():.6e}")

    # =========================================================================
    # StreamK GEMM Profile
    # =========================================================================
    print("\n" + "=" * 60)
    print("StreamK GEMM Profile")
    print("=" * 60)
    
    # Warm up StreamK
    print("\nWarming up StreamK...")
    for _ in range(10):
        _ = ek.gemm_streamk(A, B, k_splits=4)
    torch.cuda.synchronize()
    
    # Test different k_splits values
    k_splits_list = [1, 2, 4, 8]
    streamk_results = {}
    
    for k_splits in k_splits_list:
        start.record()
        for _ in range(100):
            C_streamk = ek.gemm_streamk(A, B, k_splits=k_splits)
        end.record()
        torch.cuda.synchronize()
        streamk_time = start.elapsed_time(end) / 100
        streamk_results[k_splits] = streamk_time
        
        # Verify correctness
        max_diff = torch.max(torch.abs(C_streamk - C_torch))
        print(f"StreamK (k_splits={k_splits}): {streamk_time:.4f} ms, "
              f"max_diff: {max_diff.item():.6e}")
    
    # Also benchmark Split-K
    print("\n" + "-" * 40)
    print("Split-K GEMM Profile")
    print("-" * 40)
    
    for k_splits in k_splits_list:
        start.record()
        for _ in range(100):
            C_splitk = ek.gemm_splitk(A, B, k_splits=k_splits)
        end.record()
        torch.cuda.synchronize()
        splitk_time = start.elapsed_time(end) / 100
        
        # Verify correctness
        max_diff = torch.max(torch.abs(C_splitk - C_torch))
        print(f"Split-K (k_splits={k_splits}): {splitk_time:.4f} ms, "
              f"max_diff: {max_diff.item():.6e}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"{'Method':<25} {'Time (ms)':<15} {'vs PyTorch':<15}")
    print("-" * 55)
    print(f"{'PyTorch matmul':<25} {torch_time:<15.4f} {'1.00x':<15}")
    print(f"{'Custom GEMM (shared)':<25} {custom_time:<15.4f} {torch_time/custom_time:<15.2f}x")
    best_streamk = min(streamk_results.values())
    best_k = min(streamk_results, key=streamk_results.get)
    print(f"{'StreamK (best k=' + str(best_k) + ')':<25} {best_streamk:<15.4f} {torch_time/best_streamk:<15.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    main()
