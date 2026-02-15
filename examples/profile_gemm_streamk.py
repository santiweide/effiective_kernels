"""
Profile gemm_streamk kernel using PyTorch profiler and NVIDIA Nsight

Usage:
    # Basic profiling with PyTorch profiler
    python examples/profile_gemm_streamk.py
    
    # Generate Chrome trace
    python examples/profile_gemm_streamk.py --trace
    
    # Profile with Nsight Compute (requires nsys/ncu)
    nsys profile python examples/profile_gemm_streamk.py --nsight
    ncu --set full python examples/profile_gemm_streamk.py --nsight
"""

import torch
import torch.nn.functional as F
import argparse
import os
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

# Import efficient_kernels
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import efficient_kernels as ek


def warmup(A, B, iterations=10):
    """Warmup GPU"""
    for _ in range(iterations):
        _ = ek.gemm_streamk(A, B, k_splits=4)
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()


def benchmark_streamk(A, B, k_splits_list=[1, 2, 4, 8], iterations=100):
    """Benchmark gemm_streamk with different k_splits"""
    M, K = A.shape
    _, N = B.shape
    print(f"\n{'='*70}")
    print(f"Benchmarking gemm_streamk: A[{M}, {K}] @ B[{K}, {N}]")
    print(f"{'='*70}")
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    results = {}
    
    # Benchmark PyTorch matmul
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        C_torch = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / iterations
    results['pytorch'] = torch_time
    print(f"\nPyTorch matmul:        {torch_time:.4f} ms")
    
    # Benchmark standard GEMM (shared memory)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        C_shared = ek.gemm(A, B, use_shared=True)
    end.record()
    torch.cuda.synchronize()
    shared_time = start.elapsed_time(end) / iterations
    results['gemm_shared'] = shared_time
    print(f"Custom GEMM (shared):  {shared_time:.4f} ms")
    
    # Benchmark Split-K
    for k_splits in k_splits_list:
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            C_splitk = ek.gemm_splitk(A, B, k_splits=k_splits)
        end.record()
        torch.cuda.synchronize()
        splitk_time = start.elapsed_time(end) / iterations
        results[f'splitk_{k_splits}'] = splitk_time
        print(f"Split-K (k={k_splits}):        {splitk_time:.4f} ms")
    
    # Benchmark StreamK
    for k_splits in k_splits_list:
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            C_streamk = ek.gemm_streamk(A, B, k_splits=k_splits)
        end.record()
        torch.cuda.synchronize()
        streamk_time = start.elapsed_time(end) / iterations
        results[f'streamk_{k_splits}'] = streamk_time
        
        # Verify correctness
        max_diff = torch.max(torch.abs(C_streamk - C_torch)).item()
        print(f"StreamK (k={k_splits}):        {streamk_time:.4f} ms  (diff: {max_diff:.2e})")
    
    return results


def profile_with_pytorch_profiler(A, B, k_splits=4, output_dir="profiler_traces"):
    """Profile using PyTorch profiler"""
    M, K = A.shape
    _, N = B.shape
    
    print(f"\n{'='*70}")
    print(f"PyTorch Profiler: A[{M}, {K}] @ B[{K}, {N}], k_splits={k_splits}")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("gemm_streamk"):
            for _ in range(10):
                C = ek.gemm_streamk(A, B, k_splits=k_splits)
            torch.cuda.synchronize()
        
        with record_function("gemm_splitk"):
            for _ in range(10):
                C = ek.gemm_splitk(A, B, k_splits=k_splits)
            torch.cuda.synchronize()
        
        with record_function("gemm_shared"):
            for _ in range(10):
                C = ek.gemm(A, B, use_shared=True)
            torch.cuda.synchronize()
        
        with record_function("pytorch_matmul"):
            for _ in range(10):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
    
    # Print summary
    print("\n--- CPU/CUDA Time Summary ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Print kernel-level details
    print("\n--- CUDA Kernel Summary ---")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=30))
    
    # Export Chrome trace
    trace_file = os.path.join(output_dir, f"trace_streamk_{M}x{K}x{N}_k{k_splits}.json")
    prof.export_chrome_trace(trace_file)
    print(f"\nChrome trace exported to: {trace_file}")
    print("Open chrome://tracing in Chrome browser to view")
    
    return prof


def profile_memory(A, B, k_splits=4):
    """Profile memory usage"""
    M, K = A.shape
    _, N = B.shape
    
    print(f"\n{'='*70}")
    print(f"Memory Profile: A[{M}, {K}] @ B[{K}, {N}]")
    print(f"{'='*70}")
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Measure StreamK memory
    mem_before = torch.cuda.memory_allocated()
    C_streamk = ek.gemm_streamk(A, B, k_splits=k_splits)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak_streamk = torch.cuda.max_memory_allocated()
    
    print(f"\nStreamK (k={k_splits}):")
    print(f"  Memory before: {mem_before / 1024**2:.2f} MB")
    print(f"  Memory after:  {mem_after / 1024**2:.2f} MB")
    print(f"  Peak memory:   {peak_streamk / 1024**2:.2f} MB")
    
    del C_streamk
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Measure Split-K memory
    mem_before = torch.cuda.memory_allocated()
    C_splitk = ek.gemm_splitk(A, B, k_splits=k_splits)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak_splitk = torch.cuda.max_memory_allocated()
    
    print(f"\nSplit-K (k={k_splits}):")
    print(f"  Memory before: {mem_before / 1024**2:.2f} MB")
    print(f"  Memory after:  {mem_after / 1024**2:.2f} MB")
    print(f"  Peak memory:   {peak_splitk / 1024**2:.2f} MB")
    print(f"  (Note: Split-K uses workspace of size [{k_splits}, {M}, {N}])")


def run_nsight_mode():
    """Minimal run for Nsight profiling"""
    print("Running in Nsight mode (minimal iterations for external profiler)...")
    
    # Test cases
    test_cases = [
        (512, 4096, 512),    # Large K
        (1024, 1024, 1024),  # Square
        (64, 8192, 64),      # Very large K, small M/N
    ]
    
    for M, K, N in test_cases:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        warmup(A, B, iterations=3)
        
        # Run each kernel a few times for profiler to capture
        torch.cuda.nvtx.range_push(f"streamk_{M}x{K}x{N}")
        for _ in range(5):
            C = ek.gemm_streamk(A, B, k_splits=4)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"splitk_{M}x{K}x{N}")
        for _ in range(5):
            C = ek.gemm_splitk(A, B, k_splits=4)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        
        torch.cuda.nvtx.range_push(f"pytorch_{M}x{K}x{N}")
        for _ in range(5):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser(description="Profile gemm_streamk kernel")
    parser.add_argument("--trace", action="store_true", help="Generate Chrome trace")
    parser.add_argument("--nsight", action="store_true", help="Run in Nsight profiler mode")
    parser.add_argument("--memory", action="store_true", help="Profile memory usage")
    parser.add_argument("-M", type=int, default=512, help="Matrix M dimension")
    parser.add_argument("-K", type=int, default=4096, help="Matrix K dimension")
    parser.add_argument("-N", type=int, default=512, help="Matrix N dimension")
    parser.add_argument("--k-splits", type=int, default=4, help="K splits for StreamK")
    parser.add_argument("--output-dir", type=str, default="profiler_traces", 
                        help="Output directory for traces")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return
    
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"efficient_kernels Version: {ek.__version__}")
    
    if args.nsight:
        run_nsight_mode()
        return
    
    # Create test matrices
    M, K, N = args.M, args.K, args.N
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    print("\nWarming up...")
    warmup(A, B)
    
    # Run benchmarks
    benchmark_streamk(A, B, k_splits_list=[1, 2, 4, 8])
    
    # Memory profiling
    if args.memory:
        profile_memory(A, B, k_splits=args.k_splits)
    
    # PyTorch profiler with trace
    if args.trace:
        profile_with_pytorch_profiler(A, B, k_splits=args.k_splits, 
                                       output_dir=args.output_dir)
    
    # Test different matrix sizes
    print(f"\n{'='*70}")
    print("Testing different matrix sizes")
    print(f"{'='*70}")
    
    test_cases = [
        (256, 256, 256),     # Small square
        (1024, 1024, 1024),  # Medium square
        (64, 8192, 64),      # Very large K, small M/N (good for StreamK)
        (2048, 512, 2048),   # Large M/N, small K
    ]
    
    for M, K, N in test_cases:
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        warmup(A, B, iterations=5)
        benchmark_streamk(A, B, k_splits_list=[2, 4])


if __name__ == "__main__":
    main()
