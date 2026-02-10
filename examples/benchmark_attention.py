"""
Benchmark: RoPE Attention Performance Comparison

Compares:
1. Pure PyTorch implementation (baseline)
2. torch.compile optimized version
3. FlashAttention (if available)
4. Custom fused kernel (placeholder for future CUDA implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import time
from torch.profiler import profile, record_function, ProfilerActivity

# =============================================================================
# 1. Pure PyTorch Implementation (Baseline)
# =============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device='cuda'):
    """预计算 RoPE 的旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    return freqs


def apply_rotary_emb_pytorch(xq, xk, cos, sin):
    """Pure PyTorch RoPE implementation"""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


class RoPEAttentionPyTorch(nn.Module):
    """Pure PyTorch implementation - Baseline"""
    
    def __init__(self, d_model, n_head, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        freqs = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def forward(self, x, kv_cache=None, start_pos=0):
        b, seq_len, _ = x.shape
        
        xq = self.wq(x).view(b, seq_len, self.n_head, self.head_dim)
        xk = self.wk(x).view(b, seq_len, self.n_head, self.head_dim)
        xv = self.wv(x).view(b, seq_len, self.n_head, self.head_dim)
        
        cos = self.freqs_cos[start_pos : start_pos + seq_len]
        sin = self.freqs_sin[start_pos : start_pos + seq_len]
        cos = torch.cat([cos, cos], dim=-1).view(1, seq_len, 1, self.head_dim)
        sin = torch.cat([sin, sin], dim=-1).view(1, seq_len, 1, self.head_dim)
        
        xq, xk = apply_rotary_emb_pytorch(xq, xk, cos, sin)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
        
        current_cache = (xk, xv)
        
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, xk.shape[2], device=x.device), diagonal=start_pos+1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, xv)
        output = output.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        
        return self.wo(output), current_cache


# =============================================================================
# 2. Fused RoPE Implementation (optimized for fewer kernel launches)
# =============================================================================

class RoPEAttentionFused(nn.Module):
    """Optimized version with fused operations"""
    
    def __init__(self, d_model, n_head, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Fused QKV projection
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        freqs = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def forward(self, x, kv_cache=None, start_pos=0):
        b, seq_len, _ = x.shape
        
        # Fused QKV projection (one kernel instead of three)
        qkv = self.wqkv(x)
        qkv = qkv.view(b, seq_len, 3, self.n_head, self.head_dim)
        xq, xk, xv = qkv.unbind(dim=2)
        
        # Get RoPE embeddings
        cos = self.freqs_cos[start_pos : start_pos + seq_len]
        sin = self.freqs_sin[start_pos : start_pos + seq_len]
        cos = torch.cat([cos, cos], dim=-1).view(1, seq_len, 1, self.head_dim)
        sin = torch.cat([sin, sin], dim=-1).view(1, seq_len, 1, self.head_dim)
        
        # Fused RoPE: use in-place operations where possible
        xq, xk = apply_rotary_emb_pytorch(xq, xk, cos, sin)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
        
        current_cache = (xk, xv)
        
        # Use scaled_dot_product_attention (PyTorch 2.0+ native FlashAttention)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # PyTorch 2.0+ SDPA with automatic FlashAttention backend selection
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(seq_len > 1 and kv_cache is None),  # Only causal for prefill without cache
            scale=self.scale
        )
        
        output = output.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.wo(output), current_cache


# =============================================================================
# 3. Benchmark Utilities
# =============================================================================

def benchmark_prefill(model, batch_size, seq_len, d_model, warmup=10, iterations=100):
    """Benchmark prefill stage (processing initial sequence)"""
    device = next(model.parameters()).device
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _ = model(x, kv_cache=None, start_pos=0)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _, _ = model(x, kv_cache=None, start_pos=0)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations


def benchmark_decode(model, batch_size, cache_len, d_model, n_head, warmup=10, iterations=100):
    """Benchmark decode stage (single token generation with KV cache)"""
    device = next(model.parameters()).device
    head_dim = d_model // n_head
    
    # Pre-create KV cache
    past_k = torch.randn(batch_size, cache_len, n_head, head_dim, device=device, dtype=torch.float16)
    past_v = torch.randn(batch_size, cache_len, n_head, head_dim, device=device, dtype=torch.float16)
    kv_cache = (past_k, past_v)
    
    # Single token input
    x = torch.randn(batch_size, 1, d_model, device=device, dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _ = model(x, kv_cache=kv_cache, start_pos=cache_len)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _, _ = model(x, kv_cache=kv_cache, start_pos=cache_len)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iterations


def verify_correctness(model1, model2, batch_size, seq_len, d_model):
    """Verify two models produce similar outputs"""
    device = next(model1.parameters()).device
    dtype = next(model1.parameters()).dtype
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    
    with torch.no_grad():
        out1, _ = model1(x, kv_cache=None, start_pos=0)
        out2, _ = model2(x, kv_cache=None, start_pos=0)
    
    max_diff = torch.max(torch.abs(out1 - out2)).item()
    mean_diff = torch.mean(torch.abs(out1 - out2)).item()
    
    return max_diff, mean_diff


# =============================================================================
# 4. Detailed Profiling with torch.profiler
# =============================================================================

def profile_model(model, x, kv_cache, start_pos, model_name: str, warmup=5, active=3):
    """
    Profile a model using torch.profiler to get detailed CUDA metrics.
    
    Returns profiler results with:
    - CUDA time
    - CPU time  
    - Memory operations (DRAM read/write)
    - Kernel-level breakdown
    """
    device = next(model.parameters()).device
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _ = model(x, kv_cache=kv_cache, start_pos=start_pos)
    torch.cuda.synchronize()
    
    # Profile with CUDA activities
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        with record_function(f"{model_name}_forward"):
            for _ in range(active):
                with torch.no_grad():
                    _, _ = model(x, kv_cache=kv_cache, start_pos=start_pos)
                torch.cuda.synchronize()
    
    return prof


def print_profiler_summary(prof, model_name: str, num_runs: int = 3):
    """Print formatted profiler summary"""
    print(f"\n  [{model_name}] Kernel Breakdown (top 10):")
    print("-" * 80)
    
    # Get key averages sorted by CUDA time
    key_averages = prof.key_averages()
    
    # Print table header
    print(f"  {'Kernel Name':<45} {'CUDA Time':>12} {'CPU Time':>12} {'Calls':>6}")
    print("  " + "-" * 77)
    
    # Helper to get cuda time (handle different PyTorch versions)
    def get_cuda_time(event):
        # Try different attribute names for CUDA time
        for attr in ['cuda_time_total', 'self_cuda_time_total', 'device_time_total']:
            if hasattr(event, attr):
                return getattr(event, attr)
        return 0
    
    # Sort by CUDA time and print top 10
    sorted_events = sorted(key_averages, key=get_cuda_time, reverse=True)[:10]
    
    for event in sorted_events:
        cuda_time = get_cuda_time(event) / 1000 / num_runs  # Convert to ms per run
        cpu_time = event.cpu_time_total / 1000 / num_runs if hasattr(event, 'cpu_time_total') else 0
        calls = event.count // num_runs if hasattr(event, 'count') else 0
        name = event.key[:44] if len(event.key) > 44 else event.key
        print(f"  {name:<45} {cuda_time:>10.3f}ms {cpu_time:>10.3f}ms {calls:>6}")
    
    print("-" * 80)
    
    # Calculate totals
    total_cuda_time = sum(get_cuda_time(e) for e in key_averages) / 1000 / num_runs
    total_cpu_time = sum(e.cpu_time_total for e in key_averages if hasattr(e, 'cpu_time_total')) / 1000 / num_runs
    
    print(f"  Total CUDA Time: {total_cuda_time:.3f} ms | Total CPU Time: {total_cpu_time:.3f} ms")


def print_memory_summary(prof, model_name: str):
    """Print memory usage summary from profiler"""
    print(f"\n  [{model_name}] Memory Summary:")
    print("-" * 60)
    
    key_averages = prof.key_averages()
    
    # Collect memory stats
    total_cuda_mem = 0
    total_self_cuda_mem = 0
    
    for event in key_averages:
        if hasattr(event, 'cuda_memory_usage'):
            total_cuda_mem += event.cuda_memory_usage
        if hasattr(event, 'self_cuda_memory_usage'):
            total_self_cuda_mem += event.self_cuda_memory_usage
    
    # Convert to MB
    print(f"  Total CUDA Memory Allocated: {total_cuda_mem / 1024 / 1024:.2f} MB")
    print(f"  Self CUDA Memory: {total_self_cuda_mem / 1024 / 1024:.2f} MB")


def estimate_memory_bandwidth(d_model, n_head, seq_len, batch_size, time_ms, dtype=torch.float16):
    """
    Estimate memory bandwidth utilization for attention.
    
    Theoretical memory access for attention:
    - Q, K, V read: 3 * batch * seq * d_model * sizeof(dtype)
    - Attention scores: batch * n_head * seq * seq * sizeof(dtype) (write + read)
    - Output: batch * seq * d_model * sizeof(dtype)
    """
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    
    # QKV reads
    qkv_bytes = 3 * batch_size * seq_len * d_model * bytes_per_elem
    
    # Attention matrix (write scores + read for softmax + read for matmul with V)
    attn_bytes = 3 * batch_size * n_head * seq_len * seq_len * bytes_per_elem
    
    # Output write
    output_bytes = batch_size * seq_len * d_model * bytes_per_elem
    
    # Weight reads (approximate - QKV + output projection)
    weight_bytes = (3 * d_model * d_model + d_model * d_model) * bytes_per_elem
    
    total_bytes = qkv_bytes + attn_bytes + output_bytes + weight_bytes
    total_gb = total_bytes / 1e9
    
    # Calculate bandwidth (GB/s)
    time_s = time_ms / 1000
    bandwidth_gbs = total_gb / time_s if time_s > 0 else 0
    
    return {
        'total_bytes': total_bytes,
        'total_gb': total_gb,
        'bandwidth_gbs': bandwidth_gbs,
        'qkv_bytes': qkv_bytes,
        'attn_bytes': attn_bytes,
    }


def detailed_profile(model_pytorch, model_fused, batch_size, seq_len, d_model, n_head, export_trace=True):
    """Run detailed profiling comparison between models"""
    device = next(model_pytorch.parameters()).device
    dtype = next(model_pytorch.parameters()).dtype
    
    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype)
    
    print("\n" + "=" * 80)
    print("DETAILED PROFILING (torch.profiler)")
    print("=" * 80)
    
    # Profile PyTorch baseline
    prof_pytorch = profile_model(model_pytorch, x, None, 0, "PyTorch")
    print_profiler_summary(prof_pytorch, "PyTorch Baseline")
    print_memory_summary(prof_pytorch, "PyTorch Baseline")
    
    # Profile Fused + SDPA
    prof_fused = profile_model(model_fused, x, None, 0, "Fused+SDPA")
    print_profiler_summary(prof_fused, "Fused + SDPA")
    print_memory_summary(prof_fused, "Fused + SDPA")
    
    # Estimate memory bandwidth
    print("\n" + "-" * 60)
    print("Memory Bandwidth Estimation:")
    print("-" * 60)
    
    # Helper to get cuda time (handle different PyTorch versions)
    def get_cuda_time(event):
        for attr in ['cuda_time_total', 'self_cuda_time_total', 'device_time_total']:
            if hasattr(event, attr):
                return getattr(event, attr)
        return 0
    
    # Get average time from profiler
    pytorch_time = sum(get_cuda_time(e) for e in prof_pytorch.key_averages()) / 1000 / 3
    fused_time = sum(get_cuda_time(e) for e in prof_fused.key_averages()) / 1000 / 3
    
    bw_pytorch = estimate_memory_bandwidth(d_model, n_head, seq_len, batch_size, pytorch_time, dtype)
    bw_fused = estimate_memory_bandwidth(d_model, n_head, seq_len, batch_size, fused_time, dtype)
    
    print(f"  Theoretical Data Movement: {bw_pytorch['total_gb']*1000:.2f} MB")
    print(f"  PyTorch Baseline: {bw_pytorch['bandwidth_gbs']:.1f} GB/s (effective)")
    print(f"  Fused + SDPA:     {bw_fused['bandwidth_gbs']:.1f} GB/s (effective)")
    
    # Export traces for Perfetto/Chrome viewing
    if export_trace:
        import os
        trace_dir = "profiler_traces"
        os.makedirs(trace_dir, exist_ok=True)
        
        trace_pytorch = f"{trace_dir}/trace_pytorch_{d_model}d_{seq_len}seq.json"
        trace_fused = f"{trace_dir}/trace_fused_{d_model}d_{seq_len}seq.json"
        
        try:
            prof_pytorch.export_chrome_trace(trace_pytorch)
            prof_fused.export_chrome_trace(trace_fused)
            
            print("\n" + "-" * 60)
            print("Trace Files Exported:")
            print("-" * 60)
            print(f"  PyTorch: {trace_pytorch}")
            print(f"  Fused:   {trace_fused}")
            print("\n  View traces in:")
            print("    - Perfetto: https://ui.perfetto.dev (drag & drop JSON file)")
            print("    - Chrome:   chrome://tracing (Load JSON file)")
        except Exception as e:
            print(f"\n  (Chrome trace export failed: {e})")
    
    return prof_pytorch, prof_fused


# =============================================================================
# 5. Main Benchmark
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RoPE Attention Benchmark")
    parser.add_argument("--profile", action="store_true", help="Run detailed torch.profiler analysis")
    parser.add_argument("--config", type=str, default="all", 
                        choices=["small", "medium", "large", "xlarge", "all"],
                        help="Which config to benchmark")
    parser.add_argument("--export-trace", action="store_true", help="Export Chrome traces")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RoPE Attention Benchmark")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires a CUDA-enabled GPU.")
        return
    
    # Configuration
    device = 'cuda'
    dtype = torch.float16
    
    # Model configurations to test
    all_configs = {
        "small": (512, 8, 1, 512, "Small (512d, 8h, bs=1, seq=512)"),
        "medium": (1024, 16, 1, 1024, "Medium (1024d, 16h, bs=1, seq=1024)"),
        "large": (2048, 32, 1, 2048, "Large (2048d, 32h, bs=1, seq=2048)"),
        "xlarge": (4096, 32, 1, 2048, "XLarge (4096d, 32h, bs=1, seq=2048)"),
    }
    
    if args.config == "all":
        configs = list(all_configs.values())
    else:
        configs = [all_configs[args.config]]
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using dtype: {dtype}")
    print(f"Detailed profiling: {'ON' if args.profile else 'OFF'}")
    
    for d_model, n_head, batch_size, seq_len, config_name in configs:
        print("\n" + "-" * 70)
        print(f"Config: {config_name}")
        print("-" * 70)
        
        # Create models
        model_pytorch = RoPEAttentionPyTorch(d_model, n_head).to(device).to(dtype).eval()
        model_fused = RoPEAttentionFused(d_model, n_head).to(device).to(dtype).eval()
        
        # Copy weights for fair comparison
        with torch.no_grad():
            # Split fused QKV weights
            qkv_weight = model_fused.wqkv.weight.data
            model_pytorch.wq.weight.data.copy_(qkv_weight[:d_model])
            model_pytorch.wk.weight.data.copy_(qkv_weight[d_model:2*d_model])
            model_pytorch.wv.weight.data.copy_(qkv_weight[2*d_model:])
            model_pytorch.wo.weight.data.copy_(model_fused.wo.weight.data)
        
        # Create compiled version
        model_compiled = torch.compile(model_pytorch, mode="reduce-overhead", fullgraph=False)
        
        # Benchmark Prefill
        print("\n[Prefill Stage]")
        time_pytorch = benchmark_prefill(model_pytorch, batch_size, seq_len, d_model)
        time_fused = benchmark_prefill(model_fused, batch_size, seq_len, d_model)
        
        # torch.compile needs extra warmup for first compilation
        _ = benchmark_prefill(model_compiled, batch_size, seq_len, d_model, warmup=20, iterations=10)
        time_compiled = benchmark_prefill(model_compiled, batch_size, seq_len, d_model)
        
        print(f"  PyTorch (baseline):     {time_pytorch:.4f} ms")
        print(f"  Fused + SDPA:           {time_fused:.4f} ms  ({time_pytorch/time_fused:.2f}x)")
        print(f"  torch.compile:          {time_compiled:.4f} ms  ({time_pytorch/time_compiled:.2f}x)")
        
        # Benchmark Decode
        cache_len = seq_len
        print(f"\n[Decode Stage] (cache_len={cache_len})")
        time_pytorch_dec = benchmark_decode(model_pytorch, batch_size, cache_len, d_model, n_head)
        time_fused_dec = benchmark_decode(model_fused, batch_size, cache_len, d_model, n_head)
        time_compiled_dec = benchmark_decode(model_compiled, batch_size, cache_len, d_model, n_head)
        
        print(f"  PyTorch (baseline):     {time_pytorch_dec:.4f} ms")
        print(f"  Fused + SDPA:           {time_fused_dec:.4f} ms  ({time_pytorch_dec/time_fused_dec:.2f}x)")
        print(f"  torch.compile:          {time_compiled_dec:.4f} ms  ({time_pytorch_dec/time_compiled_dec:.2f}x)")
        
        # Detailed profiling if requested
        if args.profile:
            detailed_profile(model_pytorch, model_fused, batch_size, seq_len, d_model, n_head, 
                           export_trace=args.export_trace)
        
        # Verify correctness (use full precision for verification)
        model_pytorch_fp32 = RoPEAttentionPyTorch(d_model, n_head).to(device).eval()
        model_fused_fp32 = RoPEAttentionFused(d_model, n_head).to(device).eval()
        
        # Copy weights
        with torch.no_grad():
            qkv_weight = model_fused_fp32.wqkv.weight.data
            model_pytorch_fp32.wq.weight.data.copy_(qkv_weight[:d_model])
            model_pytorch_fp32.wk.weight.data.copy_(qkv_weight[d_model:2*d_model])
            model_pytorch_fp32.wv.weight.data.copy_(qkv_weight[2*d_model:])
            model_pytorch_fp32.wo.weight.data.copy_(model_fused_fp32.wo.weight.data)
        
        max_diff, mean_diff = verify_correctness(model_pytorch_fp32, model_fused_fp32, 1, 64, d_model)
        print(f"\n[Correctness Check] Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        
        # Cleanup
        del model_pytorch, model_fused, model_compiled
        del model_pytorch_fp32, model_fused_fp32
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print("\nNotes:")
    print("- 'Fused + SDPA' uses fused QKV projection and F.scaled_dot_product_attention")
    print("- SDPA automatically selects FlashAttention/MemEfficient backend when available")
    print("- torch.compile uses Inductor backend for graph optimization")
    print("- Use --profile for detailed kernel breakdown and memory analysis")
    print("- For production, consider custom CUDA kernels for RoPE computation")


if __name__ == "__main__":
    main()
