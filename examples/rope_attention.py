"""
RoPE (Rotary Position Embedding) Attention with KV Cache

This example demonstrates:
1. RoPE core components for position encoding
2. Causal Self-Attention with KV Cache support
3. Prefill and Decode stage simulation for LLM inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# 1. RoPE 核心组件
# -----------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 RoPE 的旋转角度 (Frequency Tensor)。
    生成复数形式或 Sin/Cos 表。这里简化为直接生成 freq。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # [end, dim//2]
    return freqs

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    对 Q 和 K 应用 RoPE 旋转。
    xq, xk shape: [batch_size, seq_len, num_heads, head_dim]
    freqs_cis shape: [seq_len, head_dim/2]
    """
    # 将 Q, K 重塑为成对形式，以便进行复数或旋转矩阵计算
    # 这里使用实数域的旋转逻辑：
    # x = [x1, x2] -> [-x2, x1] * sin + [x1, x2] * cos
    
    # 1. 获取对应的 cos 和 sin
    # freqs_cis 的形状通常需要广播以匹配 batch 和 head 维度
    # 假设 freqs_cis 已经是 [1, seq_len, 1, head_dim/2] 这种可广播形状
    
    # 为了演示清晰，我们手动做 repeat 使得维度匹配 (在真实库中会用广播)
    # freqs shape: [seq_len, head_dim/2] -> 扩展为 cos/sin [batch, seq, head, head_dim]
    
    # 简单起见，我们假设传入的 freqs_cis 包含了 cos 和 sin
    # 这里复用 Llama 的 rotate_half 实现风格
    
    def rotate_half(x):
        """旋转向量的一半：[-x2, x1]"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # 这里的 freqs_cos/sin 维度需要是 [1, seq_len, 1, head_dim]
    freqs_cos, freqs_sin = freqs_cis
    
    # 核心公式：
    # output = (x * cos) + (rotate_half(x) * sin)
    xq_out = (xq * freqs_cos) + (rotate_half(xq) * freqs_sin)
    xk_out = (xk * freqs_cos) + (rotate_half(xk) * freqs_sin)
    
    return xq_out, xk_out

# -----------------------------------------------------------------------------
# 2. 带有 KV Cache 和 RoPE 的 Attention 模块
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # 线性投影
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        # 预计算 RoPE 的 cos/sin 表 (缓存起来，避免重复计算)
        freqs = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def forward(self, x, kv_cache=None, start_pos=0):
        """
        x: [batch, seq_len, d_model] -> 当前输入的 token embedding
        kv_cache: Tuple(past_k, past_v) -> 历史缓存
        start_pos: int -> 当前 x 在整个序列中的起始位置 (用于 RoPE)
        """
        b, seq_len, _ = x.shape
        
        # 1. 投影 Q, K, V
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Reshape 为 [batch, seq_len, n_head, head_dim]
        xq = xq.view(b, seq_len, self.n_head, self.head_dim)
        xk = xk.view(b, seq_len, self.n_head, self.head_dim)
        xv = xv.view(b, seq_len, self.n_head, self.head_dim)
        
        # 2. 准备 RoPE 所需的频率
        # 根据 start_pos 切片取出当前 token 对应的旋转角度
        # end_pos = start_pos + seq_len
        cos = self.freqs_cos[start_pos : start_pos + seq_len]
        sin = self.freqs_sin[start_pos : start_pos + seq_len]
        
        # 调整维度以支持广播 [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim]
        # 注意：RoPE通常作用于 head_dim 维度，sin/cos 需要 repeat 两次匹配 head_dim
        # (这步通常会有优化，这里为了逻辑清晰显式写出)
        cos = torch.cat([cos, cos], dim=-1).view(1, seq_len, 1, self.head_dim)
        sin = torch.cat([sin, sin], dim=-1).view(1, seq_len, 1, self.head_dim)
        
        # 3. 应用 RoPE (只旋转当前的 Q 和 K)
        # !!! 关键点：我们不需要重新旋转 Cache 里的旧 K，因为它们的绝对位置没变 !!!
        xq, xk = apply_rotary_emb(xq, xk, (cos, sin))
        
        # 4. KV Cache 管理
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # 将新的(已旋转的) K 和新的 V 拼接到 Cache 后面
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)
            
        # 更新后的 Cache (返回给外部循环使用)
        current_cache = (xk, xv)
        
        # 5. 计算 Attention
        # Q shape: [b, seq_len(新), n_head, head_dim]
        # K shape: [b, seq_len(总), n_head, head_dim]
        
        # 转置用于点积: [b, n_head, seq_len, head_dim]
        xq = xq.transpose(1, 2) 
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Scores: [b, n_head, seq_len(新), seq_len(总)]
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Masking (因果掩码)
        # 如果是 Prefill 阶段 (seq_len > 1)，需要 mask 掉未来的 token
        # 如果是 Decode 阶段 (seq_len == 1)，通常不需要 mask，因为只能看到过去所有的
        if seq_len > 1:
            # 创建一个 mask: [1, 1, seq_len(新), seq_len(总)]
            mask = torch.triu(torch.ones(seq_len, xk.shape[2]), diagonal=start_pos+1).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        
        # Output: [b, n_head, seq_len(新), head_dim]
        output = torch.matmul(attn_weights, xv)
        
        # 还原形状
        output = output.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.wo(output), current_cache

# -----------------------------------------------------------------------------
# 3. 模拟推理过程：Prefill -> Decode
# -----------------------------------------------------------------------------

def demo_inference(use_compile: bool = False):
    torch.manual_seed(42)
    model = CausalSelfAttention(d_model=64, n_head=4)
    model.eval()
    
    # 使用 torch.compile 加速 (PyTorch 2.0+)
    if use_compile:
        print("Using torch.compile for optimization...")
        # mode 可选: "default", "reduce-overhead", "max-autotune"
        # fullgraph=False 允许图断裂，对于动态 KV Cache 更友好
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    # 模拟输入：Batch=1, SeqLen=10 (Prefill 阶段)
    prefill_input = torch.randn(1, 10, 64)
    
    print("-" * 30)
    print("1. Prefill Stage (Length 10)")
    
    # 第一次前向传播：没有 Cache，start_pos=0
    with torch.no_grad():
        output_prefill, kv_cache = model(prefill_input, kv_cache=None, start_pos=0)
    
    print(f"Output shape: {output_prefill.shape}")
    print(f"Cache K shape: {kv_cache[0].shape}") # 应该是 [1, 10, 4, 16]
    
    # 模拟生成：Decode 阶段 (生成 5 个新 token)
    print("\n2. Decode Stage (Generating 5 tokens)")
    
    # 这里的 next_token 通常是采样出来的，这里随机生成代替
    next_token_emb = torch.randn(1, 1, 64) 
    
    # 循环 5 次
    for i in range(5):
        # 当前 Cache 长度就是新 token 的起始位置
        current_pos = kv_cache[0].shape[1] 
        
        with torch.no_grad():
            # 传入 cache，start_pos 设为 cache 的长度
            output_token, kv_cache = model(next_token_emb, kv_cache=kv_cache, start_pos=current_pos)
            
        print(f"Step {i+1}: New pos {current_pos}, Cache len becomes {kv_cache[0].shape[1]}")
        # 这里实际上会把 output_token 转为 id 然后 embedding 传入下一次...
        next_token_emb = torch.randn(1, 1, 64)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()
    
    demo_inference(use_compile=args.compile)
