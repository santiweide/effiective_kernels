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
        
        # Linear Projection
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        freqs = precompute_freqs_cis(self.head_dim, max_seq_len * 2)
        self.register_buffer("freqs_cos", torch.cos(freqs), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(freqs), persistent=False)

    def forward(self, x, kv_cache=None, start_pos=0):

        b, seq_len, _ = x.shape
        xq = self.wq(x)
        #         
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(b, seq_len, self.n_head, self.head_dim)
        xk = xk.view(b, seq_len, self.n_head, self.head_dim)
        xv = xv.view(b, seq_len, self.n_head, self.head_dim)
        
        cos = self.freqs_cos[start_pos : start_pos + seq_len]
        sin = self.freqs_sin[start_pos : start_pos + seq_len]
        
        cos = torch.cat([cos, cos], dim=-1).view(1, seq_len, 1, self.head_dim)
        sin = torch.cat([sin, sin], dim=-1).view(1, seq_len, 1, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, (cos, sin))
        
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
            mask = torch.triu(torch.ones(seq_len, xk.shape[2]), diagonal=start_pos+1).bool()
            mask = mask.to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_weights, xv)
        
        output = output.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.wo(output), current_cache

# -----------------------------------------------------------------------------
# 3. Feed-Forward Network (FFN/MLP)
# -----------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation (Llama style)
    
    SwiGLU: output = (xW1 * silu(xW_gate)) @ W2
    相比传统 FFN (xW1 -> ReLU -> W2)，SwiGLU 效果更好
    """
    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        # 默认 hidden_dim = 4 * d_model (或 8/3 * d_model for SwiGLU to match param count)
        if hidden_dim is None:
            hidden_dim = int(8 / 3 * d_model)
            # 向上取整到 256 的倍数 (为了更好的硬件效率)
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: (x @ W1) * silu(x @ W3) @ W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    比 LayerNorm 更简单高效，Llama 等模型使用
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # 计算 RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# -----------------------------------------------------------------------------
# 4. Transformer Block (完整的 Decoder Layer)
# -----------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    完整的 Transformer Decoder Block
    
    结构 (Pre-LN):
        x -> RMSNorm -> Attention -> + -> RMSNorm -> FFN -> +
             |________________________|    |_______________|
                   residual                    residual
    """
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        max_seq_len: int = 2048,
        ffn_hidden_dim: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Attention
        self.attention = CausalSelfAttention(d_model, n_head, max_seq_len)
        
        # FFN
        self.feed_forward = FeedForward(d_model, ffn_hidden_dim, dropout)
        
        # Normalization (Pre-LN style)
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: tuple = None, 
        start_pos: int = 0
    ) -> tuple:
        """
        Args:
            x: [batch, seq_len, d_model]
            kv_cache: (past_k, past_v) 或 None
            start_pos: 当前 token 的起始位置
            
        Returns:
            output: [batch, seq_len, d_model]
            new_kv_cache: (k, v)
        """
        # 1. Attention with residual
        h = x + self._attention_block(self.attention_norm(x), kv_cache, start_pos)[0]
        
        # 获取更新后的 KV Cache
        _, new_kv_cache = self._attention_block(self.attention_norm(x), kv_cache, start_pos)
        
        # 2. FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out, new_kv_cache
    
    def _attention_block(self, x, kv_cache, start_pos):
        return self.attention(x, kv_cache, start_pos)
    
    def prefill(self, x: torch.Tensor) -> tuple:
        """
        Prefill 阶段：处理完整的 prompt
        
        特点:
        - 一次性处理多个 token
        - 需要 causal mask
        - 初始化 KV Cache
        
        Args:
            x: [batch, prompt_len, d_model]
        Returns:
            output: [batch, prompt_len, d_model]
            kv_cache: (k, v) for future decoding
        """
        return self.forward(x, kv_cache=None, start_pos=0)
    
    def decode(
        self, 
        x: torch.Tensor, 
        kv_cache: tuple, 
        start_pos: int
    ) -> tuple:
        """
        Decode 阶段：逐 token 生成
        
        特点:
        - 每次只处理 1 个 token
        - 无需 causal mask (只看过去)
        - 复用并更新 KV Cache
        
        Args:
            x: [batch, 1, d_model] - 单个 token
            kv_cache: (past_k, past_v) - 历史缓存
            start_pos: 当前 token 的位置
        Returns:
            output: [batch, 1, d_model]
            new_kv_cache: 更新后的缓存
        """
        assert x.shape[1] == 1, "Decode stage expects single token input"
        return self.forward(x, kv_cache, start_pos)


# -----------------------------------------------------------------------------
# 5. 多层 Transformer (简化版 LLM Backbone)
# -----------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    多层 Transformer Decoder (类似 Llama 的结构)
    
    用于演示完整的 prefill/decode 流程
    """
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_head: int,
        max_seq_len: int = 2048,
        ffn_hidden_dim: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, max_seq_len, ffn_hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        kv_caches: list = None,
        start_pos: int = 0
    ) -> tuple:
        """
        Args:
            x: [batch, seq_len, d_model]
            kv_caches: List of (k, v) tuples for each layer, or None
            start_pos: position for RoPE
            
        Returns:
            output: [batch, seq_len, d_model]
            new_kv_caches: List of updated (k, v) tuples
        """
        if kv_caches is None:
            kv_caches = [None] * self.n_layers
        
        new_kv_caches = []
        
        h = x
        for i, layer in enumerate(self.layers):
            h, new_kv = layer(h, kv_caches[i], start_pos)
            new_kv_caches.append(new_kv)
        
        return self.norm(h), new_kv_caches
    
    def prefill(self, x: torch.Tensor) -> tuple:
        """Prefill 阶段"""
        return self.forward(x, kv_caches=None, start_pos=0)
    
    def decode(
        self, 
        x: torch.Tensor, 
        kv_caches: list, 
        start_pos: int
    ) -> tuple:
        """Decode 阶段"""
        return self.forward(x, kv_caches, start_pos)


# -----------------------------------------------------------------------------
# 6. Prompt + Query 输入结构
# -----------------------------------------------------------------------------

class PromptQueryInput:
    """
    区分 Prompt 和 Query 的输入结构
    
    Prompt: 系统提示/上下文 (可以预先缓存 KV Cache)
    Query:  用户输入的问题
    
    典型场景：
    - Prompt: "You are a helpful assistant. Answer questions concisely."
    - Query:  "What is the capital of France?"
    """
    def __init__(self, prompt_emb: torch.Tensor, query_emb: torch.Tensor):
        """
        Args:
            prompt_emb: [batch, prompt_len, d_model] - 系统提示的 embedding
            query_emb:  [batch, query_len, d_model]  - 用户问题的 embedding
        """
        self.prompt_emb = prompt_emb
        self.query_emb = query_emb
        self.prompt_len = prompt_emb.shape[1]
        self.query_len = query_emb.shape[1]
        self.total_len = self.prompt_len + self.query_len
    
    def get_combined(self) -> torch.Tensor:
        """返回拼接后的完整输入 [batch, prompt_len + query_len, d_model]"""
        return torch.cat([self.prompt_emb, self.query_emb], dim=1)
    
    def __repr__(self):
        return (f"PromptQueryInput(prompt_len={self.prompt_len}, "
                f"query_len={self.query_len}, total_len={self.total_len})")


class PromptCache:
    """
    Prompt KV Cache 管理器
    
    用于缓存固定 Prompt 的 KV，避免重复计算
    适用于同一 Prompt 多次使用的场景（如多轮对话中的系统提示）
    """
    def __init__(self):
        self.cached_kv = None
        self.prompt_len = 0
        self.prompt_hash = None  # 用于验证 prompt 是否变化
    
    def cache_prompt(self, kv_cache: tuple, prompt_len: int, prompt_hash: int = None):
        """缓存 Prompt 的 KV"""
        # 只缓存 prompt 部分的 KV
        k, v = kv_cache
        self.cached_kv = (k[:, :prompt_len].clone(), v[:, :prompt_len].clone())
        self.prompt_len = prompt_len
        self.prompt_hash = prompt_hash
        
    def get_cached_kv(self) -> tuple:
        """获取缓存的 Prompt KV"""
        return self.cached_kv
    
    def is_valid(self, prompt_hash: int = None) -> bool:
        """检查缓存是否有效"""
        if self.cached_kv is None:
            return False
        if prompt_hash is not None and self.prompt_hash != prompt_hash:
            return False
        return True
    
    def clear(self):
        """清除缓存"""
        self.cached_kv = None
        self.prompt_len = 0
        self.prompt_hash = None


# -----------------------------------------------------------------------------
# 4. 模拟推理过程：Prefill -> Decode (支持 Prompt + Query 区分)
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


def demo_prompt_query_inference(use_compile: bool = False):
    """
    演示 Prompt + Query 区分的推理流程
    
    场景：多轮对话，Prompt 固定，Query 变化
    优化：Prompt 的 KV Cache 可以复用
    """
    torch.manual_seed(42)
    model = CausalSelfAttention(d_model=64, n_head=4)
    model.eval()
    
    if use_compile:
        print("Using torch.compile for optimization...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    print("=" * 60)
    print("Prompt + Query 区分推理演示")
    print("=" * 60)
    
    # 创建 Prompt Cache 管理器
    prompt_cache = PromptCache()
    
    # 模拟固定的系统提示 (Prompt)
    # 实际中这可能是: "You are a helpful AI assistant..."
    prompt_len = 8
    prompt_emb = torch.randn(1, prompt_len, 64)
    prompt_hash = hash(prompt_emb.sum().item())  # 简单的 hash 用于验证
    
    print(f"\n[System Prompt] Length: {prompt_len} tokens")
    print("  (模拟: 'You are a helpful AI assistant...')")
    
    # ==========================================================================
    # 第一轮对话
    # ==========================================================================
    print("\n" + "-" * 40)
    print("第一轮对话")
    print("-" * 40)
    
    # 用户 Query 1
    query1_len = 5
    query1_emb = torch.randn(1, query1_len, 64)
    
    input1 = PromptQueryInput(prompt_emb, query1_emb)
    print(f"\n[User Query 1] Length: {query1_len} tokens")
    print(f"  Input: {input1}")
    
    # Prefill: 处理 Prompt + Query
    print("\n1. Prefill Stage (Prompt + Query)")
    with torch.no_grad():
        combined_input = input1.get_combined()
        output, kv_cache = model(combined_input, kv_cache=None, start_pos=0)
    
    print(f"  Combined input shape: {combined_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  KV Cache shape: K={kv_cache[0].shape}")
    
    # 缓存 Prompt 部分的 KV
    prompt_cache.cache_prompt(kv_cache, input1.prompt_len, prompt_hash)
    print(f"\n  [Cached] Prompt KV (len={prompt_cache.prompt_len})")
    
    # Decode: 生成回复
    print("\n2. Decode Stage (Generating response)")
    next_token_emb = torch.randn(1, 1, 64)
    for i in range(3):
        current_pos = kv_cache[0].shape[1]
        with torch.no_grad():
            output_token, kv_cache = model(next_token_emb, kv_cache=kv_cache, start_pos=current_pos)
        print(f"  Generated token {i+1}, Cache len: {kv_cache[0].shape[1]}")
        next_token_emb = torch.randn(1, 1, 64)
    
    # ==========================================================================
    # 第二轮对话 (复用 Prompt Cache)
    # ==========================================================================
    print("\n" + "-" * 40)
    print("第二轮对话 (复用 Prompt Cache)")
    print("-" * 40)
    
    # 用户 Query 2
    query2_len = 6
    query2_emb = torch.randn(1, query2_len, 64)
    
    input2 = PromptQueryInput(prompt_emb, query2_emb)
    print(f"\n[User Query 2] Length: {query2_len} tokens")
    print(f"  Input: {input2}")
    
    # 检查 Prompt Cache 是否有效
    if prompt_cache.is_valid(prompt_hash):
        print("\n  [Cache Hit] 复用 Prompt KV Cache!")
        
        # 只需要处理 Query 部分
        # 从 Prompt Cache 恢复 KV，然后处理新 Query
        cached_kv = prompt_cache.get_cached_kv()
        
        print(f"  Cached KV shape: K={cached_kv[0].shape}")
        
        # 只对 Query 做 Prefill，start_pos = prompt_len
        with torch.no_grad():
            output_query, kv_cache = model(
                query2_emb, 
                kv_cache=cached_kv, 
                start_pos=prompt_cache.prompt_len
            )
        
        print(f"  Only processed Query, Output shape: {output_query.shape}")
        print(f"  New KV Cache shape: K={kv_cache[0].shape}")
    else:
        print("\n  [Cache Miss] 需要重新计算 Prompt")
        # 回退到完整计算...
    
    # Decode: 生成回复
    print("\n2. Decode Stage (Generating response)")
    next_token_emb = torch.randn(1, 1, 64)
    for i in range(3):
        current_pos = kv_cache[0].shape[1]
        with torch.no_grad():
            output_token, kv_cache = model(next_token_emb, kv_cache=kv_cache, start_pos=current_pos)
        print(f"  Generated token {i+1}, Cache len: {kv_cache[0].shape[1]}")
        next_token_emb = torch.randn(1, 1, 64)
    
    # ==========================================================================
    # 统计
    # ==========================================================================
    print("\n" + "=" * 60)
    print("性能优势分析")
    print("=" * 60)
    print(f"""
    第一轮: 处理 {input1.total_len} tokens (Prompt {input1.prompt_len} + Query {input1.query_len})
    第二轮: 只处理 {input2.query_len} tokens (复用 Prompt Cache)
    
    节省计算: {input2.prompt_len} tokens 的 Prefill 计算
    适用场景:
      - 多轮对话 (System Prompt 固定)
      - 批量推理 (相同 Prompt, 不同 Query)
      - RAG 应用 (Context 固定, Query 变化)
    """)


def demo_transformer_block(use_compile: bool = False):
    """
    演示完整 Transformer Block 的 Prefill 和 Decode 阶段
    
    清晰展示两个阶段的区别和性能特点
    """
    torch.manual_seed(42)
    
    # 配置
    d_model = 256
    n_head = 8
    n_layers = 4
    batch_size = 2
    prompt_len = 32
    max_gen_tokens = 10
    
    print("=" * 70)
    print("Transformer Block: Prefill vs Decode 阶段演示")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  d_model={d_model}, n_head={n_head}, n_layers={n_layers}")
    print(f"  batch_size={batch_size}, prompt_len={prompt_len}")
    
    # 创建模型
    model = TransformerDecoder(
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        max_seq_len=2048
    )
    model.eval()
    
    if use_compile:
        print("\n使用 torch.compile 优化...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # =========================================================================
    # Prefill 阶段
    # =========================================================================
    print("\n" + "=" * 70)
    print("【Prefill 阶段】")
    print("=" * 70)
    print("""
    特点:
    - 一次性处理整个 prompt (多个 tokens)
    - 计算密集型 (Compute Bound)
    - 需要 causal mask
    - 可以充分利用 GPU 并行性
    - 延迟较高，但吞吐量高
    """)
    
    # 模拟 prompt embedding (实际中是 token -> embedding lookup)
    prompt_emb = torch.randn(batch_size, prompt_len, d_model)
    
    print(f"Input shape: {prompt_emb.shape}")
    
    # Prefill
    with torch.no_grad():
        import time
        start_time = time.perf_counter()
        
        output_prefill, kv_caches = model.prefill(prompt_emb)
        
        prefill_time = time.perf_counter() - start_time
    
    print(f"Output shape: {output_prefill.shape}")
    print(f"KV Cache shapes (per layer):")
    for i, (k, v) in enumerate(kv_caches):
        if i == 0:  # 只打印第一层
            print(f"  Layer {i}: K={k.shape}, V={v.shape}")
    print(f"  ... ({n_layers} layers total)")
    print(f"\nPrefill 耗时: {prefill_time*1000:.2f} ms")
    print(f"Tokens/s: {batch_size * prompt_len / prefill_time:.0f}")
    
    # =========================================================================
    # Decode 阶段
    # =========================================================================
    print("\n" + "=" * 70)
    print("【Decode 阶段】")
    print("=" * 70)
    print("""
    特点:
    - 逐 token 生成 (每次只处理 1 个 token)
    - 内存访问密集型 (Memory Bound)
    - 无需 causal mask (只看过去的 tokens)
    - 复用 KV Cache，避免重复计算
    - 延迟关键，每个 token 都要等待
    """)
    
    print(f"\n开始生成 {max_gen_tokens} 个 tokens:")
    
    decode_times = []
    current_pos = prompt_len
    
    # 模拟单个新 token
    next_token_emb = torch.randn(batch_size, 1, d_model)
    
    with torch.no_grad():
        for step in range(max_gen_tokens):
            start_time = time.perf_counter()
            
            # Decode 单个 token
            output_token, kv_caches = model.decode(
                next_token_emb, 
                kv_caches, 
                start_pos=current_pos
            )
            
            decode_time = time.perf_counter() - start_time
            decode_times.append(decode_time)
            
            current_pos += 1
            
            # 模拟采样下一个 token (实际中是 logits -> sample -> embed)
            next_token_emb = torch.randn(batch_size, 1, d_model)
            
            print(f"  Step {step+1}: pos={current_pos-1}, "
                  f"time={decode_time*1000:.2f}ms, "
                  f"KV len={kv_caches[0][0].shape[1]}")
    
    avg_decode_time = sum(decode_times) / len(decode_times)
    print(f"\n平均 Decode 耗时: {avg_decode_time*1000:.2f} ms/token")
    print(f"Decode Tokens/s: {batch_size / avg_decode_time:.0f}")
    
    # =========================================================================
    # 性能对比
    # =========================================================================
    print("\n" + "=" * 70)
    print("【性能对比分析】")
    print("=" * 70)
    
    prefill_tokens_per_sec = batch_size * prompt_len / prefill_time
    decode_tokens_per_sec = batch_size / avg_decode_time
    
    print(f"""
    Prefill 阶段:
      - 处理 tokens: {batch_size * prompt_len}
      - 总耗时: {prefill_time*1000:.2f} ms
      - 吞吐量: {prefill_tokens_per_sec:.0f} tokens/s
      - 计算特点: Compute Bound (大矩阵乘法)
    
    Decode 阶段:
      - 每步 tokens: {batch_size}
      - 平均耗时: {avg_decode_time*1000:.2f} ms/step
      - 吞吐量: {decode_tokens_per_sec:.0f} tokens/s
      - 计算特点: Memory Bound (读取 KV Cache)
    
    关键洞察:
      - Prefill 吞吐量通常远高于 Decode ({prefill_tokens_per_sec/decode_tokens_per_sec:.1f}x)
      - Decode 是生成速度的瓶颈
      - 优化策略:
        * Prefill: 增大 batch, 使用更大 tile
        * Decode: KV Cache 压缩, 投机解码, 批处理
    """)
    
    # =========================================================================
    # KV Cache 内存分析
    # =========================================================================
    print("=" * 70)
    print("【KV Cache 内存分析】")
    print("=" * 70)
    
    # 计算 KV Cache 大小
    # 每层: 2 * batch * seq_len * n_head * head_dim * sizeof(float)
    head_dim = d_model // n_head
    kv_cache_size_per_layer = 2 * batch_size * current_pos * n_head * head_dim * 4  # float32
    total_kv_cache_size = kv_cache_size_per_layer * n_layers
    
    print(f"""
    当前状态:
      - 序列长度: {current_pos} tokens
      - KV Cache 大小: {total_kv_cache_size / 1024:.2f} KB ({total_kv_cache_size / 1024**2:.2f} MB)
    
    如果序列长度增长:
      - 1K tokens: {2 * batch_size * 1024 * n_head * head_dim * 4 * n_layers / 1024**2:.2f} MB
      - 4K tokens: {2 * batch_size * 4096 * n_head * head_dim * 4 * n_layers / 1024**2:.2f} MB
      - 32K tokens: {2 * batch_size * 32768 * n_head * head_dim * 4 * n_layers / 1024**2:.2f} MB
    
    KV Cache 内存公式:
      Memory = 2 * batch * seq_len * n_layers * n_head * head_dim * sizeof(dtype)
    """)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--prompt-query", action="store_true", 
                        help="Demo prompt+query separation")
    parser.add_argument("--transformer", action="store_true",
                        help="Demo full transformer block with prefill/decode")
    args = parser.parse_args()
    
    if args.transformer:
        demo_transformer_block(use_compile=args.compile)
    elif args.prompt_query:
        demo_prompt_query_inference(use_compile=args.compile)
    else:
        demo_inference(use_compile=args.compile)
