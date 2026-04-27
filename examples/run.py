#!/usr/bin/env python3
"""最小 gist + prefix KV-cache 单机 demo"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gist_cache_lib import GistCacheStore

# ── 1. 加载模型 + 添加 gist token ──────────────────────────────
MODEL  = "gpt2"
K_GIST = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
tokenizer.pad_token = tokenizer.eos_token

# 注册 gist special tokens 并初始化 embedding
gist_tokens = [f"<gist_{i}>" for i in range(K_GIST)]
tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})
model.resize_token_embeddings(len(tokenizer))
gist_ids = tokenizer.convert_tokens_to_ids(gist_tokens)

# ── 2. 构建 gist KV cache ─────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful AI assistant. " * 20  # 长 prefix

prompt_ids = tokenizer.encode(SYSTEM_PROMPT)
all_ids = prompt_ids + gist_ids
with torch.no_grad():
    out = model(input_ids=torch.tensor([all_ids], device=DEVICE), use_cache=True)

# 只切出 gist token 对应的 KV 切片
gs, ge = len(prompt_ids), len(prompt_ids) + K_GIST
gist_cache = tuple(
    (k[:, :, gs:ge, :].contiguous(), v[:, :, gs:ge, :].contiguous())
    for k, v in out.past_key_values
)

# ── 3. 存入 GistCacheStore（contiguous pool + handle） ──────────
store  = GistCacheStore(device=DEVICE, pool=True)
handle = store.put_past(gist_cache)
print(f"CacheHandle: id={handle.cache_id}, k_gist={handle.k_gist}, "
      f"pool={handle.pool_packed}")
print(f"Store: {store.summary()}")

# ── 4. 用 handle 取出 KV，做推理 ───────────────────────────────
query = "What is the capital of France?"
q_text = f"\n\nUser: {query}\nAssistant:"
q_ids  = tokenizer.encode(q_text)
ids    = torch.tensor([q_ids], device=DEVICE)

past = store.get_past(handle)                       # zero-copy view
k_len = past[0][0].shape[2]                         # gist prefix 长度
pos   = torch.arange(k_len, k_len + len(q_ids), device=DEVICE).unsqueeze(0)

with torch.no_grad():
    out = model(input_ids=ids, past_key_values=past,
                position_ids=pos, use_cache=True)
    past = out.past_key_values
    tok  = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]
    cur = k_len + len(q_ids)
    for _ in range(49):
        p = torch.tensor([[cur]], device=DEVICE)
        out = model(input_ids=tok, past_key_values=past,
                    position_ids=p, use_cache=True)
        past = out.past_key_values
        tok  = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(tok.item())
        cur += 1
        if tok.item() == tokenizer.eos_token_id:
            break

print(f"\nQuery: {query}")
print(f"Output: {tokenizer.decode(generated, skip_special_tokens=True)}")

# ── 5. 清理 ────────────────────────────────────────────────────
store.release(handle)