#!/usr/bin/env python3
"""
Minimal executable demo: Gist Cache accelerates inference.

Compares two inference strategies on GPT-2:
  1) Baseline   – prepend the full system-prompt for every request
  2) Gist Cache – build a compressed KV-cache once, reuse for every request

Metrics reported:
  • per-request latency (ms)
  • KV-cache memory (KB)
  • break-even point (# requests after which gist cache pays for itself)

Usage:
    pip install transformers torch
    python examples/gist_cache_demo.py
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class Metrics:
    """Fine-grained per-request timing."""
    prefill_tokens: int = 0        # tokens consumed during prefill
    generated_tokens: int = 0      # tokens produced during decode
    t_prefill: float = 0.0         # prefill forward pass (s)
    t_first_token: float = 0.0     # TTFT: start → first token ready (s)
    t_decode: float = 0.0          # total decode phase (s)
    t_total: float = 0.0           # wall-clock start → finish (s)

    @property
    def prefill_tok_s(self) -> float:
        return self.prefill_tokens / self.t_prefill if self.t_prefill > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.generated_tokens / self.t_decode if self.t_decode > 0 else 0.0

# ── Configuration ────────────────────────────────────────────────
MODEL_NAME = "gpt2"
K_GIST = 4                # number of gist tokens (≪ prompt length)
MAX_NEW_TOKENS = 50        # tokens to generate per request
NUM_REQUESTS = 5           # number of requests sharing the same prompt
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# A deliberately long system prompt (reused across all requests).
# In production this could be a few-shot prompt, RAG context, etc.
_SYSTEM_BASE = (
    "You are a helpful, harmless, and honest AI assistant. "
    "You have deep expertise in mathematics, science, history, and programming. "
    "When answering questions, provide clear, concise, and accurate information. "
    "Always cite sources when possible. If unsure, say so rather than fabricating. "
    "Be respectful and patient. Format responses with markdown when appropriate. "
    "Break complex problems into smaller steps. Show work for math problems. "
    "Use code blocks for programming examples. Provide multiple perspectives for "
    "controversial topics. Prioritize safety and ethics in every response. "
)
SYSTEM_PROMPT = _SYSTEM_BASE * 3          # repeat → longer prompt, bigger win

USER_QUERIES = [
    "What is the capital of France?",
    "Explain quantum entanglement briefly.",
    "Write a Python function to sort a list.",
    "What causes rainbows?",
    "Summarize the theory of relativity.",
]

# ── Setup ────────────────────────────────────────────────────────

def setup_model():
    """Load GPT-2, add gist tokens, return (model, tokenizer, gist_ids)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register gist tokens
    gist_tokens = [f"<gist_{i}>" for i in range(K_GIST)]
    tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})
    model.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        std = emb.std().item()
        for tok in gist_tokens:
            emb[tokenizer.convert_tokens_to_ids(tok)].normal_(0.0, std)

    gist_ids = tokenizer.convert_tokens_to_ids(gist_tokens)
    return model, tokenizer, gist_ids


# ── Helper: cuda-aware timestamp ────────────────────────────────

def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ── Baseline: full prompt every time ────────────────────────────

@torch.no_grad()
def generate_baseline(model, tokenizer, prompt, query, max_new):
    """Encode full (prompt + query), prefill, then decode. Returns (text, Metrics)."""
    text = prompt + "\n\nUser: " + query + "\nAssistant:"
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    prefill_len = ids.shape[1]

    _sync()
    t_start = time.perf_counter()

    # ── Prefill ──
    out = model(input_ids=ids, use_cache=True)
    _sync()
    t_after_prefill = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]

    _sync()
    t_first_tok = time.perf_counter()          # TTFT: first token ready

    # ── Decode ──
    for _ in range(max_new - 1):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(tok.item())
        if tok.item() == tokenizer.eos_token_id:
            break

    _sync()
    t_end = time.perf_counter()

    m = Metrics(
        prefill_tokens=prefill_len,
        generated_tokens=len(generated),
        t_prefill=t_after_prefill - t_start,
        t_first_token=t_first_tok - t_start,
        t_decode=t_end - t_first_tok,
        t_total=t_end - t_start,
    )
    return tokenizer.decode(generated, skip_special_tokens=True), m


# ── Gist Cache ───────────────────────────────────────────────────

@torch.no_grad()
def build_gist_cache(model, tokenizer, prompt, gist_token_ids):
    """
    One forward pass on [prompt_tokens | gist_tokens].
    Returns only the KV slices at the gist positions.
    """
    prompt_ids = tokenizer.encode(prompt)
    all_ids = prompt_ids + list(gist_token_ids)
    ids = torch.tensor([all_ids], device=DEVICE)

    out = model(input_ids=ids, use_cache=True)
    full_past = out.past_key_values

    gs = len(prompt_ids)
    ge = gs + len(gist_token_ids)
    gist_cache = tuple(
        (k[:, :, gs:ge, :].contiguous(),
         v[:, :, gs:ge, :].contiguous())
        for k, v in full_past
    )
    return gist_cache, len(prompt_ids)


@torch.no_grad()
def generate_with_gist(model, tokenizer, gist_cache, query, max_new):
    """Decode with gist_cache as the only prefix KV (no prompt tokens). Returns (text, Metrics)."""
    text = "\n\nUser: " + query + "\nAssistant:"
    q_ids = tokenizer.encode(text)
    ids = torch.tensor([q_ids], device=DEVICE)

    k = gist_cache[0][0].shape[2]                       # gist seq len
    pos = torch.arange(k, k + len(q_ids), device=DEVICE).unsqueeze(0)
    prefill_len = len(q_ids)  # only the query tokens are prefilled

    _sync()
    t_start = time.perf_counter()

    # ── Prefill user query on top of gist cache ──
    out = model(input_ids=ids, past_key_values=gist_cache,
                position_ids=pos, use_cache=True)
    _sync()
    t_after_prefill = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]
    cur = k + len(q_ids)

    _sync()
    t_first_tok = time.perf_counter()

    # ── Decode ──
    for _ in range(max_new - 1):
        p = torch.tensor([[cur]], device=DEVICE)
        out = model(input_ids=tok, past_key_values=past,
                    position_ids=p, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(tok.item())
        cur += 1
        if tok.item() == tokenizer.eos_token_id:
            break

    _sync()
    t_end = time.perf_counter()

    m = Metrics(
        prefill_tokens=prefill_len,
        generated_tokens=len(generated),
        t_prefill=t_after_prefill - t_start,
        t_first_token=t_first_tok - t_start,
        t_decode=t_end - t_first_tok,
        t_total=t_end - t_start,
    )
    return tokenizer.decode(generated, skip_special_tokens=True), m


# ── Helpers ──────────────────────────────────────────────────────

def kv_bytes(past):
    """Total bytes stored in a past_key_values tuple."""
    return sum(
        k.nelement() * k.element_size() + v.nelement() * v.element_size()
        for k, v in past
    )


# ── Main benchmark ───────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Gist-Cache Inference Demo")
    print("=" * 70)

    model, tokenizer, gist_ids = setup_model()
    prompt_ntok = len(tokenizer.encode(SYSTEM_PROMPT))

    print(f"  Model            : {MODEL_NAME}")
    print(f"  Device / dtype   : {DEVICE} / {DTYPE}")
    print(f"  Prompt tokens    : {prompt_ntok}")
    print(f"  Gist tokens (k)  : {K_GIST}  "
          f"(compression {prompt_ntok}→{K_GIST}, "
          f"{prompt_ntok / K_GIST:.0f}x)")
    print(f"  Max new tokens   : {MAX_NEW_TOKENS}")
    print(f"  Requests         : {NUM_REQUESTS}")
    print()

    # ── Warmup (not timed) ──────────────────────────────────────
    print("  Warming up …")
    generate_baseline(model, tokenizer, SYSTEM_PROMPT, USER_QUERIES[0], 3)
    gc, _ = build_gist_cache(model, tokenizer, SYSTEM_PROMPT, gist_ids)
    generate_with_gist(model, tokenizer, gc, USER_QUERIES[0], 3)
    _sync()
    print()

    # ── helper: print per-request row ───────────────────────────
    hdr = (f"  {'#':>3}  {'T_total':>9} {'T_prefill':>10} {'TTFT':>9} "
           f"{'T_decode':>9} {'prefill':>9} {'decode':>9}  Query")
    units = (f"  {'':>3}  {'(ms)':>9} {'(ms)':>10} {'(ms)':>9} "
             f"{'(ms)':>9} {'(tok/s)':>9} {'(tok/s)':>9}")

    def print_row(idx: int, m: Metrics, query: str):
        print(f"  {idx:>3}  {m.t_total*1e3:9.1f} {m.t_prefill*1e3:10.1f} "
              f"{m.t_first_token*1e3:9.1f} {m.t_decode*1e3:9.1f} "
              f"{m.prefill_tok_s:9.0f} {m.decode_tok_s:9.0f}  {query[:38]}")

    def print_avg(metrics: List[Metrics]):
        n = len(metrics)
        print(f"  {'avg':>3}  "
              f"{sum(m.t_total for m in metrics)/n*1e3:9.1f} "
              f"{sum(m.t_prefill for m in metrics)/n*1e3:10.1f} "
              f"{sum(m.t_first_token for m in metrics)/n*1e3:9.1f} "
              f"{sum(m.t_decode for m in metrics)/n*1e3:9.1f} "
              f"{sum(m.prefill_tok_s for m in metrics)/n:9.0f} "
              f"{sum(m.decode_tok_s for m in metrics)/n:9.0f}")

    # ── 1. Baseline ─────────────────────────────────────────────
    print("─" * 90)
    print("  [A] Baseline – re-process full prompt for every request")
    print("─" * 90)
    print(hdr)
    print(units)

    base_metrics: List[Metrics] = []
    base_texts: List[str] = []
    for i, q in enumerate(USER_QUERIES[:NUM_REQUESTS]):
        txt, m = generate_baseline(model, tokenizer, SYSTEM_PROMPT, q, MAX_NEW_TOKENS)
        base_metrics.append(m)
        base_texts.append(txt)
        print_row(i + 1, m, q)

    print("  " + "-" * 86)
    print_avg(base_metrics)
    print()

    # ── 2. Gist Cache ──────────────────────────────────────────
    print("─" * 90)
    print("  [B] Gist Cache – build once, reuse for every request")
    print("─" * 90)

    # Build (one-time)
    _sync()
    t_build = time.perf_counter()
    gist_cache, _ = build_gist_cache(model, tokenizer, SYSTEM_PROMPT, gist_ids)
    _sync()
    build_ms = (time.perf_counter() - t_build) * 1000
    print(f"  cache build: {build_ms:.1f} ms  (one-time cost)")
    print(hdr)
    print(units)

    gist_metrics: List[Metrics] = []
    gist_texts: List[str] = []
    for i, q in enumerate(USER_QUERIES[:NUM_REQUESTS]):
        txt, m = generate_with_gist(model, tokenizer, gist_cache, q, MAX_NEW_TOKENS)
        gist_metrics.append(m)
        gist_texts.append(txt)
        print_row(i + 1, m, q)

    print("  " + "-" * 86)
    print_avg(gist_metrics)
    print()

    # ── 3. KV-cache memory ──────────────────────────────────────
    with torch.no_grad():
        full_ids = tokenizer.encode(SYSTEM_PROMPT, return_tensors="pt").to(DEVICE)
        full_past = model(input_ids=full_ids, use_cache=True).past_key_values
    full_kb = kv_bytes(full_past) / 1024
    gist_kb = kv_bytes(gist_cache) / 1024

    # ── 4. Summary ──────────────────────────────────────────────
    n = len(base_metrics)
    avg = lambda ms, attr: sum(getattr(m, attr) for m in ms) / len(ms)

    base_total_avg   = avg(base_metrics, "t_total")
    base_prefill_avg = avg(base_metrics, "t_prefill")
    base_ttft_avg    = avg(base_metrics, "t_first_token")
    base_decode_avg  = avg(base_metrics, "t_decode")
    base_pf_tps      = avg(base_metrics, "prefill_tok_s")
    base_dec_tps     = avg(base_metrics, "decode_tok_s")

    gist_total_avg   = avg(gist_metrics, "t_total")
    gist_prefill_avg = avg(gist_metrics, "t_prefill")
    gist_ttft_avg    = avg(gist_metrics, "t_first_token")
    gist_decode_avg  = avg(gist_metrics, "t_decode")
    gist_pf_tps      = avg(gist_metrics, "prefill_tok_s")
    gist_dec_tps     = avg(gist_metrics, "decode_tok_s")

    print("=" * 90)
    print("  Summary  (averages over {n} requests)".format(n=n))
    print("=" * 90)

    row = "  {label:<16} {total:>9} {prefill:>10} {ttft:>9} {decode:>9} {pf:>9} {dec:>9}"
    print(row.format(label="", total="T_total", prefill="T_prefill",
                     ttft="TTFT", decode="T_decode", pf="prefill", dec="decode"))
    print(row.format(label="", total="(ms)", prefill="(ms)",
                     ttft="(ms)", decode="(ms)", pf="(tok/s)", dec="(tok/s)"))
    print("  " + "-" * 86)
    print(row.format(
        label="Baseline",
        total=f"{base_total_avg*1e3:.1f}",
        prefill=f"{base_prefill_avg*1e3:.1f}",
        ttft=f"{base_ttft_avg*1e3:.1f}",
        decode=f"{base_decode_avg*1e3:.1f}",
        pf=f"{base_pf_tps:.0f}",
        dec=f"{base_dec_tps:.0f}",
    ))
    print(row.format(
        label="Gist Cache",
        total=f"{gist_total_avg*1e3:.1f}",
        prefill=f"{gist_prefill_avg*1e3:.1f}",
        ttft=f"{gist_ttft_avg*1e3:.1f}",
        decode=f"{gist_decode_avg*1e3:.1f}",
        pf=f"{gist_pf_tps:.0f}",
        dec=f"{gist_dec_tps:.0f}",
    ))
    print("  " + "-" * 86)

    def ratio_str(a, b):
        if b > 0:
            return f"{a / b:.2f}x"
        return "n/a"

    print(row.format(
        label="Speedup",
        total=ratio_str(base_total_avg, gist_total_avg),
        prefill=ratio_str(base_prefill_avg, gist_prefill_avg),
        ttft=ratio_str(base_ttft_avg, gist_ttft_avg),
        decode=ratio_str(base_decode_avg, gist_decode_avg),
        pf=ratio_str(gist_pf_tps, base_pf_tps),
        dec=ratio_str(gist_dec_tps, base_dec_tps),
    ))
    print()

    print(f"  Prompt KV-cache memory:")
    print(f"    Full prompt : {full_kb:8.1f} KB  ({prompt_ntok} tokens)")
    print(f"    Gist cache  : {gist_kb:8.1f} KB  ({K_GIST} tokens)")
    print(f"    Reduction   : {full_kb / gist_kb:8.1f}x")
    print()

    saved_ms = (base_total_avg - gist_total_avg) * 1000
    if saved_ms > 0:
        breakeven = build_ms / saved_ms
        print(f"  Break-even point: {breakeven:.1f} requests")
        print(f"    (cache build cost is amortized after ~{int(breakeven)+1} requests)")
    else:
        print("  No per-request speedup observed on this hardware / prompt length.")
        print("  Try a longer prompt or GPU for a clearer win.")

    print()
    print("─" * 70)
    print("  Sample outputs (first query)")
    print("─" * 70)
    print(f"  Query   : {USER_QUERIES[0]}")
    print(f"  Baseline: {base_texts[0][:120]}")
    print(f"  Gist    : {gist_texts[0][:120]}")
    print()
    print("  NOTE: Outputs differ because gist tokens are NOT fine-tuned in this")
    print("  demo.  With proper gist-token training, output quality matches the")
    print("  baseline while retaining the latency & memory benefits shown above.")


if __name__ == "__main__":
    main()
