"""
gist_bench.engine – Model setup + all inference paths.

Three strategies:
  1. Baseline   – full prompt every request
  2. KV Reuse   – precomputed full-prompt KV, cloned per request
  3. Gist Cache – compressed KV (k gist tokens) reused per request

Each generate_* function returns ``(decoded_text, Metrics)``.
Batched variants (bench_*) return ``Metrics`` only.

CacheStore integration
~~~~~~~~~~~~~~~~~~~~~~
Functions suffixed with ``_cached`` accept a
:class:`~gist_cache_lib.CacheHandle` instead of raw tensors.
They resolve the handle via the store at call time, keeping the
model forward path unchanged.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple, TYPE_CHECKING

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import MODEL_NAME, K_GIST, DEVICE, DTYPE
from .metrics import Metrics

if TYPE_CHECKING:
    from gist_cache_lib import GistCacheStore, CacheHandle

# ═══════════════════════════════════════════════════════════════════
#  Sync helper
# ═══════════════════════════════════════════════════════════════════

def _sync():
    """Barrier so CPU timers agree with GPU."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════
#  Model setup
# ═══════════════════════════════════════════════════════════════════

def setup_model():
    """Load model, add gist tokens.  Returns (model, tokenizer, gist_token_ids)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def make_prompt_ids(tokenizer, target_len: int) -> List[int]:
    """Return exactly *target_len* token IDs as a synthetic prompt."""
    base = (
        "You are a helpful, harmless, and honest AI assistant with deep "
        "expertise in mathematics, science, history, and programming. "
        "When answering questions, provide clear and accurate information. "
    )
    ids = tokenizer.encode(base)
    while len(ids) < target_len:
        ids = ids + ids
    return ids[:target_len]


# ═══════════════════════════════════════════════════════════════════
#  KV helpers
# ═══════════════════════════════════════════════════════════════════

def kv_bytes(past) -> int:
    """Total bytes stored in a ``past_key_values`` tuple."""
    return sum(
        k.nelement() * k.element_size() + v.nelement() * v.element_size()
        for k, v in past
    )


def _clone_kv(past):
    """Deep-clone a past_key_values tuple so the original stays untouched."""
    return tuple((k.clone(), v.clone()) for k, v in past)


# ═══════════════════════════════════════════════════════════════════
#  1. Baseline – full prompt every time
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_baseline(model, tokenizer, prompt: str, query: str, max_new: int):
    """Encode full (prompt+query), prefill, decode.  Returns (text, Metrics)."""
    text = prompt + "\n\nUser: " + query + "\nAssistant:"
    ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    prefill_len = ids.shape[1]

    _sync(); t_start = time.perf_counter()

    out = model(input_ids=ids, use_cache=True)
    _sync(); t_after_prefill = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]
    _sync(); t_first_tok = time.perf_counter()

    for _ in range(max_new - 1):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(tok.item())
        if tok.item() == tokenizer.eos_token_id:
            break

    _sync(); t_end = time.perf_counter()

    m = Metrics(
        prefill_tokens=prefill_len, generated_tokens=len(generated),
        t_prefill=t_after_prefill - t_start,
        t_first_token=t_first_tok - t_start,
        t_decode=t_end - t_first_tok,
        t_total=t_end - t_start,
    )
    return tokenizer.decode(generated, skip_special_tokens=True), m


# ═══════════════════════════════════════════════════════════════════
#  2. KV Reuse – compute prompt KV once, clone per request
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_prompt_kv_cache(model, tokenizer, prompt: str):
    """One forward on full prompt → complete past_key_values."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    out = model(input_ids=ids, use_cache=True)
    return tuple((k.contiguous(), v.contiguous()) for k, v in out.past_key_values)


@torch.no_grad()
def generate_with_kv_reuse(model, tokenizer, prompt_kv, query: str, max_new: int):
    """Reuse precomputed full-prompt KV cache.  Returns (text, Metrics)."""
    text = "\n\nUser: " + query + "\nAssistant:"
    q_ids = tokenizer.encode(text)
    ids = torch.tensor([q_ids], device=DEVICE)

    prefix_len = prompt_kv[0][0].shape[2]
    pos = torch.arange(prefix_len, prefix_len + len(q_ids),
                       device=DEVICE).unsqueeze(0)
    past = _clone_kv(prompt_kv)

    _sync(); t_start = time.perf_counter()

    out = model(input_ids=ids, past_key_values=past,
                position_ids=pos, use_cache=True)
    _sync(); t_after_prefill = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]
    cur = prefix_len + len(q_ids)
    _sync(); t_first_tok = time.perf_counter()

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

    _sync(); t_end = time.perf_counter()

    m = Metrics(
        prefill_tokens=len(q_ids), generated_tokens=len(generated),
        t_prefill=t_after_prefill - t_start,
        t_first_token=t_first_tok - t_start,
        t_decode=t_end - t_first_tok,
        t_total=t_end - t_start,
    )
    return tokenizer.decode(generated, skip_special_tokens=True), m


# ═══════════════════════════════════════════════════════════════════
#  3. Gist Cache – compressed KV
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def build_gist_cache(model, tokenizer_or_ids, prompt_or_gist_ids, gist_token_ids=None):
    """
    Build gist cache.  Two call signatures:

    1) ``build_gist_cache(model, tokenizer, prompt_str, gist_ids)``
    2) ``build_gist_cache(model, prompt_id_list, gist_id_list)``
    """
    if gist_token_ids is not None:
        # signature 1: (model, tokenizer, prompt_str, gist_ids)
        tokenizer = tokenizer_or_ids
        prompt_ids = tokenizer.encode(prompt_or_gist_ids)
        gist_ids = list(gist_token_ids)
    else:
        # signature 2: (model, prompt_id_list, gist_id_list)
        prompt_ids = list(tokenizer_or_ids)
        gist_ids = list(prompt_or_gist_ids)

    all_ids = prompt_ids + gist_ids
    ids = torch.tensor([all_ids], device=DEVICE)
    out = model(input_ids=ids, use_cache=True)
    gs, ge = len(prompt_ids), len(prompt_ids) + len(gist_ids)
    gist_cache = tuple(
        (k[:, :, gs:ge, :].contiguous(), v[:, :, gs:ge, :].contiguous())
        for k, v in out.past_key_values
    )
    return gist_cache, len(prompt_ids)


@torch.no_grad()
def generate_with_gist(model, tokenizer, gist_cache, query: str, max_new: int):
    """Decode with gist_cache as the only prefix KV.  Returns (text, Metrics)."""
    text = "\n\nUser: " + query + "\nAssistant:"
    q_ids = tokenizer.encode(text)
    ids = torch.tensor([q_ids], device=DEVICE)

    k = gist_cache[0][0].shape[2]
    pos = torch.arange(k, k + len(q_ids), device=DEVICE).unsqueeze(0)

    _sync(); t_start = time.perf_counter()

    out = model(input_ids=ids, past_key_values=gist_cache,
                position_ids=pos, use_cache=True)
    _sync(); t_after_prefill = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [tok.item()]
    cur = k + len(q_ids)
    _sync(); t_first_tok = time.perf_counter()

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

    _sync(); t_end = time.perf_counter()

    m = Metrics(
        prefill_tokens=len(q_ids), generated_tokens=len(generated),
        t_prefill=t_after_prefill - t_start,
        t_first_token=t_first_tok - t_start,
        t_decode=t_end - t_first_tok,
        t_total=t_end - t_start,
    )
    return tokenizer.decode(generated, skip_special_tokens=True), m


# ═══════════════════════════════════════════════════════════════════
#  Batched variants  (used by sweep)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def bench_baseline_batched(model, prompt_ids, query_ids, max_new: int, batch_size: int):
    """Batched baseline: B copies of (prompt+query).  Returns Metrics."""
    B = batch_size
    full_ids = list(prompt_ids) + list(query_ids)
    ids = torch.tensor([full_ids] * B, device=DEVICE)

    _sync(); t0 = time.perf_counter()

    out = model(input_ids=ids, use_cache=True)
    _sync(); t_pf = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    _sync(); t_ft = time.perf_counter()

    n_gen = 1
    for _ in range(max_new - 1):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        n_gen += 1

    _sync(); t_end = time.perf_counter()

    return Metrics(
        prefill_tokens=len(full_ids) * B, generated_tokens=n_gen * B,
        t_prefill=t_pf - t0, t_first_token=t_ft - t0,
        t_decode=t_end - t_ft, t_total=t_end - t0, batch_size=B,
    )


@torch.no_grad()
def bench_gist_batched(model, gist_cache, query_ids, max_new: int, batch_size: int):
    """Batched gist: expand gist KV to B.  Returns Metrics."""
    B = batch_size
    k = gist_cache[0][0].shape[2]

    expanded = tuple(
        (kk.expand(B, -1, -1, -1), vv.expand(B, -1, -1, -1))
        for kk, vv in gist_cache
    )
    ids = torch.tensor([list(query_ids)] * B, device=DEVICE)
    pos = torch.arange(k, k + len(query_ids), device=DEVICE,
                       dtype=torch.long).unsqueeze(0).expand(B, -1)

    _sync(); t0 = time.perf_counter()

    out = model(input_ids=ids, past_key_values=expanded,
                position_ids=pos, use_cache=True)
    _sync(); t_pf = time.perf_counter()

    past = out.past_key_values
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    _sync(); t_ft = time.perf_counter()

    cur = k + len(query_ids)
    n_gen = 1
    for _ in range(max_new - 1):
        p = torch.full((B, 1), cur, device=DEVICE, dtype=torch.long)
        out = model(input_ids=tok, past_key_values=past,
                    position_ids=p, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        cur += 1
        n_gen += 1

    _sync(); t_end = time.perf_counter()

    return Metrics(
        prefill_tokens=len(query_ids) * B, generated_tokens=n_gen * B,
        t_prefill=t_pf - t0, t_first_token=t_ft - t0,
        t_decode=t_end - t_ft, t_total=t_end - t0, batch_size=B,
    )


# ═══════════════════════════════════════════════════════════════════
#  CacheStore-aware wrappers
# ═══════════════════════════════════════════════════════════════════

def build_gist_cache_to_store(
    store: "GistCacheStore",
    model,
    tokenizer_or_ids,
    prompt_or_gist_ids,
    gist_token_ids=None,
) -> "CacheHandle":
    """Build gist cache and register it in *store*.  Returns a CacheHandle.

    Accepts the same call signatures as :func:`build_gist_cache`.
    """
    gist_cache, prefix_len = build_gist_cache(
        model, tokenizer_or_ids, prompt_or_gist_ids, gist_token_ids
    )
    handle = store.put_past(gist_cache)
    return handle


@torch.no_grad()
def generate_with_gist_cached(
    model,
    tokenizer,
    store: "GistCacheStore",
    handle: "CacheHandle",
    query: str,
    max_new: int,
):
    """Decode with gist cache resolved from *handle*.

    Same behaviour as :func:`generate_with_gist`, but retrieves the
    KV tensors via ``store.get_past(handle)`` instead of taking them
    directly.  Returns ``(text, Metrics)``.
    """
    gist_cache = store.get_past(handle)
    return generate_with_gist(model, tokenizer, gist_cache, query, max_new)


@torch.no_grad()
def bench_gist_batched_cached(
    model,
    store: "GistCacheStore",
    handle: "CacheHandle",
    query_ids,
    max_new: int,
    batch_size: int,
):
    """Batched gist benchmark resolved from *handle*.

    Same as :func:`bench_gist_batched` but retrieves gist cache from
    the store.  Returns ``Metrics``.
    """
    gist_cache = store.get_past(handle)
    return bench_gist_batched(model, gist_cache, query_ids, max_new, batch_size)

