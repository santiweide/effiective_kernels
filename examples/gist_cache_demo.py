#!/usr/bin/env python3
"""
Gist-Cache 3-way Comparison Demo (CacheHandle edition).

Compares three inference strategies on GPT-2:
  [A] Baseline   – full prompt every request
  [B] KV Reuse   – precomputed full-prompt KV, cloned per request
  [C] Gist Cache – compressed K_GIST-token KV, served via GistCacheStore

The Gist Cache arm uses GistCacheStore + CacheHandle, demonstrating the
handle-based API.  The model forward path receives standard
past_key_values – nothing changes from the model's perspective.

Uses: gist_bench.{config, engine, metrics, reporting}
      gist_cache_lib.{GistCacheStore, CacheHandle}

Usage:
    python examples/gist_cache_demo.py
"""

import sys, os, time, warnings
from typing import List

# Allow running from repo root:  python examples/gist_cache_demo.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*mean_resizing.*")

from gist_bench.config import (
    MODEL_NAME, K_GIST, DEVICE, DTYPE,
    MAX_NEW_TOKENS, NUM_REQUESTS, SYSTEM_PROMPT, USER_QUERIES,
)
from gist_bench.engine import (
    setup_model, kv_bytes, _sync,
    generate_baseline, build_prompt_kv_cache, generate_with_kv_reuse,
    build_gist_cache, generate_with_gist,
    # CacheStore-aware wrappers
    build_gist_cache_to_store, generate_with_gist_cached,
)
from gist_bench.metrics import Metrics, pack_avgs
from gist_bench.reporting import (
    DEMO_HDR, DEMO_UNITS, print_row, print_avg,
    print_summary_table, ratio_str,
)
from gist_cache_lib import GistCacheStore, CacheHandle


def main():
    print("=" * 70)
    print("  Gist-Cache Inference Demo  (CacheHandle edition)")
    print("=" * 70)

    model, tokenizer, gist_ids = setup_model()
    prompt_ntok = len(tokenizer.encode(SYSTEM_PROMPT))

    # ── Initialise GistCacheStore ───────────────────────────────
    store = GistCacheStore(
        device=DEVICE,
        pool=True,                           # pack KV into contiguous buffer
        model_config=model.config,           # hash-based model validation
    )

    print(f"  Model            : {MODEL_NAME}")
    print(f"  Device / dtype   : {DEVICE} / {DTYPE}")
    print(f"  Prompt tokens    : {prompt_ntok}")
    print(f"  Gist tokens (k)  : {K_GIST}  "
          f"(compression {prompt_ntok}→{K_GIST}, "
          f"{prompt_ntok / K_GIST:.0f}x)")
    print(f"  Max new tokens   : {MAX_NEW_TOKENS}")
    print(f"  Requests         : {NUM_REQUESTS}")
    print(f"  CacheStore       : pool={store.use_pool}, "
          f"model_hash={store.model_hash[:8]}…")
    print()

    # ── Warmup ──────────────────────────────────────────────────
    print("  Warming up …")
    generate_baseline(model, tokenizer, SYSTEM_PROMPT, USER_QUERIES[0], 3)
    _pkv = build_prompt_kv_cache(model, tokenizer, SYSTEM_PROMPT)
    generate_with_kv_reuse(model, tokenizer, _pkv, USER_QUERIES[0], 3)
    # Warmup gist path via store
    _h = build_gist_cache_to_store(store, model, tokenizer, SYSTEM_PROMPT, gist_ids)
    generate_with_gist_cached(model, tokenizer, store, _h, USER_QUERIES[0], 3)
    store.release(_h)
    del _pkv
    _sync()
    print()

    # ── [A] Baseline ────────────────────────────────────────────
    print("─" * 90)
    print("  [A] Baseline – re-process full prompt for every request")
    print("─" * 90)
    print(DEMO_HDR); print(DEMO_UNITS)

    base_metrics: List[Metrics] = []
    base_texts: List[str] = []
    for i, q in enumerate(USER_QUERIES[:NUM_REQUESTS]):
        txt, m = generate_baseline(model, tokenizer, SYSTEM_PROMPT, q, MAX_NEW_TOKENS)
        base_metrics.append(m); base_texts.append(txt)
        print_row(i + 1, m, q)
    print("  " + "-" * 86); print_avg(base_metrics); print()

    # ── [B] KV Reuse ───────────────────────────────────────────
    print("─" * 90)
    print("  [B] KV Reuse – compute prompt KV once, reuse for every request")
    print("─" * 90)

    _sync(); t0 = time.perf_counter()
    prompt_kv = build_prompt_kv_cache(model, tokenizer, SYSTEM_PROMPT)
    _sync(); kv_build_ms = (time.perf_counter() - t0) * 1000
    print(f"  KV build: {kv_build_ms:.1f} ms  (one-time cost)")
    print(DEMO_HDR); print(DEMO_UNITS)

    reuse_metrics: List[Metrics] = []
    reuse_texts: List[str] = []
    for i, q in enumerate(USER_QUERIES[:NUM_REQUESTS]):
        txt, m = generate_with_kv_reuse(model, tokenizer, prompt_kv, q, MAX_NEW_TOKENS)
        reuse_metrics.append(m); reuse_texts.append(txt)
        print_row(i + 1, m, q)
    print("  " + "-" * 86); print_avg(reuse_metrics); print()

    # ── [C] Gist Cache (via CacheStore) ───────────────────────────
    print("─" * 90)
    print("  [C] Gist Cache – build once, store as handle, reuse per request")
    print("─" * 90)

    _sync(); t0 = time.perf_counter()
    gist_handle = build_gist_cache_to_store(
        store, model, tokenizer, SYSTEM_PROMPT, gist_ids)
    _sync(); build_ms = (time.perf_counter() - t0) * 1000
    print(f"  gist build + store.put_past: {build_ms:.1f} ms  (one-time cost)")
    print(f"  CacheHandle: id={gist_handle.cache_id}  "
          f"k_gist={gist_handle.k_gist}  pool={gist_handle.pool_packed}  "
          f"model_hash={gist_handle.model_hash[:8]}…")
    print(f"  Store: {store.summary()}")
    print(DEMO_HDR); print(DEMO_UNITS)

    gist_metrics: List[Metrics] = []
    gist_texts: List[str] = []
    for i, q in enumerate(USER_QUERIES[:NUM_REQUESTS]):
        # Resolve handle → past_key_values via store, then generate
        txt, m = generate_with_gist_cached(
            model, tokenizer, store, gist_handle, q, MAX_NEW_TOKENS)
        gist_metrics.append(m); gist_texts.append(txt)
        print_row(i + 1, m, q)
    print("  " + "-" * 86); print_avg(gist_metrics); print()

    # ── KV memory ──────────────────────────────────────────────
    full_kb = kv_bytes(prompt_kv) / 1024
    # Gist KB from pool (contiguous buffer)
    gist_pool = store.get_pool(gist_handle)
    gist_kb = gist_pool.nbytes / 1024

    # ── Summary table ──────────────────────────────────────────
    b  = pack_avgs(base_metrics)
    rv = pack_avgs(reuse_metrics)
    g  = pack_avgs(gist_metrics)

    print_summary_table(
        n_requests=len(base_metrics),
        packs=[
            ("Baseline",   b,  full_kb),
            ("KV Reuse",   rv, full_kb),
            ("Gist Cache", g,  gist_kb),
        ],
        baseline_pack=b,
        comparisons=[("KV Reuse", rv), ("Gist Cache", g)],
        gist_vs_kv=(g, rv, f"{full_kb / gist_kb:.0f}x less"),
    )

    print(f"  KV Memory:")
    print(f"    Full prompt (Baseline / KV Reuse) : {full_kb:8.1f} KB  ({prompt_ntok} tokens)")
    print(f"    Gist cache                        : {gist_kb:8.1f} KB  ({K_GIST} tokens)")
    print(f"    Reduction                         : {full_kb / gist_kb:8.1f}x")
    print()
    print("  Build costs (one-time):")
    print(f"    KV Reuse  : {kv_build_ms:8.1f} ms")
    print(f"    Gist Cache: {build_ms:8.1f} ms")
    print()

    W = 96
    print("─" * W)
    print("  Interpretation Guide")
    print("─" * W)
    print("  • Baseline vs KV Reuse: shows pure prefill savings from caching")
    print("    (KV Reuse has identical output quality, same decode speed,")
    print("     but still stores full prompt length in KV memory per request).")
    print("  • KV Reuse vs Gist Cache: same prefill speed (both skip prompt),")
    print(f"    but Gist Cache uses {full_kb / gist_kb:.0f}x less KV memory per request,")
    print(f"    enabling {full_kb / gist_kb:.0f}x more concurrent requests in same GPU memory.")
    print("  • Gist Cache decode may be slightly faster due to shorter KV")
    print("    reducing memory bandwidth during attention.")
    print()

    print("─" * W)
    print("  Sample outputs (first query)")
    print("─" * W)
    print(f"  Query    : {USER_QUERIES[0]}")
    print(f"  Baseline : {base_texts[0][:120]}")
    print(f"  KV Reuse : {reuse_texts[0][:120]}")
    print(f"  Gist     : {gist_texts[0][:120]}")
    print()
    print("  NOTE: Baseline and KV Reuse outputs should be IDENTICAL (same KV, same")
    print("  logits).  Gist outputs differ because gist tokens are NOT fine-tuned")
    print("  in this demo.  With proper training, quality matches while retaining")
    print("  the memory benefits shown above.")


if __name__ == "__main__":
    main()
