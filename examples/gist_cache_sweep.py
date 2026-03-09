#!/usr/bin/env python3
"""
Gist-Cache Sweep Benchmark (CacheHandle edition).

Sweeps three dimensions (Prompt length L, Output length G, Batch size B)
and compares Baseline vs Gist Cache inference.

The Gist Cache arm uses GistCacheStore + CacheHandle, demonstrating the
handle-based API.

Uses: gist_bench.{config, engine, metrics, reporting}
      gist_cache_lib.{GistCacheStore}

Usage:
    python examples/gist_cache_sweep.py
    python examples/gist_cache_sweep.py --csv results.csv
"""

import argparse, itertools, sys, os, warnings

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore", message=".*past_key_values.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*mean_resizing.*")

from gist_bench.config import (
    MODEL_NAME, K_GIST, DEVICE, DTYPE,
    PROMPT_LENGTHS, GEN_LENGTHS, BATCH_SIZES,
    WARMUP_RUNS, BENCH_RUNS, SWEEP_QUERY,
)
from gist_bench.engine import (
    setup_model, make_prompt_ids, kv_bytes,
    build_gist_cache, bench_baseline_batched, bench_gist_batched,
    # CacheStore-aware
    build_gist_cache_to_store, bench_gist_batched_cached,
)
from gist_bench.metrics import avg_metrics
from gist_bench.reporting import ratio, ratio_str, export_csv
from gist_cache_lib import GistCacheStore


def _rx(a: float, b: float) -> str:
    r = ratio(a, b)
    return f"{r:.2f}x"


def main():
    parser = argparse.ArgumentParser(description="Gist-Cache Sweep Benchmark")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    args = parser.parse_args()

    model, tokenizer, gist_ids = setup_model()
    max_ctx = model.config.n_positions

    # ── Initialise GistCacheStore ───────────────────────────────
    store = GistCacheStore(
        device=DEVICE,
        pool=True,
        model_config=model.config,
    )

    query_ids = tokenizer.encode("\n\nUser: " + SWEEP_QUERY + "\nAssistant:")
    q_len = len(query_ids)

    # ── Header ──────────────────────────────────────────────────
    W = 116
    print("=" * W)
    print("  Gist-Cache Sweep Benchmark")
    print("=" * W)
    print(f"  Model: {MODEL_NAME}  |  Device: {DEVICE}  |  Dtype: {DTYPE}  "
          f"|  K_gist: {K_GIST}  |  Max context: {max_ctx}")
    print(f"  Sweep: L={PROMPT_LENGTHS}  G={GEN_LENGTHS}  B={BATCH_SIZES}")
    print(f"  Bench runs: {BENCH_RUNS} (avg)  |  Warmup: {WARMUP_RUNS}")
    print(f"  Query: \"{SWEEP_QUERY}\" ({q_len} tokens)")
    print()

    # ── Build sweep grid ────────────────────────────────────────
    configs = list(itertools.product(PROMPT_LENGTHS, GEN_LENGTHS, BATCH_SIZES))
    valid = [(L, G, B) for L, G, B in configs if L + q_len + G <= max_ctx]
    skipped = len(configs) - len(valid)
    if skipped:
        print(f"  Skipping {skipped}/{len(configs)} configs "
              f"(exceed context window {max_ctx})")
    print(f"  Running {len(valid)} configs …")
    print()

    # ── Progress header ─────────────────────────────────────────
    print(f"  {'#':>3}  {'L':>5} {'G':>4} {'B':>3}  │"
          f"  {'base_tot':>8} {'gist_tot':>8} {'sp_tot':>7}  │"
          f"  {'base_pf':>8} {'gist_pf':>8} {'sp_pf':>7}  │"
          f"  {'TTFT_b':>7} {'TTFT_g':>7} {'sp':>6}")
    print(f"  {'':>3}  {'':>5} {'':>4} {'':>3}  │"
          f"  {'(ms)':>8} {'(ms)':>8} {'':>7}  │"
          f"  {'(ms)':>8} {'(ms)':>8} {'':>7}  │"
          f"  {'(ms)':>7} {'(ms)':>7} {'':>6}")
    print("  " + "─" * (W - 2))

    results = []

    for ci, (L, G, B) in enumerate(valid):
        prompt_ids = make_prompt_ids(tokenizer, L)
        # Build gist cache into store → handle
        handle = build_gist_cache_to_store(store, model, prompt_ids, gist_ids)
        gist_cache = store.get_past(handle)   # views for bench functions
        gist_pool = store.get_pool(handle)

        # Warmup
        for _ in range(WARMUP_RUNS):
            bench_baseline_batched(model, prompt_ids, query_ids, min(G, 4), B)
            bench_gist_batched(model, gist_cache, query_ids, min(G, 4), B)

        # Timed
        base_runs = [bench_baseline_batched(model, prompt_ids, query_ids, G, B)
                     for _ in range(BENCH_RUNS)]
        gist_runs = [bench_gist_batched_cached(model, store, handle, query_ids, G, B)
                     for _ in range(BENCH_RUNS)]

        bm = avg_metrics(base_runs)
        gm = avg_metrics(gist_runs)

        sp_total   = ratio(bm.t_total,      gm.t_total)
        sp_prefill = ratio(bm.t_prefill,     gm.t_prefill)
        sp_ttft    = ratio(bm.t_first_token, gm.t_first_token)

        results.append(dict(
            L=L, G=G, B=B, base=bm, gist=gm,
            gist_cache_kb=gist_pool.nbytes / 1024,
            sp_total=sp_total, sp_prefill=sp_prefill, sp_ttft=sp_ttft,
        ))

        # Release this config's cache to free GPU memory
        store.release(handle)

        print(f"  {ci+1:>3}  {L:>5} {G:>4} {B:>3}  │"
              f"  {bm.t_total*1e3:8.1f} {gm.t_total*1e3:8.1f}"
              f" {sp_total:6.2f}x  │"
              f"  {bm.t_prefill*1e3:8.1f} {gm.t_prefill*1e3:8.1f}"
              f" {sp_prefill:6.2f}x  │"
              f"  {bm.t_first_token*1e3:7.1f}"
              f" {gm.t_first_token*1e3:7.1f}"
              f" {sp_ttft:5.2f}x")

    # ── Summary tables ──────────────────────────────────────────
    print()
    print("=" * W)
    print("  Summary Table  (all times in ms; tok/s = aggregate across batch)")
    print("=" * W)

    h1 = (f"  {'L':>5} {'G':>4} {'B':>3}  │"
          f"  {'T_total':>23s}  │  {'T_prefill':>23s}  │"
          f"  {'TTFT':>23s}  │  {'T_decode':>23s}")
    h2 = (f"  {'':>5} {'':>4} {'':>3}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}")
    print(h1); print(h2)
    print("  " + "─" * (W - 2))

    for r in results:
        bm, gm = r["base"], r["gist"]
        print(f"  {r['L']:>5} {r['G']:>4} {r['B']:>3}  │"
              f"  {bm.t_total*1e3:7.1f} {gm.t_total*1e3:7.1f}"
              f" {_rx(bm.t_total, gm.t_total):>7s}  │"
              f"  {bm.t_prefill*1e3:7.1f} {gm.t_prefill*1e3:7.1f}"
              f" {_rx(bm.t_prefill, gm.t_prefill):>7s}  │"
              f"  {bm.t_first_token*1e3:7.1f} {gm.t_first_token*1e3:7.1f}"
              f" {_rx(bm.t_first_token, gm.t_first_token):>7s}  │"
              f"  {bm.t_decode*1e3:7.1f} {gm.t_decode*1e3:7.1f}"
              f" {_rx(bm.t_decode, gm.t_decode):>7s}")

    # Throughput sub-table
    print()
    t1 = (f"  {'L':>5} {'G':>4} {'B':>3}  │"
          f"  {'prefill (tok/s)':>23s}  │"
          f"  {'decode (tok/s)':>23s}  │  {'KV mem':>7s}")
    t2 = (f"  {'':>5} {'':>4} {'':>3}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}  │"
          f"  {'Base':>7} {'Gist':>7} {'Ratio':>7}  │  {'(KB)':>7s}")
    print(t1); print(t2)
    print("  " + "─" * (W - 2))

    for r in results:
        bm, gm = r["base"], r["gist"]
        print(f"  {r['L']:>5} {r['G']:>4} {r['B']:>3}  │"
              f"  {bm.prefill_tok_s:7.0f} {gm.prefill_tok_s:7.0f}"
              f" {_rx(gm.prefill_tok_s, bm.prefill_tok_s):>7s}  │"
              f"  {bm.decode_tok_s:7.0f} {gm.decode_tok_s:7.0f}"
              f" {_rx(gm.decode_tok_s, bm.decode_tok_s):>7s}  │"
              f"  {r['gist_cache_kb']:7.1f}")

    # ── Key observations ────────────────────────────────────────
    print()
    print("─" * W)
    print("  Key Observations")
    print("─" * W)

    best_pf  = max(results, key=lambda r: r["sp_prefill"])
    best_tot = max(results, key=lambda r: r["sp_total"])
    max_L    = max(r["L"] for r in results)

    print(f"  • Max prefill speedup : {best_pf['sp_prefill']:.1f}x "
          f"at (L={best_pf['L']}, G={best_pf['G']}, B={best_pf['B']})")
    print(f"  • Max total speedup   : {best_tot['sp_total']:.2f}x "
          f"at (L={best_tot['L']}, G={best_tot['G']}, B={best_tot['B']})")
    print(f"  • KV memory reduction : up to {max_L / K_GIST:.0f}x "
          f"(L={max_L} → {K_GIST} gist tokens)")
    print()
    print("  Patterns to look for:")
    print("    ↑ L (longer prompt)  → bigger prefill & TTFT speedup")
    print("    ↓ G (shorter output) → T_total dominated by prefill → bigger total speedup")
    print("    ↑ B (larger batch)   → prefill more compute-bound → gist advantage clearer")
    print()
    print("  NOTE: Gist tokens are NOT fine-tuned in this demo.  With proper")
    print("  gist-token training, output quality matches baseline while retaining")
    print("  the latency & memory benefits shown above.")
    print()

    if args.csv:
        export_csv(args.csv, results)
        print(f"  Results exported to {args.csv}")
        print()


if __name__ == "__main__":
    main()
