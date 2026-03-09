"""
gist_bench.reporting – Printing helpers, ratio formatting, CSV export.
"""

from __future__ import annotations

import csv
from typing import List

from .metrics import Metrics


# ═══════════════════════════════════════════════════════════════════
#  Ratio helpers
# ═══════════════════════════════════════════════════════════════════

def ratio(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def ratio_str(a: float, b: float) -> str:
    r = ratio(a, b)
    return f"{r:.2f}x" if r > 0 else "n/a"


# ═══════════════════════════════════════════════════════════════════
#  Per-request table helpers  (demo)
# ═══════════════════════════════════════════════════════════════════

DEMO_HDR = (f"  {'#':>3}  {'T_total':>9} {'T_prefill':>10} {'TTFT':>9} "
            f"{'T_decode':>9} {'prefill':>9} {'decode':>9}  Query")
DEMO_UNITS = (f"  {'':>3}  {'(ms)':>9} {'(ms)':>10} {'(ms)':>9} "
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


# ═══════════════════════════════════════════════════════════════════
#  3-way summary table  (demo)
# ═══════════════════════════════════════════════════════════════════

_ROW = "  {label:<16} {total:>9} {prefill:>10} {ttft:>9} {decode:>9} {pf:>9} {dec:>9} {mem:>8}"


def print_summary_table(
    n_requests: int,
    packs: List[tuple],          # [(label, avg_dict, mem_kb), …]
    baseline_pack: dict,         # avg_dict of baseline
    comparisons: List[tuple],    # [(label, avg_dict), …]  speedup vs baseline
    gist_vs_kv: tuple | None = None,  # (gist_dict, kv_dict, mem_ratio_str)
):
    W = 96
    print("=" * W)
    print(f"  Summary  (averages over {n_requests} requests)")
    print("=" * W)

    print(_ROW.format(label="", total="T_total", prefill="T_prefill",
                      ttft="TTFT", decode="T_decode", pf="prefill",
                      dec="decode", mem="KV mem"))
    print(_ROW.format(label="", total="(ms)", prefill="(ms)",
                      ttft="(ms)", decode="(ms)", pf="(tok/s)",
                      dec="(tok/s)", mem="(KB)"))
    print("  " + "-" * (W - 2))

    for label, d, mem_kb in packs:
        print(_ROW.format(
            label=label,
            total=f"{d['t_total']*1e3:.1f}",
            prefill=f"{d['t_prefill']*1e3:.1f}",
            ttft=f"{d['t_first']*1e3:.1f}",
            decode=f"{d['t_decode']*1e3:.1f}",
            pf=f"{d['pf_tps']:.0f}",
            dec=f"{d['dec_tps']:.0f}",
            mem=f"{mem_kb:.1f}",
        ))
    print("  " + "-" * (W - 2))

    b = baseline_pack
    print("  Speedup vs Baseline:")
    for label, d in comparisons:
        print(_ROW.format(
            label=f"  {label}",
            total=ratio_str(b["t_total"], d["t_total"]),
            prefill=ratio_str(b["t_prefill"], d["t_prefill"]),
            ttft=ratio_str(b["t_first"], d["t_first"]),
            decode=ratio_str(b["t_decode"], d["t_decode"]),
            pf=ratio_str(d["pf_tps"], b["pf_tps"]),
            dec=ratio_str(d["dec_tps"], b["dec_tps"]),
            mem="",
        ))
    print()

    if gist_vs_kv is not None:
        g, rv, mem_ratio = gist_vs_kv
        print("  Gist Cache vs KV Reuse:")
        print(_ROW.format(
            label="  Ratio",
            total=ratio_str(rv["t_total"], g["t_total"]),
            prefill=ratio_str(rv["t_prefill"], g["t_prefill"]),
            ttft=ratio_str(rv["t_first"], g["t_first"]),
            decode=ratio_str(rv["t_decode"], g["t_decode"]),
            pf=ratio_str(g["pf_tps"], rv["pf_tps"]),
            dec=ratio_str(g["dec_tps"], rv["dec_tps"]),
            mem=mem_ratio,
        ))
        print()


# ═══════════════════════════════════════════════════════════════════
#  CSV export  (sweep)
# ═══════════════════════════════════════════════════════════════════

def export_csv(path: str, results: list):
    """Write per-config sweep results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "L", "G", "B",
            "base_T_total_ms", "base_T_prefill_ms", "base_TTFT_ms",
            "base_T_decode_ms", "base_prefill_tok_s", "base_decode_tok_s",
            "gist_T_total_ms", "gist_T_prefill_ms", "gist_TTFT_ms",
            "gist_T_decode_ms", "gist_prefill_tok_s", "gist_decode_tok_s",
            "speedup_total", "speedup_prefill", "speedup_TTFT",
            "gist_cache_KB",
        ])
        for r in results:
            bm, gm = r["base"], r["gist"]
            w.writerow([
                r["L"], r["G"], r["B"],
                f"{bm.t_total*1e3:.2f}",        f"{bm.t_prefill*1e3:.2f}",
                f"{bm.t_first_token*1e3:.2f}",   f"{bm.t_decode*1e3:.2f}",
                f"{bm.prefill_tok_s:.1f}",        f"{bm.decode_tok_s:.1f}",
                f"{gm.t_total*1e3:.2f}",         f"{gm.t_prefill*1e3:.2f}",
                f"{gm.t_first_token*1e3:.2f}",   f"{gm.t_decode*1e3:.2f}",
                f"{gm.prefill_tok_s:.1f}",        f"{gm.decode_tok_s:.1f}",
                f"{r['sp_total']:.3f}",           f"{r['sp_prefill']:.3f}",
                f"{r['sp_ttft']:.3f}",
                f"{r['gist_cache_kb']:.1f}",
            ])
