"""
gist_bench.metrics – Timing dataclass used by every benchmark path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Metrics:
    """Fine-grained per-run timing breakdown."""

    prefill_tokens:   int   = 0      # tokens consumed during prefill
    generated_tokens: int   = 0      # tokens produced during decode
    t_prefill:        float = 0.0    # prefill forward pass  (s)
    t_first_token:    float = 0.0    # TTFT: start → first token ready  (s)
    t_decode:         float = 0.0    # decode phase after first token  (s)
    t_total:          float = 0.0    # wall-clock start → finish  (s)
    batch_size:       int   = 1

    # ── derived throughput ──────────────────────────────────────

    @property
    def prefill_tok_s(self) -> float:
        """Aggregate prefill throughput (tok/s across batch)."""
        return self.prefill_tokens / self.t_prefill if self.t_prefill > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        """Aggregate decode throughput (tok/s across batch)."""
        return self.generated_tokens / self.t_decode if self.t_decode > 0 else 0.0


# ── aggregation helpers ─────────────────────────────────────────

def avg_metrics(runs: List[Metrics]) -> Metrics:
    """Average timing fields over repeated runs (keeps first run's token counts)."""
    n = len(runs)
    return Metrics(
        prefill_tokens   = runs[0].prefill_tokens,
        generated_tokens = runs[0].generated_tokens,
        t_prefill        = sum(r.t_prefill for r in runs) / n,
        t_first_token    = sum(r.t_first_token for r in runs) / n,
        t_decode         = sum(r.t_decode for r in runs) / n,
        t_total          = sum(r.t_total for r in runs) / n,
        batch_size       = runs[0].batch_size,
    )


def avg_field(metrics: List[Metrics], attr: str) -> float:
    """Average a single numeric attribute across a list of Metrics."""
    return sum(getattr(m, attr) for m in metrics) / len(metrics)


def pack_avgs(metrics: List[Metrics]) -> dict:
    """Return a dict of averaged timing / throughput for summary tables."""
    return dict(
        t_total   = avg_field(metrics, "t_total"),
        t_prefill = avg_field(metrics, "t_prefill"),
        t_first   = avg_field(metrics, "t_first_token"),
        t_decode  = avg_field(metrics, "t_decode"),
        pf_tps    = avg_field(metrics, "prefill_tok_s"),
        dec_tps   = avg_field(metrics, "decode_tok_s"),
    )
