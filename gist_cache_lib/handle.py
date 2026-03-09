"""
gist_cache_lib.handle – Lightweight, serializable cache reference.

A CacheHandle carries just enough metadata to **locate and
validate** a cached gist KV entry on a single device.
"""

from __future__ import annotations

import hashlib
import json
import time as _time
from dataclasses import dataclass, asdict
from typing import Tuple


@dataclass(frozen=True)
class CacheHandle:
    """Lightweight, serialisable reference to a gist-KV cache entry.

    **Shape / layout**
      ``k_gist``, ``n_layers``, ``n_head``, ``head_dim``, ``dtype``,
      ``layout``, ``pool_packed``

    **Local resolution**
      ``cache_id``

    **Lifecycle**
      ``created_at``, ``ttl_s``, ``refcount``
    """

    # ── shape / layout ─────────────────────────────────────────
    k_gist:         int                 # gist-token count
    dtype:          str                 # e.g. "torch.float16"
    model_sig:      str   = ""          # model config hash
    n_layers:       int   = 0
    n_head:         int   = 0
    head_dim:       int   = 0
    layout:         str   = "hf_past_kv"
    pool_packed:    bool  = False

    # ── local resolution ───────────────────────────────────────
    cache_id:       int   = 0           # local store key

    # ── lifecycle ──────────────────────────────────────────────
    created_at:     float = 0.0         # epoch seconds
    ttl_s:          float = 0.0         # 0 = no expiry
    refcount:       int   = 0           # hint; real tracking in store

    # ── shape tuple (for validation) ───────────────────────────

    @property
    def kv_shape(self) -> Tuple[int, ...]:
        """Expected shape of each K or V tensor: (1, n_head, k_gist, head_dim)."""
        return (1, self.n_head, self.k_gist, self.head_dim)

    @property
    def is_expired(self) -> bool:
        if self.ttl_s <= 0:
            return False
        return _time.time() > self.created_at + self.ttl_s

    # ── serialisation helpers ───────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "CacheHandle":
        d = dict(d)
        # backward compat: old handles may have model_hash instead of model_sig
        if "model_hash" in d and "model_sig" not in d:
            d["model_sig"] = d.pop("model_hash")
        elif "model_hash" in d:
            d.pop("model_hash")
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, s: str) -> "CacheHandle":
        return cls.from_dict(json.loads(s))

    # ── mutation helpers (frozen, so return new) ───────────────

    def with_local_id(self, cache_id: int) -> "CacheHandle":
        """Return a copy with a different *cache_id*."""
        return _replace(self, cache_id=cache_id)

    def with_refcount(self, n: int) -> "CacheHandle":
        return _replace(self, refcount=n)


def _replace(handle: CacheHandle, **kw) -> CacheHandle:
    """Frozen-dataclass replace helper."""
    d = handle.to_dict()
    d.update(kw)
    return CacheHandle(**{k: v for k, v in d.items()
                          if k in CacheHandle.__dataclass_fields__})


# ── hashing helpers ────────────────────────────────────────────


def model_config_hash(config) -> str:
    """Produce a short hash from a HuggingFace model config.

    Includes model architecture, size, RoPE params, dtype — enough to
    prevent silent cross-model misuse.
    """
    key_attrs = (
        "model_type", "n_layer", "n_head", "n_embd",
        "num_hidden_layers", "num_attention_heads", "hidden_size",
        "vocab_size",
        # RoPE / position
        "max_position_embeddings", "rotary_dim", "rope_theta",
    )
    parts = []
    for a in key_attrs:
        v = getattr(config, a, None)
        if v is not None:
            parts.append(f"{a}={v}")
    blob = "|".join(sorted(parts))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def prefix_content_hash(token_ids: list, gist_ids: list) -> str:
    """Stable hash of prompt + gist token IDs.

    Two different workers that build the *same* prompt with the *same*
    gist tokens will produce the same ``prefix_hash``.
    """
    blob = f"prompt={token_ids}|gist={gist_ids}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
