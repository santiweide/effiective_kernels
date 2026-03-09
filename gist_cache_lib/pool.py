"""
gist_cache_lib.pool – Packed KV Pool for contiguous GPU storage.

Instead of keeping per-layer ``(K, V)`` tuples with separate CUDA
allocations, the pool packs *all* layers' K and V into **one**
contiguous buffer.  This gives:

* Fewer CUDA allocations
* Better memory locality for prefetching
* Trivial ``as_strided`` / ``view`` to reconstruct per-layer K/V

Layout (single buffer, interleaved K/V)::

    buf shape: [n_layers, 2, batch, n_head, k_gist, head_dim]
               ─────────  ↑
                          0 = K, 1 = V

After ``get_past()`` the caller receives the standard HF format::

    tuple[ (K_0, V_0), (K_1, V_1), … ]     each (batch, n_head, k_gist, head_dim)

The views share storage with the pool → **zero copy**.
"""

from __future__ import annotations

from typing import Tuple

import torch

# Type alias for standard HuggingFace past_key_values
PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class KVPool:
    """Contiguous GPU buffer that holds gist KV for one cache entry.

    Parameters
    ----------
    past_key_values : PastKV
        Standard HF past_key_values to pack.
    copy : bool
        If True (default), data is copied into a fresh contiguous buffer.
        If False, wrapping is skipped and the caller must ensure the
        original tensors stay alive (useful when you just need the views).
    """

    def __init__(self, past_key_values: PastKV, *, copy: bool = True):
        n_layers = len(past_key_values)
        k0, v0 = past_key_values[0]
        batch, n_head, seq_len, head_dim = k0.shape

        self.n_layers = n_layers
        self.batch    = batch
        self.n_head   = n_head
        self.k_gist   = seq_len
        self.head_dim = head_dim
        self.dtype    = k0.dtype
        self.device   = k0.device

        if copy:
            # Allocate single contiguous buffer
            self.buf = torch.empty(
                (n_layers, 2, batch, n_head, seq_len, head_dim),
                dtype=self.dtype, device=self.device,
            )
            for i, (k, v) in enumerate(past_key_values):
                self.buf[i, 0].copy_(k)
                self.buf[i, 1].copy_(v)
        else:
            # Stack without copy (tensors must already be contiguous)
            ks = torch.stack([k for k, _ in past_key_values])  # (L, B, H, S, D)
            vs = torch.stack([v for _, v in past_key_values])
            self.buf = torch.stack([ks, vs], dim=1)             # (L, 2, B, H, S, D)

    # ── views back to HF format ────────────────────────────────

    def as_past_kv(self) -> PastKV:
        """Return per-layer (K, V) views into the contiguous buffer.

        Views share storage → zero copy.
        """
        return tuple(
            (self.buf[i, 0], self.buf[i, 1])
            for i in range(self.n_layers)
        )

    # ── size introspection ─────────────────────────────────────

    @property
    def nbytes(self) -> int:
        return self.buf.nelement() * self.buf.element_size()

    @property
    def shape_summary(self) -> str:
        return (f"KVPool(layers={self.n_layers}, n_head={self.n_head}, "
                f"k_gist={self.k_gist}, head_dim={self.head_dim}, "
                f"buf={list(self.buf.shape)}, "
                f"{self.nbytes / 1024:.1f} KB)")
