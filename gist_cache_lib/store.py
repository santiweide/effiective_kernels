"""
gist_cache_lib.store – GistCacheStore: put / get / release gist KV caches.

Two storage modes:

  1. **Same-process registry** (always available)
     – ``put_past`` stores tensor references; ``get_past`` returns views.

  2. **Packed KV pool** (always available when pool=True)
     – Copies KV into a single contiguous buffer per entry.
     – ``get_past`` returns zero-copy views into the buffer.

Lifecycle management
~~~~~~~~~~~~~~~~~~~~
Each entry tracks ``last_access`` (for LRU), ``created_at`` (for TTL),
and a cooperative ``refcount``.  The eviction policy is:

* Entries whose ``refcount > 0`` are **never** evicted.
* Among eligible entries, the least-recently-accessed is evicted first.
* Entries past their TTL are evicted eagerly on the next ``put_past``.

The model forward path stays unchanged::

    past = store.get_past(handle)
    out = model(input_ids=..., past_key_values=past, ...)
"""

from __future__ import annotations

import logging
import threading
import time as _time
from typing import Dict, Optional

import torch

from .handle import CacheHandle, model_config_hash
from .pool import KVPool, PastKV

log = logging.getLogger(__name__)


class GistCacheStore:
    """Central registry for gist KV caches on a single device.

    Parameters
    ----------
    device : str
        Target device, e.g. ``"cuda"`` or ``"cuda:0"``.
    capacity : int
        Max number of cache entries.  When exceeded, LRU eviction
        removes the least-recently-used entry with ``refcount == 0``.
    pool : bool
        If True (default), ``put_past`` packs KV into a contiguous
        :class:`KVPool` buffer for better memory locality.
    model_config : object | None
        Optional HuggingFace model config for hash-based validation.
    default_ttl_s : float
        Default time-to-live in seconds (0 = no expiry).
    """

    def __init__(
        self,
        device: str = "cuda",
        capacity: int = 1024,
        pool: bool = True,
        model_config=None,
        default_ttl_s: float = 0.0,
    ):
        self.device = device
        self.capacity = capacity
        self.use_pool = pool
        self.model_hash = model_config_hash(model_config) if model_config else ""
        self.default_ttl_s = default_ttl_s

        self._entries: Dict[int, _Entry] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────

    def put_past(self, past_key_values: PastKV, ttl_s: float | None = None) -> CacheHandle:
        """Register a gist KV cache.  Returns a lightweight handle.

        If ``pool=True`` (default), the KV data is copied into a
        single contiguous buffer for better memory locality.
        """
        n_layers = len(past_key_values)
        k0, v0 = past_key_values[0]
        _batch, n_head, k_gist, head_dim = k0.shape
        dtype_str = str(k0.dtype)
        device_str = str(k0.device)

        if self.use_pool:
            pool = KVPool(past_key_values, copy=True)
            raw = None
        else:
            pool = None
            raw = past_key_values

        now = _time.time()
        effective_ttl = ttl_s if ttl_s is not None else self.default_ttl_s

        with self._lock:
            # evict expired / LRU before inserting
            self._evict_expired_locked(now)
            if len(self._entries) >= self.capacity:
                self._evict_lru_locked()

            cache_id = self._next_id
            self._next_id += 1
            self._entries[cache_id] = _Entry(
                pool=pool, raw=raw,
                created_at=now, last_access=now,
                ttl_s=effective_ttl, refcount=0,
            )

        handle = CacheHandle(
            model_sig=self.model_hash,
            cache_id=cache_id,
            n_layers=n_layers,
            k_gist=k_gist,
            n_head=n_head,
            head_dim=head_dim,
            dtype=dtype_str,
            layout="hf_past_kv",
            pool_packed=self.use_pool,
            created_at=now,
            ttl_s=effective_ttl,
        )
        log.debug("put_past → %s", handle)
        return handle

    def get_past(self, handle: CacheHandle) -> PastKV:
        """Resolve a handle back to a standard ``past_key_values`` tuple.

        When pool-backed, the returned (K, V) tensors are **views** into
        the contiguous buffer (zero copy).

        Updates ``last_access`` for LRU tracking.
        """
        self._validate(handle)
        entry = self._entries[handle.cache_id]
        entry.last_access = _time.time()
        if entry.pool is not None:
            return entry.pool.as_past_kv()
        assert entry.raw is not None
        return entry.raw

    def release(self, handle: CacheHandle) -> None:
        """Remove an entry and free its GPU memory."""
        with self._lock:
            entry = self._entries.pop(handle.cache_id, None)
        if entry is None:
            log.warning("release: cache_id=%d not found", handle.cache_id)
        else:
            log.debug("released cache_id=%d", handle.cache_id)

    # ── refcount (cooperative) ─────────────────────────────────

    def acquire(self, handle: CacheHandle) -> None:
        """Increment refcount (prevent LRU eviction)."""
        self._validate(handle)
        entry = self._entries[handle.cache_id]
        with self._lock:
            entry.refcount += 1

    def release_ref(self, handle: CacheHandle) -> None:
        """Decrement refcount."""
        if handle.cache_id in self._entries:
            entry = self._entries[handle.cache_id]
            with self._lock:
                entry.refcount = max(0, entry.refcount - 1)

    # ── introspection ──────────────────────────────────────────

    @property
    def num_entries(self) -> int:
        return len(self._entries)

    def total_bytes(self) -> int:
        """Total GPU bytes across all entries."""
        total = 0
        for e in self._entries.values():
            if e.pool is not None:
                total += e.pool.nbytes
            elif e.raw is not None:
                total += sum(
                    k.nelement() * k.element_size() +
                    v.nelement() * v.element_size()
                    for k, v in e.raw
                )
        return total

    def summary(self) -> str:
        n = self.num_entries
        mb = self.total_bytes() / (1024 * 1024)
        return (f"GistCacheStore(entries={n}, "
                f"total={mb:.2f} MB, pool={self.use_pool}, "
                f"device={self.device})")

    # ── eviction ───────────────────────────────────────────────

    def _evict_expired_locked(self, now: float) -> int:
        """Remove entries past their TTL.  Caller must hold ``_lock``."""
        expired = [
            cid for cid, e in self._entries.items()
            if e.ttl_s > 0 and now > e.created_at + e.ttl_s
        ]
        for cid in expired:
            del self._entries[cid]
        if expired:
            log.debug("evicted %d expired entries", len(expired))
        return len(expired)

    def _evict_lru_locked(self) -> bool:
        """Remove the least-recently-used entry with refcount == 0.

        Caller must hold ``_lock``.  Returns True if an entry was evicted.
        """
        candidates = [
            (cid, e) for cid, e in self._entries.items() if e.refcount <= 0
        ]
        if not candidates:
            log.warning("GistCacheStore capacity (%d) reached; "
                        "all entries have refcount > 0, cannot evict",
                        self.capacity)
            return False
        # LRU: oldest last_access
        cid, _ = min(candidates, key=lambda x: x[1].last_access)
        del self._entries[cid]
        log.debug("LRU-evicted cache_id=%d", cid)
        return True

    # ── internal ───────────────────────────────────────────────

    def _validate(self, handle: CacheHandle):
        if handle.cache_id not in self._entries:
            raise KeyError(
                f"cache_id={handle.cache_id} not found in store "
                f"({self.num_entries} entries)")
        if self.model_hash and handle.model_sig and \
           handle.model_sig != self.model_hash:
            raise ValueError(
                f"Model hash mismatch: handle={handle.model_sig}, "
                f"store={self.model_hash}.  "
                f"This handle belongs to a different model.")


class _Entry:
    """Internal storage record with lifecycle metadata."""
    __slots__ = ("pool", "raw", "created_at", "last_access", "ttl_s", "refcount")

    def __init__(
        self,
        pool: Optional[KVPool],
        raw: Optional[PastKV],
        created_at: float = 0.0,
        last_access: float = 0.0,
        ttl_s: float = 0.0,
        refcount: int = 0,
    ):
        self.pool = pool
        self.raw = raw
        self.created_at = created_at
        self.last_access = last_access
        self.ttl_s = ttl_s
        self.refcount = refcount
