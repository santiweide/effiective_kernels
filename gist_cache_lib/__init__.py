"""
gist_cache_lib – GPU KV Cache Library for Gist-Token Inference.

::

    from gist_cache_lib import GistCacheStore, CacheHandle

    store = GistCacheStore(device="cuda")
    handle = store.put_past(gist_cache_tensors)
    past = store.get_past(handle)
    out = model(input_ids=..., past_key_values=past, ...)
    store.release(handle)

Components:

1. **GistCacheStore** – per-device registry (LRU/TTL/refcount).
2. **KVPool** – contiguous buffer packing.
"""

from .handle import CacheHandle, model_config_hash, prefix_content_hash
from .store import GistCacheStore
from .pool import KVPool

__all__ = [
    "CacheHandle",
    "GistCacheStore",
    "KVPool",
    "model_config_hash",
    "prefix_content_hash",
]

__version__ = "0.3.0"
