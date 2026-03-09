"""
gist_bench – Gist-Cache benchmarking toolkit.

Provides model setup, three inference strategies (Baseline / KV Reuse /
Gist Cache), fine-grained timing metrics, and reporting utilities.

Quick start::

    from gist_bench import config, engine, metrics, reporting

CacheStore integration::

    from gist_cache_lib import GistCacheStore
    from gist_bench.engine import build_gist_cache_to_store, generate_with_gist_cached
"""

from . import config, engine, metrics, reporting

__all__ = ["config", "engine", "metrics", "reporting"]
