"""
gist_bench.config – All tuneable knobs in one place.

Modify this file (or override at import-time) to change models,
sweep grids, or default demo parameters.
"""

import torch

# ═══════════════════════════════════════════════════════════════════
#  Hardware
# ═══════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# ═══════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════
MODEL_NAME = "gpt2"
K_GIST     = 4          # number of gist tokens (≪ prompt length)

# ═══════════════════════════════════════════════════════════════════
#  Demo defaults  (used by gist_cache_demo.py)
# ═══════════════════════════════════════════════════════════════════
MAX_NEW_TOKENS = 50      # tokens to generate per request
NUM_REQUESTS   = 5       # requests sharing the same prompt

# Reusable system prompt (repeated to be long enough to matter)
_SYSTEM_BASE = (
    "You are a helpful, harmless, and honest AI assistant. "
    "You have deep expertise in mathematics, science, history, and programming. "
    "When answering questions, provide clear, concise, and accurate information. "
    "Always cite sources when possible. If unsure, say so rather than fabricating. "
    "Be respectful and patient. Format responses with markdown when appropriate. "
    "Break complex problems into smaller steps. Show work for math problems. "
    "Use code blocks for programming examples. Provide multiple perspectives for "
    "controversial topics. Prioritize safety and ethics in every response. "
)
SYSTEM_PROMPT = _SYSTEM_BASE * 3

USER_QUERIES = [
    "What is the capital of France?",
    "Explain quantum entanglement briefly.",
    "Write a Python function to sort a list.",
    "What causes rainbows?",
    "Summarize the theory of relativity.",
]

# ═══════════════════════════════════════════════════════════════════
#  Sweep defaults  (used by gist_cache_sweep.py)
# ═══════════════════════════════════════════════════════════════════
PROMPT_LENGTHS = [64, 256, 512, 768]    # L: prompt tokens
GEN_LENGTHS    = [8, 64, 128]           # G: max new tokens
BATCH_SIZES    = [1, 4, 16]             # B: concurrent requests

WARMUP_RUNS = 1      # warmup iterations per config (not timed)
BENCH_RUNS  = 3      # timed iterations per config (averaged)

SWEEP_QUERY = "What is the capital of France?"
