"""Oczy LM perception layer.

This package sits at the IO boundary of :class:`OrganismAgent` (and the
ablation agents in ``experiments/baselines.py``).  It wraps a small
local LLM and exposes two functions:

* :meth:`LanguageAdapter.nl_to_episode`  -- parse a free-form NL
  utterance into a canonical :class:`oczy_common.episode.Episode`
  dict so the organism only ever sees structured state.
* :meth:`LanguageAdapter.episode_to_nl`  -- render an Episode back as
  natural English so the organism's structured output reaches the user
  as a sentence rather than a JSON blob.

Default backend configuration is the Pareto-optimal one identified by
``bench_cross_backend.py`` (see the 2026-06-22 session log):

    repo_id  = LiquidAI/LFM2.5-1.2B-Instruct-GGUF
    filename = LFM2.5-1.2B-Instruct-Q4_K_M.gguf
    n_ctx    = 1024, n_threads = 4, use_mmap = True

On this host (i7-1260P, no usable GPU) that hits **38 tok/s, 1.6 GB peak
RSS, 697 MB on disc** -- the best of every config we benched.
"""

from __future__ import annotations

from .adapter import LanguageAdapter, LanguageAdapterConfig
from .cvec_driver import CVecDriverConfig, LlamaCVecDriver

__all__ = [
    "LanguageAdapter",
    "LanguageAdapterConfig",
    "CVecDriverConfig",
    "LlamaCVecDriver",
]