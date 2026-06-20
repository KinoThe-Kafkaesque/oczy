"""Shared glue utilities for the Oczy monorepo.

This package contains code that is *not* organ-specific:

- :class:`Episode` --- the canonical shape of an experience episode dict,
  so writers and readers across organs and the experiment layer agree on
  field names.  Organs are not required to import this; it exists as a
  contract and a small validation helper.
- :mod:`oczy_common.text_utils` --- a single tokenizer, stopword set, and
  correction-text heuristics used by the agent glue layer
  (``experiments/``).
- :mod:`oczy_common.bytes` --- a single pickle-based memory-bytes helper
  so ``status()["serialized_bytes"]`` means the same thing for every organ.

Organs themselves keep their own internal tokenizers when those have
organ-specific tuning (the autoencoder's stopword filter and the immune
cortex's trigger extractor are *not* the same problem and intentionally
differ).  This package only formalises the surface where organs meet
agents.
"""

from __future__ import annotations

from .bytes import mem_bytes
from .episode import Episode, EPISODE_FIELDS, validate_episode
from .text_utils import (
    STOPWORDS,
    extract_expected_from_correction,
    tokenize,
)

__all__ = [
    "Episode",
    "EPISODE_FIELDS",
    "validate_episode",
    "mem_bytes",
    "STOPWORDS",
    "extract_expected_from_correction",
    "tokenize",
]