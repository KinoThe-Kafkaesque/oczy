"""Dependency-free shared types for the :mod:`oczy.lm` package.

This module deliberately imports nothing heavyweight (no ``llama_cpp``,
no ``torch``, no ``numpy``) so that lightweight drivers such as
:mod:`oczy.lm.hf_driver` can use the shared steering-surface handle
without paying the import cost of the llama-cpp ctypes boundary.

:class:`ReservedPosition` is the canonical class object; ``cvec_driver``
re-exports it and ``hf_driver`` imports it directly from here, so all
three export paths (``oczy.lm._types``, ``oczy.lm.cvec_driver``, and
``oczy.lm``) share a single class identity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReservedPosition:
    """A reserved KV-position steering surface injected as a literal prefix.

    This is a first-class handle for the soft-prompt / reserved-position
    mechanism: instead of passing around raw strings, callers carry a small
    dataclass that records provenance and (optionally) measured uptake so
    the organism can later learn which positions work.
    """

    text: str
    source: str = "hand_coded"
    exact_uptake_score: float | None = None
    domain_uptake_score: float | None = None


__all__ = ["ReservedPosition"]
