"""Required controls per proposal — enumerated + runnable evaluations.

Controls:
  1 zero cortex
  2 random cortex
  3 trained cortex (primary)
  4 shuffled feedback
  5 zeroed state after consolidation
  6 swapped state between tasks
  7 same query under different states
  8 byte-matched retrieval baseline (exact-match nearest correction)
  9 oracle z* upper bound
"""
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class ControlResult:
    name: str
    accuracy: float
    note: str = ""

def evaluate_controls():
    # stub: actual numbers are produced by pretrain + integration runs
    return []
