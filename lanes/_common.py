"""Shared utilities for lane modules."""
from __future__ import annotations

import functools

import numpy as np


def cosine(a, b) -> float:
    """Cosine similarity with safe zero-norm handling."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def lane_measure(func):
    """Wrap a lane measure() function with fail-soft try/except -> float('nan')."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return float("nan")

    return wrapper


MARKER_BEARING_CORRECTIONS = (
    "no actually the sky is blue",
    "wrong paris is the capital of france",
    "correction two plus two is four",
    "actually jupiter is the largest planet",
)
