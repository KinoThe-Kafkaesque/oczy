"""Shared text utilities for the Oczy glue layer.

Five different ``_tokenize`` helpers and four different copies of
``_extract_expected_from_correction`` grew up across ``organism.py``,
``baselines.py``, ``autoencoder.py``, ``critic.py``, and ``hypernet.py``.
They existed because each organ wanted slightly different behaviour,
but the *glue layer* (the agent classes in ``experiments/``) ended up
duplicating the simplest versions and silently drifting away from the
richer organism heuristics.

This module is the single source of truth for the glue layer.  Organs
keep their own internal tokenizers when they need organ-specific tuning
--- the experience autoencoder's stopword filter and the skill immune
cortex's trigger extractor are *not* the same problem.
"""

from __future__ import annotations

import re


#: Tokens that carry no semantic signal at the agent-glue level.  This
#: is the union of the four historical private ``_STOPWORDS`` sets with
#: obvious duplicates removed.  Organs that need stricter or looser
#: filtering keep their own local lists.
STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "but", "for", "with", "on", "at", "from", "as",
        "it", "its", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "our", "their",
        "do", "does", "did", "done", "doing",
        "have", "has", "had",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "must", "not", "no", "yes", "so", "if", "then", "than", "very", "just",
        "only", "also", "about", "into", "through", "over", "under", "again",
        "further", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "what", "which", "who", "whom", "whose",
        # correction-sentence boilerplate that we never want as a label
        "no", "here", "means", "in", "product", "should be", "use",
        "refers", "now",
    }
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization, dropping stopwords.

    ``len(tok) >= 2`` matches every historical caller's minimum length.
    Single-character tokens are noise at the glue layer.
    """
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in STOPWORDS
    ]


# Correction-templates ordered by specificity: the most specific match
# (longest marker) wins.  Kept as plain strings rather than regex for
# readability and so the comparison is anchored at the first occurrence.
_CORRECTION_TEMPLATES = (
    r"(.*?)\s+means\s+(.*)",
    r"(.*?)\s+is\s+(.*)",
    r"(.*?)\s+refers to\s+(.*)",
    r"(.*?)\s+should be\s+(.*)",
    r"use\s+(.*)",
)

_CORRECTION_MARKERS = (
    "no, ",
    "no:",
    "wrong, ",
    "wrong:",
    "correction:",
    "correct:",
    "expected:",
)


def extract_expected_from_correction(correction: str) -> str:
    """Best-effort recovery of the corrected label from a correction sentence.

    Mirrors the heuristic that lived in :class:`OrganismAgent` before
    extraction; the four baseline agents in ``experiments/baselines.py``
    had a thinner copy each, which silently diverged from the organism's
    richer version.

    The order of preference is:
    1. The right-hand side of "X means/is/refers to/should be Y" (where
       ``Y`` is the definition) unless ``Y`` is much shorter than ``X``.
    2. The right-hand side of a "use Y" imperative.
    3. The whole correction text with leading markers stripped.
    """
    text = correction.lower().strip().strip(".'\"")

    for marker in _CORRECTION_MARKERS:
        if text.startswith(marker):
            text = text[len(marker):].strip()
            break

    for template in _CORRECTION_TEMPLATES:
        match = re.search(template, text)
        if not match:
            continue
        left, right = match.group(1), match.group(2)
        # Prefer the right-hand side (definition) unless it is much
        # shorter than the left-hand side (the term being defined).
        candidate = right if len(right) >= len(left) / 2 else left
        candidate = candidate.strip().strip(".'\"")
        if candidate:
            return candidate

    return correction.strip().strip(".'\"")