"""Matching and scoring helpers for the organism curriculum driver.

Keeps the matching contract close to ``experiments.eval_suite`` while adding
support for the curriculum's ``match_mode`` hint.
"""

from __future__ import annotations

import re
from typing import Iterable

from experiments.organism_curriculum.dataset import Episode, MatchMode, Probe


_STOPWORDS: frozenset[str] = frozenset(
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
        "such", "what", "which", "who", "whom", "whose", "in",
    }
)


def _normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(str(text).strip().lower().split())


def _token_set(text: str, drop_stopwords: bool = True) -> set[str]:
    """Return alphanumeric tokens, optionally dropping stopwords."""
    tokens = set(re.findall(r"[a-z0-9]+", str(text).lower()))
    if drop_stopwords:
        tokens -= _STOPWORDS
    return tokens


def matches(
    answer: str,
    expected: str,
    ambiguous_token: str | None = None,
    match_mode: MatchMode = "sense",
) -> bool:
    """Return whether ``answer`` matches ``expected`` under ``match_mode``.

    Modes:
      - ``exact``: normalised string equality.
      - ``contains``: substring relationship in either direction.
      - ``sense``: non-empty token overlap after removing stopwords and the
        ambiguous token.
    """
    if match_mode == "exact":
        return _normalize(answer) == _normalize(expected)

    if match_mode == "contains":
        low_a = answer.lower()
        low_e = expected.lower()
        return low_e in low_a or low_a in low_e

    # ``sense`` mode: token overlap, ignoring the ambiguous token.
    ans_tokens = _token_set(answer)
    exp_tokens = _token_set(expected)
    if ambiguous_token:
        amb = ambiguous_token.lower()
        ans_tokens.discard(amb)
        exp_tokens.discard(amb)
    if not ans_tokens or not exp_tokens:
        # If one side has no content tokens, fall back to contains.
        low_a = answer.lower()
        low_e = expected.lower()
        return low_e in low_a or low_a in low_e
    return bool(ans_tokens & exp_tokens)


def probe_matches(answer: str, probe: Probe, episode: Episode) -> bool:
    """Convenience wrapper that extracts the ambiguous token for sense mode."""
    amb = episode.ambiguous_token() if probe.match_mode == "sense" else None
    return matches(answer, probe.expected, ambiguous_token=amb, match_mode=probe.match_mode)


def battery_accuracy(
    results: Iterable[tuple[bool, ...]],
) -> tuple[int, int, float]:
    """Return (correct_count, total, accuracy) for a list of boolean results."""
    total = 0
    correct = 0
    for r in results:
        total += 1
        if r:
            correct += 1
    return correct, total, (correct / total) if total else 0.0


def categorize_results(
    probe_results: list[tuple[Probe, str, bool]]
) -> dict[str, tuple[int, int, float]]:
    """Group probe results by category and compute accuracy per category."""
    by_category: dict[str, list[bool]] = {}
    for probe, _answer, ok in probe_results:
        by_category.setdefault(probe.category, []).append(ok)
    return {
        cat: (sum(items), len(items), sum(items) / len(items) if items else 0.0)
        for cat, items in by_category.items()
    }
