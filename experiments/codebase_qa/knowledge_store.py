"""Key-value knowledge store with keyword or embedding retrieval.

This store supports a deterministic keyword fallback, which is used after a
pickle round-trip when the original embedding function is no longer available.
"""

from __future__ import annotations

import pickle
import re
from typing import Callable

import numpy as np


_LOWERCASE_ALNUM_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens from ``text``."""
    return set(_LOWERCASE_ALNUM_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard index between two token sets, returning 0.0 for empty union."""
    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class KnowledgeStore:
    """Retrievable key-value store for codebase facts.

    Parameters
    ----------
    embed_fn:
        Optional callable ``str -> np.ndarray``. When provided, the store
        embeds every key and value and ranks recalls by cosine similarity.
        When absent, a deterministic keyword overlap metric is used instead.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray] | None = None) -> None:
        self.embed_fn = embed_fn
        self._facts: list[dict] = []

    def add_fact(
        self,
        key: str,
        value: str,
        metadata: dict | None = None,
    ) -> None:
        """Add a fact to the store."""
        fact: dict = {
            "key": key,
            "value": value,
            "metadata": metadata if metadata is not None else {},
        }

        if self.embed_fn is not None:
            fact["key_emb"] = self.embed_fn(key)
            fact["value_emb"] = self.embed_fn(value)
            # Combined key/value embedding is usually the best retrieval signal
            # because it includes both the fact name and its full content.
            fact["kv_emb"] = self.embed_fn(f"{key} | {value}")

        self._facts.append(fact)

    def _keyword_score(self, query: str, fact: dict) -> float:
        query_tokens = _tokenize(query)
        key_tokens = _tokenize(fact["key"])
        value_tokens = _tokenize(fact["value"])
        return 0.5 * _jaccard(query_tokens, key_tokens) + 0.5 * _jaccard(
            query_tokens, value_tokens
        )

    def _embedding_score(self, query: str, fact: dict) -> float:
        assert self.embed_fn is not None
        query_emb = self.embed_fn(query)
        scores = [_cosine(query_emb, fact["kv_emb"])]
        if "key_emb" in fact:
            scores.append(_cosine(query_emb, fact["key_emb"]))
        if "value_emb" in fact:
            scores.append(_cosine(query_emb, fact["value_emb"]))
        return max(scores)

    def recall(self, query: str, k: int = 3) -> list[dict]:
        """Return top-``k`` facts for ``query``, sorted by relevance score."""
        if not self._facts:
            return []

        # Keyword-strong hybrid.  Repository questions usually name their target
        # explicitly (a file, config key, or concept), so keyword overlap is a
        # more reliable signal than the final-layer LM embeddings in this regime.
        # Embeddings remain the fallback when keyword overlap is sparse.
        use_embed = self.embed_fn is not None
        ranked = []
        for fact in self._facts:
            keyword = self._keyword_score(query, fact)
            semantic = self._embedding_score(query, fact) if use_embed else keyword
            if keyword >= 0.20:
                # Strong keyword match: let keyword dominate, embeddings break ties.
                rank_score = keyword + 0.1 * semantic
            else:
                # Weak keyword overlap: prefer semantic similarity.
                rank_score = semantic + 0.5 * keyword
            ranked.append(
                {
                    "key": fact["key"],
                    "value": fact["value"],
                    "score": semantic,
                    "_rank_score": rank_score,
                    "metadata": fact["metadata"],
                }
            )

        ranked.sort(key=lambda item: item["_rank_score"], reverse=True)
        # Remove the private rank key before returning.
        for item in ranked:
            item.pop("_rank_score", None)
        return ranked[:k]

    def format_context(
        self,
        query: str,
        k: int = 3,
        header: str = "Retrieved repository facts:",
    ) -> str:
        """Format top-``k`` recalled facts as a text block."""
        facts = self.recall(query, k=k)
        lines = [header]
        for fact in facts:
            lines.append(f"- Key: {fact['key']}")
            lines.append(f"  Value: {fact['value']}")
        lines.append("")
        return "\n".join(lines)

    def status(self) -> dict:
        """Return serializable status metadata."""
        dim = None
        if self._facts and "key_emb" in self._facts[0]:
            dim = int(self._facts[0]["key_emb"].shape[0])
        return {
            "project": "experiments.codebase_qa.knowledge_store",
            "serialized_bytes": len(pickle.dumps(self)),
            "record_count": len(self._facts),
            "dim": dim,
        }

    def __getstate__(self) -> dict:
        # Drop the embedding function; it may be unpickleable.
        state = self.__dict__.copy()
        state["embed_fn"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.embed_fn = None
