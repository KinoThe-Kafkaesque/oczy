"""Key-value knowledge store with keyword or embedding retrieval.

This store supports a deterministic keyword fallback, which is used after a
pickle round-trip when the original embedding function is no longer available.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from typing import Callable

import numpy as np


# Common English stopwords plus conversational/query words.  Removing them
# prevents value-side filler text (e.g., 'what becomes a memory update') from
# matching question stopwords ('what', 'is', 'does') and swamping retrieval.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "dare", "ought", "used", "to", "of", "in", "on", "at",
    "for", "with", "from", "by", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "among", "within",
    "what", "which", "who", "when", "where", "why", "how", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
}

_LOWERCASE_ALNUM_RE = re.compile(r"[a-z0-9]+")


def _stem(token: str) -> str:
    """Minimal stemming: strip trailing plural suffixes.

    This lets queries like 'reject' match fact values containing 'rejects'
    without pulling in a full natural-language stemmer.
    """
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es"):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from ``text``.

    CamelCase names are split into separate words so that queries like
    'NeuralHippocampus' match facts keyed on 'neural hippocampus'.
    Underscores and hyphens are treated as token boundaries by the
    alphanumeric regular expression.  Stopwords are removed and tokens are
    lightly stemmed.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", r" ", text)
    tokens = _LOWERCASE_ALNUM_RE.findall(spaced.lower())
    return [_stem(tok) for tok in tokens if tok not in _STOPWORDS]


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two dense vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class _KeywordIndex:
    """Deterministic TF-IDF-style keyword index over fact keys and values.

    Rare tokens (e.g., a package or organ name) are weighted more heavily than
    generic tokens (e.g., 'the', 'value', 'role'), which fixes retrieval when
    many facts share a common schema word.
    """

    def __init__(self, facts: list[dict]) -> None:
        self.facts = facts
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_vectors: list[dict[str, float]] = []
        self._build()

    def _build(self) -> None:
        # Gather document frequency across all fact keys and values.
        df: Counter[str] = Counter()
        doc_tokens: list[list[str]] = []
        for fact in self.facts:
            tokens = _tokenize(fact["key"]) + _tokenize(fact["value"])
            unique = set(tokens)
            df.update(unique)
            doc_tokens.append(tokens)

        # Build vocabulary and IDF weights.
        vocab = sorted(df.keys())
        self._vocab = {tok: idx for idx, tok in enumerate(vocab)}
        n_docs = max(len(self.facts), 1)
        self._idf = {
            tok: math.log(n_docs / (1.0 + freq)) for tok, freq in df.items()
        }

        # Precompute weighted TF-IDF vectors for each fact.
        for tokens in doc_tokens:
            counts = Counter(tokens)
            vec = {tok: counts[tok] * self._idf[tok] for tok in counts}
            self._doc_vectors.append(vec)

    def _vectorize(self, text: str) -> dict[str, float]:
        counts = Counter(_tokenize(text))
        return {tok: counts[tok] * self._idf.get(tok, 0.0) for tok in counts}

    def score(self, query: str, fact_idx: int) -> float:
        query_vec = self._vectorize(query)
        doc_vec = self._doc_vectors[fact_idx]
        if not query_vec or not doc_vec:
            return 0.0
        # Sparse cosine using only tokens present in either vector.
        all_tokens = set(query_vec.keys()) | set(doc_vec.keys())
        q = np.array([query_vec.get(tok, 0.0) for tok in all_tokens], dtype=np.float32)
        d = np.array([doc_vec.get(tok, 0.0) for tok in all_tokens], dtype=np.float32)
        return _cosine_dense(q, d)


class KnowledgeStore:
    """Retrievable key-value store for codebase facts.

    Parameters
    ----------
    embed_fn:
        Optional callable ``str -> np.ndarray``. When provided, the store
        embeds every key and value and ranks recalls by cosine similarity.
        When absent, a deterministic TF-IDF keyword index is used instead.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray] | None = None) -> None:
        self.embed_fn = embed_fn
        self._facts: list[dict] = []
        self._keyword_index: _KeywordIndex | None = None

    def _rebuild_keyword_index(self) -> None:
        self._keyword_index = _KeywordIndex(self._facts)

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
        self._keyword_index = None  # invalidated; rebuilt lazily on recall

    def _keyword_score(self, query: str, fact_idx: int) -> float:
        if self._keyword_index is None:
            self._rebuild_keyword_index()
        assert self._keyword_index is not None
        return self._keyword_index.score(query, fact_idx)

    def _embedding_score(self, query: str, fact: dict) -> float:
        assert self.embed_fn is not None
        query_emb = self.embed_fn(query)
        scores = [_cosine_dense(query_emb, fact["kv_emb"])]
        if "key_emb" in fact:
            scores.append(_cosine_dense(query_emb, fact["key_emb"]))
        if "value_emb" in fact:
            scores.append(_cosine_dense(query_emb, fact["value_emb"]))
        return max(scores)

    def recall(self, query: str, k: int = 3) -> list[dict]:
        """Return top-``k`` facts for ``query``, sorted by relevance score."""
        if not self._facts:
            return []

        # Keyword score is the primary signal for this corpus. Embeddings are
        # kept as a fallback where overlap is sparse.
        use_embed = self.embed_fn is not None
        ranked = []
        for idx, fact in enumerate(self._facts):
            keyword = self._keyword_score(query, idx)
            semantic = self._embedding_score(query, fact) if use_embed else keyword
            if keyword >= 0.15:
                # Strong keyword match dominates; embeddings only break ties.
                rank_score = keyword + 0.1 * semantic
            else:
                # Weak keyword overlap: lean on semantic similarity.
                rank_score = semantic + 0.5 * keyword
            ranked.append(
                {
                    "key": fact["key"],
                    "value": fact["value"],
                    "score": semantic if use_embed else keyword,
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
        state["_keyword_index"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.embed_fn = None
        self._keyword_index = None
