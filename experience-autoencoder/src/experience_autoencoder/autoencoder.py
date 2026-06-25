"""Experience Autoencoder prototype.

Encodes a structured episode into a small latent delta vector Δz.
A pseudo-learned random projection is used for the measurement step, and
decoding uses orthogonal matching pursuit to recover sparse token-level
features from Δz.

This implementation is intentionally tiny and dependency-free beyond NumPy.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from typing import Any

import numpy as np

HEBBIAN_LR = 0.01


LATENT_DIM = 32
OUTCOME_DIM = 4
RESIDUAL_DIM = LATENT_DIM - OUTCOME_DIM
NUM_SOURCES = 4
MAX_VOCAB = 256
DECODE_SPARSITY = 10
HIDDEN_DELTA_SCALE = 1.0

_OUTCOME_LABELS = ["accepted", "corrected", "failed", "unknown"]
_OUTCOME_TO_IDX = {label: i for i, label in enumerate(_OUTCOME_LABELS)}

_FAILURE_MAP = {
    "accepted": "none",
    "corrected": "semantic_misgrounding",
    "failed": "execution_error",
    "unknown": "unknown",
}

_SOURCE_NAMES = ["situation", "model_answer", "correction", "revised_answer"]
_SOURCE_WEIGHTS = np.array([1.0, 1.0, 2.0, 1.25], dtype=float)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "for",
    "with",
    "as",
    "this",
    "that",
    "it",
    "its",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 2 and tok not in _STOPWORDS]


# Back-compat mapping from canonical Episode keys (oczy_common.episode.Episode)
# to the autoencoder's internal source names. The autoencoder's curriculum/eval
# scripts emit the legacy field names; canonical Episodes emit query/answer/
# corrected_answer. We accept either, preferring the legacy field when both
# are present so existing pipelines stay byte-for-byte unchanged.
_CANONICAL_ALIASES: dict[str, str] = {
    "query": "situation",
    "answer": "model_answer",
    "corrected_answer": "revised_answer",
}


def _normalize_episode(episode: dict[str, Any]) -> dict[str, str]:
    """Return a stringified copy of ``episode`` with canonical aliases folded in.

    For each alias pair (canonical -> legacy), if the legacy source field is
    missing or empty and the canonical field has content, the canonical value
    is copied across. This lets :class:`ExperienceEncoder` consume both shapes
    without callers needing to know which schema is canonical.
    """
    normalized: dict[str, str] = {k: str(v) for k, v in episode.items()}
    for canonical, legacy in _CANONICAL_ALIASES.items():
        if not normalized.get(legacy) and normalized.get(canonical):
            normalized[legacy] = normalized[canonical]
    return normalized


def _make_sensing_matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(RESIDUAL_DIM, NUM_SOURCES * MAX_VOCAB))
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (A / norms).astype(float)

def _make_hidden_sensing_matrix(seed: int, d_hidden: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((RESIDUAL_DIM, d_hidden))
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (A / norms).astype(float)


def _outcome_vector(outcome: str) -> np.ndarray:
    outcome_idx = _OUTCOME_TO_IDX.get(outcome, _OUTCOME_TO_IDX["unknown"])
    outcome_vec = np.full(OUTCOME_DIM, -0.2, dtype=float)
    outcome_vec[outcome_idx] = 0.8
    return outcome_vec


class ExperienceEncoder:
    """Encode an episode dict into a small latent delta vector Δz."""

    def __init__(self, vocab: dict[str, int], sensing_matrix: np.ndarray) -> None:
        self._token_to_idx = vocab
        self._idx_to_token: dict[int, str] = {i: t for t, i in vocab.items()}
        self._A = sensing_matrix

    def _ensure_token(self, token: str) -> int | None:
        if token in self._token_to_idx:
            return self._token_to_idx[token]
        if len(self._token_to_idx) < MAX_VOCAB:
            idx = len(self._token_to_idx)
            self._token_to_idx[token] = idx
            self._idx_to_token[idx] = token
            return idx
        return None

    def extract_features(self, episode: dict[str, Any]) -> np.ndarray:
        """Build the weighted bag-of-words feature vector for one episode.

        This is the encoder's internal input signal. Both :meth:`encode` and
        :class:`ExperienceAutoencoder.train_step` consume it. The shared
        vocab is mutated as a side effect (new tokens are assigned indices).

        Both legacy field names (``situation``/``model_answer``/
        ``revised_answer``) and canonical Episode names (``query``/
        ``answer``/``corrected_answer``) are accepted via
        :func:`_normalize_episode`.
        """
        normalized = _normalize_episode(episode)
        f = np.zeros(NUM_SOURCES * MAX_VOCAB, dtype=float)
        for s_i, source in enumerate(_SOURCE_NAMES):
            text = normalized.get(source, "")
            counts = Counter(_tokenize(text))
            for token, count in counts.items():
                idx = self._ensure_token(token)
                if idx is not None:
                    f[s_i * MAX_VOCAB + idx] += count * _SOURCE_WEIGHTS[s_i]
        return f

    def encode(self, episode: dict[str, Any]) -> np.ndarray:
        normalized = _normalize_episode(episode)
        f = self.extract_features(normalized)

        residual = self._A @ f
        scale = 1.0 + np.linalg.norm(f)
        residual = np.tanh(residual / scale)

        outcome_vec = _outcome_vector(normalized.get("outcome", "unknown"))

        delta_z = np.empty(LATENT_DIM, dtype=float)
        delta_z[:OUTCOME_DIM] = outcome_vec
        delta_z[OUTCOME_DIM:] = residual
        return delta_z

    @property
    def vocab(self) -> dict[str, int]:
        return self._token_to_idx


class ExperienceDecoder:
    """Reconstruct human-readable learning signals from a latent delta Δz."""

    def __init__(self, vocab: dict[str, int], sensing_matrix: np.ndarray) -> None:
        self._token_to_idx = vocab
        self._A = sensing_matrix

    def decode(self, delta_z: np.ndarray) -> dict[str, Any]:
        delta_z = np.asarray(delta_z, dtype=float).reshape(-1)
        outcome_idx = int(np.argmax(delta_z[:OUTCOME_DIM]))
        outcome = _OUTCOME_LABELS[outcome_idx]
        failure_class = _extract_failure_class(outcome, delta_z)

        residual = delta_z[OUTCOME_DIM:]
        idx_to_token = {i: t for t, i in self._token_to_idx.items()}
        allowed = {
            s_i * MAX_VOCAB + tok_idx
            for s_i in range(NUM_SOURCES)
            for tok_idx in idx_to_token
        }
        x = _omp(self._A, residual, sparsity=DECODE_SPARSITY, allowed=allowed)

        # Gather selected tokens per source and overall.
        selected: dict[int, list[tuple[str, float]]] = {s: [] for s in range(NUM_SOURCES)}
        global_selection: list[tuple[str, float]] = []
        for flat_idx, coeff in enumerate(x):
            if abs(coeff) < 1e-8:
                continue
            s_i = flat_idx // MAX_VOCAB
            tok_idx = flat_idx % MAX_VOCAB
            token = idx_to_token.get(tok_idx)
            if token is None:
                continue
            selected[s_i].append((token, float(coeff)))
            global_selection.append((token, float(coeff)))

        # Trigger conditions: high-salience tokens, prefer correction block.
        trigger_conditions = self._select_tokens(
            global_selection,
            preferred_sources=selected.get(2, []) + selected.get(0, []),
            top_k=8,
        )

        # Corrected behavior hint: map important correction tokens to revised targets.
        corrected_hint: dict[str, str] = {}
        correction_tokens = self._rank_tokens(selected.get(2, []))
        revised_tokens = self._rank_tokens(selected.get(3, []))
        fallback_target = _target_label_for(failure_class)
        for token in correction_tokens[:5]:
            if revised_tokens:
                target = revised_tokens[0]
            else:
                target = fallback_target
            corrected_hint[token] = target

        # Counterexamples: concrete wrong-way statements inferred from model-answer tokens.
        counterexamples: list[str] = []
        model_tokens = self._rank_tokens(selected.get(1, []))
        for token in (model_tokens[:3] if model_tokens else correction_tokens[:3]):
            counterexamples.append(
                f"Avoid interpreting '{token}' as in: '{failure_class}' context."
            )

        return {
            "corrected_behavior_hint": corrected_hint,
            "failure_class": failure_class,
            "trigger_conditions": trigger_conditions,
            "counterexamples": counterexamples,
        }

    @staticmethod
    def _rank_tokens(source_pairs: list[tuple[str, float]]) -> list[str]:
        pairs = sorted(source_pairs, key=lambda p: abs(p[1]), reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for token, _ in pairs:
            if token not in seen:
                seen.add(token)
                out.append(token)
        return out

    def _select_tokens(
        self,
        global_pairs: list[tuple[str, float]],
        preferred_sources: list[tuple[str, float]],
        top_k: int,
    ) -> list[str]:
        preferred_tokens = self._rank_tokens(preferred_sources)
        global_tokens = self._rank_tokens(global_pairs)
        combined = preferred_tokens + [t for t in global_tokens if t not in preferred_tokens]
        return combined[:top_k]


def _extract_failure_class(outcome: str, delta_z: np.ndarray) -> str:
    base = _FAILURE_MAP.get(outcome, "unknown")
    if outcome == "corrected":
        # Use the residual sign of a stable dimension as a cheap tie-breaker.
        tail = delta_z[OUTCOME_DIM:]
        if tail.size > 0 and float(tail[0]) < -0.1:
            return "fact_correction"
    return base


def _target_label_for(failure_class: str) -> str:
    return {
        "semantic_misgrounding": "use_intended_meaning",
        "fact_correction": "use_correct_fact",
        "execution_error": "avoid_failing_approach",
        "none": "maintain_behavior",
    }.get(failure_class, "revise_behavior")


def _omp(A: np.ndarray, b: np.ndarray, sparsity: int, allowed: set[int] | None = None) -> np.ndarray:
    """Very small orthogonal matching pursuit for sparse token recovery."""
    x = np.zeros(A.shape[1], dtype=float)
    residual = np.asarray(b, dtype=float).copy()
    selected: list[int] = []
    if allowed is None:
        allowed = set(range(A.shape[1]))
    for _ in range(min(sparsity, A.shape[0] - 1)):
        correlations = A.T @ residual
        # Restrict to allowed (known vocabulary) and exclude already-selected indices.
        mask = np.zeros_like(correlations, dtype=bool)
        mask[list(allowed)] = True
        correlations[~mask] = 0.0
        for idx in selected:
            correlations[idx] = 0.0
        chosen = int(np.argmax(np.abs(correlations)))
        if abs(correlations[chosen]) < 1e-9:
            break
        selected.append(chosen)
        As = A[:, selected]
        coeffs, *_ = np.linalg.lstsq(As, b, rcond=None)
        residual = b - As @ coeffs
    if selected:
        x[selected] = coeffs
    return x


class ExperienceAutoencoder:
    """Compress episodes into Δz vectors and reconstruct learning signals."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.config.setdefault("use_hidden_delta", False)
        self.config.setdefault("hidden_delta_lr", HEBBIAN_LR)

        seed = int(self.config.get("seed", 42))
        self._vocab: dict[str, int] = {}
        self._A = _make_sensing_matrix(seed)
        self._encoder = ExperienceEncoder(self._vocab, self._A)
        self._decoder = ExperienceDecoder(self._vocab, self._A)

        # Lazy state for the optional hidden-state delta path.
        self._d_hidden: int | None = None
        self._A_hidden: np.ndarray | None = None
        self._hidden_delta_stats: dict[str, float] = {
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "count": 0.0,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Backfill attributes added after the initial release."""
        self.__dict__.update(state)
        if not isinstance(self.config, dict):
            self.config = {}
        self.config.setdefault("use_hidden_delta", False)
        self.config.setdefault("hidden_delta_lr", HEBBIAN_LR)
        if not hasattr(self, "_A_hidden"):
            self._A_hidden = None
        if not hasattr(self, "_d_hidden"):
            self._d_hidden = None
        if not hasattr(self, "_hidden_delta_stats"):
            self._hidden_delta_stats = {
                "mean_norm": 0.0,
                "std_norm": 0.0,
                "count": 0.0,
            }

    def encode(self, episode: dict[str, Any]) -> np.ndarray:
        """Encode one episode into Δz.

        If ``episode`` carries a ``hidden_delta`` array and hidden-delta mode is
        enabled, route to :meth:`encode_hidden_delta`; otherwise fall back to
        the legacy text-token path.
        """
        if self.config.get("use_hidden_delta") and "hidden_delta" in episode:
            return self.encode_hidden_delta(
                episode["hidden_delta"],
                outcome=str(episode.get("outcome", "unknown")),
            )
        return self._encoder.encode(episode)
    def encode_hidden_delta(
        self,
        delta_h: np.ndarray,
        outcome: str = "unknown",
    ) -> np.ndarray:
        """Encode a hidden-state delta into Δz."""
        delta_z, _ = self._encode_hidden_delta(delta_h, outcome)
        return delta_z
    def _encode_hidden_delta(
        self,
        delta_h: np.ndarray,
        outcome: str = "unknown",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a hidden-state delta; returns (Δz, normalized_delta)."""
        delta_h = np.asarray(delta_h, dtype=float).reshape(-1)
        d_hidden = delta_h.shape[0]

        if self._A_hidden is None:
            self._d_hidden = d_hidden
            seed = int(self.config.get("seed", 42))
            self._A_hidden = _make_hidden_sensing_matrix(seed + 1, d_hidden)

        if d_hidden != self._d_hidden:
            raise ValueError(
                f"hidden delta dimension {d_hidden} does not match "
                f"initialized dimension {self._d_hidden}"
            )

        norm = float(np.linalg.norm(delta_h))
        stats = self._hidden_delta_stats
        stats["count"] += 1.0
        n = stats["count"]
        old_mean = stats["mean_norm"]
        new_mean = old_mean + (norm - old_mean) / n
        stats["mean_norm"] = new_mean
        if n > 1.0:
            old_var = stats["std_norm"] ** 2
            new_var = (
                (n - 1.0) * old_var + (norm - old_mean) * (norm - new_mean)
            ) / n
            stats["std_norm"] = math.sqrt(max(0.0, new_var))

        if stats["std_norm"] > 0.0:
            normalized_delta = delta_h / (stats["std_norm"] + 1e-8)
        else:
            normalized_delta = delta_h / (stats["mean_norm"] + 1.0)
        normalized_delta = normalized_delta * HIDDEN_DELTA_SCALE

        residual = self._A_hidden @ normalized_delta
        residual = np.tanh(residual / (1.0 + np.linalg.norm(normalized_delta)))

        delta_z = np.empty(LATENT_DIM, dtype=float)
        delta_z[:OUTCOME_DIM] = _outcome_vector(outcome)
        delta_z[OUTCOME_DIM:] = residual
        return delta_z, normalized_delta

    def decode(self, delta_z: np.ndarray) -> dict[str, Any]:
        """Decode a Δz vector back into learning fields."""
        return self._decoder.decode(delta_z)

    def decode_hidden_delta(
        self,
        delta_z: np.ndarray,
    ) -> dict[str, Any]:
        """Decode a hidden-state Δz into outcome and reconstructed delta."""
        delta_z = np.asarray(delta_z, dtype=float).reshape(-1)
        outcome_vec = delta_z[:OUTCOME_DIM]
        outcome_idx = int(np.argmax(outcome_vec))
        outcome = _OUTCOME_LABELS[min(outcome_idx, len(_OUTCOME_LABELS) - 1)]
        failure_class = _extract_failure_class(outcome, delta_z)
        residual = delta_z[OUTCOME_DIM:]
        latent_drift_score = float(np.linalg.norm(residual))
        if self._A_hidden is not None:
            delta_estimated = self._A_hidden.T @ residual
        else:
            delta_estimated = np.zeros(self._d_hidden or 0, dtype=float)
        return {
            "outcome": outcome,
            "failure_class": failure_class,
            "latent_drift_score": latent_drift_score,
            "delta_estimated": delta_estimated,
        }

    def _decode_hidden_to_delta(self, delta_z: np.ndarray) -> np.ndarray:
        """Reconstruct the hidden delta from the residual tail of Δz."""
        residual = np.asarray(delta_z, dtype=float).reshape(-1)[OUTCOME_DIM:]
        if self._A_hidden is not None:
            return self._A_hidden.T @ residual
        return np.zeros(self._d_hidden or 0, dtype=float)

    def decode_latent(self, delta_z: np.ndarray) -> dict[str, Any]:
        """Dispatch to the appropriate decoder for this latent vector."""
        if self.config.get("use_hidden_delta") and self._A_hidden is not None:
            return self.decode_hidden_delta(delta_z)
        return self.decode(delta_z)

    def update_identity(self, current_z: np.ndarray | None, episode: dict[str, Any]) -> np.ndarray:
        """Accumulate a new episode delta into the running identity latent z."""
        delta_z = self.encode(episode)
        if current_z is None:
            return delta_z.copy()
        current_z = np.asarray(current_z, dtype=float).reshape(-1)
        new_z = current_z + delta_z
        return new_z

    def reconstruction_error(self, original: dict[str, Any], decoded: dict[str, Any]) -> float:
        """Scalar distance between an episode and its decoded reconstruction."""
        tokens = _episode_tokens(original)

        expected_failure = _expected_failure_class(original)
        failure_penalty = 0.0 if decoded["failure_class"] == expected_failure else 0.25

        trigger_tokens = set(decoded.get("trigger_conditions", []))
        trigger_overlap = _jaccard(tokens, trigger_tokens)
        trigger_penalty = 1.0 - trigger_overlap

        counter_tokens = set()
        for ex in decoded.get("counterexamples", []):
            counter_tokens.update(_tokenize(ex))
        counter_overlap = _jaccard(tokens, counter_tokens)
        counter_penalty = 1.0 - counter_overlap

        hint_tokens: set[str] = set()
        for k, v in decoded.get("corrected_behavior_hint", {}).items():
            hint_tokens.update(_tokenize(str(k)))
            hint_tokens.update(_tokenize(str(v)))
        hint_overlap = _jaccard(tokens, hint_tokens)
        hint_penalty = 1.0 - hint_overlap

        return float(
            (failure_penalty + trigger_penalty + counter_penalty + hint_penalty) / 4.0
        )

    def compress(self, episodes: list[dict[str, Any]]) -> list[np.ndarray]:
        """Encode a batch of episodes into Δz vectors."""
        return [self.encode(ep) for ep in episodes]

    def train_step(
        self,
        episode: dict[str, Any],
        lr: float = HEBBIAN_LR,
    ) -> float:
        """Apply one Hebbian-style passive update.

        Routes to the hidden-delta update when ``hidden_delta`` is present and
        hidden-delta mode is enabled; otherwise applies the legacy text-token
        update.
        """
        if self.config.get("use_hidden_delta") and "hidden_delta" in episode:
            hidden_lr = self.config.get("hidden_delta_lr", lr)
            return self.train_step_hidden_delta(
                episode["hidden_delta"],
                outcome=str(episode.get("outcome", "unknown")),
                lr=hidden_lr,
            )

        # Legacy text-token path.
        # Encode first so vocab is fully populated; the residual portion of Δz
        # is the reinforcing signal for this episode's input direction.
        delta_z = self.encode(episode)
        residual_target = delta_z[OUTCOME_DIM:]

        # Capture pre-update reconstruction error for caller convergence tracking.
        error = self.reconstruction_error(episode, self.decode(delta_z))

        # Re-extract the input feature vector via the encoder's internal builder.
        # encode() above already extended the vocab, so this returns the same f
        # without further mutation.
        feature_signal = self._encoder.extract_features(episode)

        # Rank-1 Hebbian update. outer(residual_target, feature_signal) has shape
        # (RESIDUAL_DIM, NUM_SOURCES * MAX_VOCAB) — same as the sensing matrix.
        # Columns whose feature_signal is zero are left untouched.
        self._A += lr * np.outer(residual_target, feature_signal)

        # Renormalize columns to unit L2-norm, mirroring `_make_sensing_matrix`.
        # Guard against divide-by-zero on never-touched columns.
        norms = np.linalg.norm(self._A, axis=0, keepdims=True)
        self._A /= np.where(norms == 0, 1.0, norms)
        return float(error)

    def train_step_hidden_delta(
        self,
        delta_h: np.ndarray,
        outcome: str = "unknown",
        lr: float | None = None,
    ) -> float:
        """Apply one Hebbian update to the hidden-delta sensing matrix.

        Returns the pre-update Euclidean reconstruction error.
        """
        if lr is None:
            lr = self.config.get("hidden_delta_lr", HEBBIAN_LR)
        delta_z, normalized_delta = self._encode_hidden_delta(delta_h, outcome)
        residual_target = delta_z[OUTCOME_DIM:]
        delta_h = np.asarray(delta_h, dtype=float).reshape(-1)
        error = float(np.linalg.norm(delta_h - self._decode_hidden_to_delta(delta_z)))
        self._A_hidden += lr * np.outer(residual_target, normalized_delta)
        norms = np.linalg.norm(self._A_hidden, axis=0, keepdims=True)
        self._A_hidden /= np.where(norms == 0, 1.0, norms)
        return float(error)

    def status(self, include_size: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "project": "experience_autoencoder",
            "ready": True,
            "latent_dim": LATENT_DIM,
            "vocab_size": len(self._vocab),
            "record_count": len(self._vocab),
            "use_hidden_delta": self.config.get("use_hidden_delta", False),
            "hidden_dim": self._d_hidden,
            "hidden_matrix_initialized": self._A_hidden is not None,
        }
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            )
        return result



def _episode_tokens(episode: dict[str, Any]) -> set[str]:
    toks: set[str] = set()
    normalized = _normalize_episode(episode)
    for key in _SOURCE_NAMES:
        toks.update(_tokenize(normalized.get(key, "")))
    return toks


def _expected_failure_class(episode: dict[str, Any]) -> str:
    outcome = str(episode.get("outcome", "unknown"))
    base = _FAILURE_MAP.get(outcome, "unknown")
    if outcome == "corrected":
        correction = str(episode.get("correction", "")).lower()
        if any(k in correction for k in ("wrong", "incorrect", "not right")):
            return "fact_correction"
    return base


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
