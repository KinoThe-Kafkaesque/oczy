"""Lane 06: Bounded-Growth Consolidation (A0b seed-regenerable variant).

A0b: ExperienceAutoencoder's dense ``_A`` is regenerated on load from
``numpy.random.RandomState(seed)`` and excluded from the pickled state;
only seed + shape + token vocab are persisted. IdentityHypernetwork at
default. Threshold <= 22947 bytes. Deterministic; never raises.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter
from typing import Any

import numpy as np

_LATENT_DIM, _OUTCOME_DIM, _NUM_SOURCES, _MAX_VOCAB = 32, 4, 4, 256
_RESIDUAL_DIM = _LATENT_DIM - _OUTCOME_DIM
_OUTCOME_LABELS = ["accepted", "corrected", "failed", "unknown"]
_OUTCOME_TO_IDX = {l: i for i, l in enumerate(_OUTCOME_LABELS)}
_SOURCE_NAMES = ["situation", "model_answer", "correction", "revised_answer"]
_SOURCE_WEIGHTS = np.array([1.0, 1.0, 2.0, 1.25], dtype=float)
_STOPWORDS = {"the","a","an","is","are","was","were","be","been","being","to",
    "of","and","or","in","on","at","for","with","as","this","that","it","its",
    "i","you","he","she","we","they","my","your","his","her","our","their"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= 2 and t not in _STOPWORDS]


def _outcome_vector(outcome: str) -> np.ndarray:
    vec = np.full(_OUTCOME_DIM, -0.2, dtype=float)
    vec[_OUTCOME_TO_IDX.get(outcome, _OUTCOME_TO_IDX["unknown"])] = 0.8
    return vec


class A0bAutoencoder:
    """Seed-regenerable autoencoder. Persists seed + shape + token vocab;
    the dense ``_A`` is regenerated from RandomState(seed) on load and
    excluded from the pickled state via ``__getstate__``."""
    def __init__(self, seed: int = 42,
                 n_input: int = _NUM_SOURCES * _MAX_VOCAB,
                 latent_dim: int = _LATENT_DIM) -> None:
        self.seed, self.n_input, self.latent_dim = int(seed), int(n_input), int(latent_dim)
        self.residual_dim = self.latent_dim - _OUTCOME_DIM
        self._vocab: dict[str, int] = {}
        self._A = self._regenerate_A()
    def _regenerate_A(self) -> np.ndarray:
        rs = np.random.RandomState(self.seed)
        A = rs.randn(self.residual_dim, self.n_input)
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        return (A / np.where(norms == 0, 1.0, norms)).astype(float)
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_A", None)  # regenerated from seed on load
        return state
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._A = self._regenerate_A()
    def _ensure_token(self, token: str) -> int | None:
        if token in self._vocab:
            return self._vocab[token]
        if len(self._vocab) < _MAX_VOCAB:
            idx = len(self._vocab)
            self._vocab[token] = idx
            return idx
        return None
    def _extract_features(self, episode: dict[str, Any]) -> np.ndarray:
        f = np.zeros(_NUM_SOURCES * _MAX_VOCAB, dtype=float)
        for s_i, source in enumerate(_SOURCE_NAMES):
            for tok, c in Counter(_tokenize(str(episode.get(source, "")))).items():
                idx = self._ensure_token(tok)
                if idx is not None:
                    f[s_i * _MAX_VOCAB + idx] += c * _SOURCE_WEIGHTS[s_i]
        return f
    def encode(self, episode: dict[str, Any]) -> np.ndarray:
        f = self._extract_features(episode)
        residual = np.tanh((self._A @ f) / (1.0 + np.linalg.norm(f)))
        dz = np.empty(self.latent_dim, dtype=float)
        dz[:_OUTCOME_DIM] = _outcome_vector(episode.get("outcome", "unknown"))
        dz[_OUTCOME_DIM:] = residual
        return dz
    def train_step(self, episode: dict[str, Any], lr: float = 0.01) -> float:
        return 0.0  # A0b persists no Hebbian deltas; _A regenerated from seed.
    def status(self, include_size: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"project": "experience_autoencoder_a0b",
            "ready": True, "latent_dim": self.latent_dim,
            "vocab_size": len(self._vocab), "seed": self.seed,
            "shape": (self.residual_dim, self.n_input)}
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        return result


def name() -> str:
    return "lane_06_combined_footprint_bytes"


def measure() -> float:
    try:
        from identity_hypernetwork import IdentityHypernetwork
        ae = A0bAutoencoder(seed=42)
        hn = IdentityHypernetwork()  # default latent_dim=8, seed=0
        episodes = [
            ("what color is the sky on a clear day", "green", "no the sky is blue not green", "blue"),
            ("capital of france", "lyon", "incorrect the capital is paris", "paris"),
            ("two plus two", "five", "wrong two plus two is four", "four"),
            ("largest planet", "mars", "no jupiter is the largest planet", "jupiter"),
        ]
        for situ, wrong, corr, right in episodes:
            ep = {"situation": situ, "model_answer": wrong, "correction": corr,
                  "revised_answer": right, "outcome": "corrected"}
            ae.encode(ep)
            ae.train_step(ep)
            hn.update_identity({"source": "user_correction",
                                "correct_label": right, "token": right})
        ae_bytes = int(ae.status(include_size=True).get("serialized_bytes", 0))
        hn_bytes = int(hn.status(include_size=True).get("serialized_bytes", 0))
        return float(ae_bytes + hn_bytes)
    except Exception:
        return float("nan")