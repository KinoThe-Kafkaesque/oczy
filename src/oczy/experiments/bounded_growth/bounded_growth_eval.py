"""Experiment 06: Bounded-Growth Consolidation.

Measures the combined memory footprint (experience autoencoder + identity
hypernetwork) of progressively more compact organism configurations against
the baseline (A0).  The primary metric is::

    METRIC bounded_growth_m1_ratio=<float>

defined as ``min(A1, A2, A3 combined) / A0 combined``.  Lower is better;
``<= 0.10`` is the acceptance threshold.

Conditions
----------
A0     : default ``OrganismAgent`` — full ``ExperienceAutoencoder`` (dense
         sensing matrix persisted) + default ``IdentityHypernetwork``.
A0b    : seed-regenerable autoencoder (dense ``_A`` excluded from pickle,
         regenerated from ``RandomState(seed)`` on load) + default HN.
A1     : trained low-rank autoencoder (``_A = U @ V``, float32) + default HN.
A2     : A1 autoencoder + ``ConceptEmbeddingHypernetwork`` (compact concept-
         embedding variant with reduced latent dim).
A3     : A1 autoencoder + ultra-compact ``ConceptEmbeddingHypernetwork``
         (minimal latent dim, fixed tiny vocab).
REF-lo : reference low-footprint configuration (A0b autoencoder + minimal HN).

The experiment is driver-free (pure NumPy, no LM loaded).  It runs
``EvalSuite.run(agent)`` over a 1-level curriculum subset so each condition
gets a byte measurement after consolidation.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from identity_hypernetwork import IdentityHypernetwork
from oczy.experiments.curriculum import Curriculum, CurriculumLevel, build_curriculum
from oczy.experiments.eval_suite import EvalSuite
from oczy.experiments.organism import OrganismAgent

# ---------------------------------------------------------------------------
# Shared constants (mirrors lanes/lane_06.py and experience_autoencoder)
# ---------------------------------------------------------------------------

_LATENT_DIM, _OUTCOME_DIM, _NUM_SOURCES, _MAX_VOCAB = 32, 4, 4, 256
_RESIDUAL_DIM = _LATENT_DIM - _OUTCOME_DIM
_OUTCOME_LABELS = ["accepted", "corrected", "failed", "unknown"]
_OUTCOME_TO_IDX = {l: i for i, l in enumerate(_OUTCOME_LABELS)}
_SOURCE_NAMES = ["situation", "model_answer", "correction", "revised_answer"]
_SOURCE_WEIGHTS = np.array([1.0, 1.0, 2.0, 1.25], dtype=float)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "at", "for", "with", "as",
    "this", "that", "it", "its", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Canonical episode aliases — curriculum episodes use ``query``/``answer``/
# ``corrected_answer`` while the autoencoders expect ``situation``/
# ``model_answer``/``revised_answer``.
_CANONICAL_ALIASES: dict[str, str] = {
    "query": "situation",
    "answer": "model_answer",
    "corrected_answer": "revised_answer",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) >= 2 and t not in _STOPWORDS]


def _outcome_vector(outcome: str) -> np.ndarray:
    vec = np.full(_OUTCOME_DIM, -0.2, dtype=float)
    vec[_OUTCOME_TO_IDX.get(outcome, _OUTCOME_TO_IDX["unknown"])] = 0.8
    return vec


def _normalize_episode(episode: dict[str, Any]) -> dict[str, str]:
    """Fold canonical aliases into legacy field names."""
    normalized: dict[str, str] = {k: str(v) for k, v in episode.items()}
    for canonical, legacy in _CANONICAL_ALIASES.items():
        if not normalized.get(legacy) and normalized.get(canonical):
            normalized[legacy] = normalized[canonical]
    return normalized


# ---------------------------------------------------------------------------
# A0b: seed-regenerable autoencoder (adapted from lanes/lane_06.py)
# ---------------------------------------------------------------------------

class A0bAutoencoder:
    """Seed-regenerable autoencoder.

    Persists seed + shape + token vocab; the dense ``_A`` is regenerated
    from ``RandomState(seed)`` on load and excluded from the pickled state
    via ``__getstate__``.
    """

    def __init__(self, seed: int = 42,
                 n_input: int = _NUM_SOURCES * _MAX_VOCAB,
                 latent_dim: int = _LATENT_DIM) -> None:
        self.seed = int(seed)
        self.n_input = int(n_input)
        self.latent_dim = int(latent_dim)
        self.residual_dim = self.latent_dim - _OUTCOME_DIM
        self._vocab: dict[str, int] = {}
        self._corr_tokens: list[str] = []
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
        normalized = _normalize_episode(episode)
        f = np.zeros(_NUM_SOURCES * _MAX_VOCAB, dtype=float)
        for s_i, source in enumerate(_SOURCE_NAMES):
            for tok, c in Counter(_tokenize(str(normalized.get(source, "")))).items():
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

    def decode(self, dz: np.ndarray) -> np.ndarray:
        residual = dz[_OUTCOME_DIM:]
        f_hat = np.linalg.pinv(self._A) @ residual
        outcome_hat = np.full(_OUTCOME_DIM, -0.2, dtype=float)
        outcome_hat[np.argmax(dz[:_OUTCOME_DIM])] = 0.8
        return np.concatenate([outcome_hat, f_hat])

    def reconstruction_error(self, episode: dict[str, Any]) -> float:
        target = self._build_target(episode)
        if np.linalg.norm(target) < 1e-9:
            return 0.0
        dz = self.encode(episode)
        pred = np.zeros(A1Autoencoder._TARGET_DIM, dtype=float)
        pred[:_OUTCOME_DIM] = np.full(_OUTCOME_DIM, -0.2, dtype=float)
        pred[np.argmax(dz[:_OUTCOME_DIM])] = 0.8
        residual = dz[_OUTCOME_DIM:]
        f_hat = np.linalg.pinv(self._A) @ residual
        corr_dim = A1Autoencoder._TARGET_DIM - _OUTCOME_DIM
        pred[_OUTCOME_DIM:] = f_hat[:corr_dim]
        return float(np.mean((pred - target) ** 2))

    def _build_target(self, episode: dict[str, Any]) -> np.ndarray:
        target = np.zeros(A1Autoencoder._TARGET_DIM, dtype=float)
        target[:_OUTCOME_DIM] = _outcome_vector(episode.get("outcome", "unknown"))
        for tok in _tokenize(str(episode.get("correction", ""))):
            if tok in self._corr_tokens:
                ci = self._corr_tokens.index(tok)
            elif len(self._corr_tokens) < A1Autoencoder._TARGET_DIM - _OUTCOME_DIM:
                self._corr_tokens.append(tok)
                ci = len(self._corr_tokens) - 1
            else:
                continue
            target[_OUTCOME_DIM + ci] = 1.0
        return target

    def train_step(self, episode: dict[str, Any], lr: float = 0.01) -> float:
        return 0.0  # A0b persists no Hebbian deltas; _A regenerated from seed.

    def status(self, include_size: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "project": "experience_autoencoder_a0b",
            "ready": True,
            "latent_dim": self.latent_dim,
            "vocab_size": len(self._vocab),
            "seed": self.seed,
            "shape": (self.residual_dim, self.n_input),
        }
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        return result


# ---------------------------------------------------------------------------
# A1: trained compact autoencoder (adapted from lanes/lane_06.py)
# ---------------------------------------------------------------------------

class A1Autoencoder:
    """Trained compact autoencoder.

    Replaces A0b's random projection with a learned low-rank encoder
    ``_A = U @ V`` and a trained decoder ``_D`` that reconstructs a compact
    target (outcome + correction token presence) from the latent Δz.
    Trained offline on MSE reconstruction loss; persists U, V, D as
    float32 to stay within the byte budget.
    """

    _TARGET_DIM = 16  # 4 outcome + 12 correction-token presence dims

    def __init__(self, seed: int = 42, rank: int = 3,
                 n_input: int = _NUM_SOURCES * _MAX_VOCAB,
                 latent_dim: int = _LATENT_DIM) -> None:
        self.seed = int(seed)
        self.rank = int(rank)
        self.n_input = int(n_input)
        self.latent_dim = int(latent_dim)
        self.residual_dim = self.latent_dim - _OUTCOME_DIM
        self.target_dim = self._TARGET_DIM
        self._vocab: dict[str, int] = {}
        self._corr_tokens: list[str] = []
        rs = np.random.RandomState(self.seed)
        self._U = (rs.randn(self.residual_dim, self.rank) * 0.1).astype(np.float32)
        self._V = (rs.randn(self.rank, self.n_input) * 0.1).astype(np.float32)
        self._b_enc = np.zeros(self.residual_dim, dtype=np.float32)
        self._D = (rs.randn(self.target_dim, self.latent_dim) * 0.1).astype(np.float32)
        self._b_dec = np.zeros(self.target_dim, dtype=np.float32)

    def _A(self) -> np.ndarray:
        return self._U @ self._V

    def __getstate__(self) -> dict[str, Any]:
        return {k: self.__dict__[k] for k in
                ("seed", "rank", "n_input", "latent_dim", "residual_dim",
                 "target_dim", "_vocab", "_corr_tokens",
                 "_U", "_V", "_b_enc", "_D", "_b_dec")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _ensure_token(self, token: str) -> int | None:
        if token in self._vocab:
            return self._vocab[token]
        if len(self._vocab) < _MAX_VOCAB:
            idx = len(self._vocab)
            self._vocab[token] = idx
            return idx
        return None

    def _ensure_corr_token(self, token: str) -> int | None:
        if token in self._corr_tokens:
            return self._corr_tokens.index(token)
        if len(self._corr_tokens) < self.target_dim - _OUTCOME_DIM:
            self._corr_tokens.append(token)
            return len(self._corr_tokens) - 1
        return None

    def _extract_features(self, episode: dict[str, Any]) -> np.ndarray:
        normalized = _normalize_episode(episode)
        f = np.zeros(_NUM_SOURCES * _MAX_VOCAB, dtype=float)
        for s_i, source in enumerate(_SOURCE_NAMES):
            for tok, c in Counter(_tokenize(str(normalized.get(source, "")))).items():
                idx = self._ensure_token(tok)
                if idx is not None:
                    f[s_i * _MAX_VOCAB + idx] += c * _SOURCE_WEIGHTS[s_i]
        return f

    def _build_target(self, episode: dict[str, Any]) -> np.ndarray:
        target = np.zeros(self.target_dim, dtype=float)
        target[:_OUTCOME_DIM] = _outcome_vector(episode.get("outcome", "unknown"))
        for tok in _tokenize(str(episode.get("correction", ""))):
            ci = self._ensure_corr_token(tok)
            if ci is not None:
                target[_OUTCOME_DIM + ci] = 1.0
        return target

    def encode(self, episode: dict[str, Any]) -> np.ndarray:
        f = self._extract_features(episode)
        residual = np.tanh((self._A() @ f) / (1.0 + np.linalg.norm(f)) + self._b_enc)
        dz = np.empty(self.latent_dim, dtype=float)
        dz[:_OUTCOME_DIM] = _outcome_vector(episode.get("outcome", "unknown"))
        dz[_OUTCOME_DIM:] = residual
        return dz

    def decode(self, dz: np.ndarray) -> np.ndarray:
        return self._D @ dz + self._b_dec

    def train_step(self, episode: dict[str, Any], lr: float = 0.05,
                   train_encoder: bool = True) -> float:
        f = self._extract_features(episode)
        if np.linalg.norm(f) < 1e-9:
            return 0.0
        target = self._build_target(episode)
        A = self._U @ self._V
        norm_f = 1.0 + np.linalg.norm(f)
        pre = (A @ f) / norm_f + self._b_enc
        residual = np.tanh(pre)
        dz = np.empty(self.latent_dim, dtype=float)
        dz[:_OUTCOME_DIM] = target[:_OUTCOME_DIM]
        dz[_OUTCOME_DIM:] = residual
        pred = self._D @ dz + self._b_dec
        loss = float(np.mean((pred - target) ** 2))
        n = target.shape[0]
        grad_pred = (2.0 / n) * (pred - target)
        grad_D = np.outer(grad_pred, dz)
        grad_b_dec = grad_pred
        self._D -= lr * grad_D.astype(np.float32)
        self._b_dec -= lr * grad_b_dec.astype(np.float32)
        if not train_encoder:
            return loss
        grad_dz = self._D.T @ grad_pred
        grad_residual = grad_dz[_OUTCOME_DIM:]
        grad_pre = grad_residual * (1.0 - residual ** 2)
        grad_Af = grad_pre / norm_f
        grad_b_enc = grad_pre
        outer_gf = np.outer(grad_Af, f)
        grad_U = outer_gf @ self._V.T
        grad_V = self._U.T @ outer_gf
        self._U -= lr * grad_U.astype(np.float32)
        self._V -= lr * grad_V.astype(np.float32)
        self._b_enc -= lr * grad_b_enc.astype(np.float32)
        return loss

    def reconstruction_error(self, episode: dict[str, Any]) -> float:
        target = self._build_target(episode)
        if np.linalg.norm(target) < 1e-9:
            return 0.0
        dz = self.encode(episode)
        pred = self.decode(dz)
        return float(np.mean((pred - target) ** 2))

    def status(self, include_size: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "project": "experience_autoencoder_a1",
            "ready": True,
            "latent_dim": self.latent_dim,
            "vocab_size": len(self._vocab),
            "seed": self.seed,
            "rank": self.rank,
            "target_dim": self.target_dim,
            "shape": (self.residual_dim, self.n_input),
        }
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        return result


# ---------------------------------------------------------------------------
# ConceptEmbeddingHypernetwork: compact concept-embedding HN variant
# ---------------------------------------------------------------------------

class ConceptEmbeddingHypernetwork(IdentityHypernetwork):
    """Compact concept-embedding variant of :class:`IdentityHypernetwork`.

    Wraps the base hypernetwork but with a reduced ``latent_dim`` and a
    fixed small concept vocabulary so the pickled footprint is smaller.
    The adapter-generation and identity-update interfaces are identical
    to the parent, so ``OrganismAgent`` can use it as a drop-in replacement.
    """

    _COMPACT_VOCAB: list[str] = [
        "profile", "domain", "formal", "error", "correct",
        "project", "business", "mistake",
    ]

    def __init__(self, latent_dim: int = 4, seed: int = 0,
                 learning_rate: float = 0.1,
                 vocab: list[str] | None = None) -> None:
        super().__init__(
            latent_dim=latent_dim,
            seed=seed,
            learning_rate=learning_rate,
        )
        # Override the concept vocabulary with a compact fixed set.
        self.concepts = list(vocab or self._COMPACT_VOCAB)
        self.concept_index = {concept: i for i, concept in enumerate(self.concepts)}
        self.output_dim = len(self.concepts)
        # Rebuild W with the new output_dim.
        scale = 1.0 / np.sqrt(self.input_dim)
        self.W = self.rng.standard_normal((self.output_dim, self.input_dim)) * scale
        self._concept_ages = [0] * self.output_dim


# ---------------------------------------------------------------------------
# Curriculum subset helper
# ---------------------------------------------------------------------------

class _SubsetCurriculum:
    """Thin wrapper exposing only the first ``n`` levels of a curriculum.

    ``EvalSuite`` only calls ``.levels()``, ``__len__``, and ``.seed``, so
    this duck-typed wrapper is sufficient.
    """

    def __init__(self, base: Curriculum, n_levels: int = 1) -> None:
        self._base = base
        self._levels = base.levels()[:n_levels]

    @property
    def seed(self) -> int:
        return self._base.seed

    def levels(self) -> tuple[CurriculumLevel, ...]:
        return self._levels

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self):
        return iter(self._levels)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def _build_agent(condition: str) -> OrganismAgent:
    """Construct an :class:`OrganismAgent` configured for *condition*.

    The ``OrganismAgent`` constructor is never modified; we overwrite
    ``experience_autoencoder`` and ``identity_hypernetwork`` after
    construction as instructed.
    """
    agent = OrganismAgent()

    if condition == "A0":
        pass  # default ExperienceAutoencoder + IdentityHypernetwork

    elif condition == "A0b":
        agent.experience_autoencoder = A0bAutoencoder(seed=42)

    elif condition == "A1":
        ae = A1Autoencoder(seed=42, rank=3)
        agent.experience_autoencoder = ae

    elif condition == "A2":
        agent.experience_autoencoder = A1Autoencoder(seed=42, rank=3)
        agent.identity_hypernetwork = ConceptEmbeddingHypernetwork(
            latent_dim=4, seed=0)

    elif condition == "A3":
        agent.experience_autoencoder = A1Autoencoder(seed=42, rank=3)
        agent.identity_hypernetwork = ConceptEmbeddingHypernetwork(
            latent_dim=2, seed=0,
            vocab=["profile", "domain", "error", "correct"])

    elif condition == "REF-lo":
        agent.experience_autoencoder = A0bAutoencoder(seed=42)
        agent.identity_hypernetwork = ConceptEmbeddingHypernetwork(
            latent_dim=2, seed=0,
            vocab=["profile", "domain", "error", "correct"])

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    return agent


# ---------------------------------------------------------------------------
# Footprint measurement
# ---------------------------------------------------------------------------

def _module_bytes(module: Any) -> int:
    """Best-effort pickle size of a module."""
    try:
        status = module.status(include_size=True)
        if isinstance(status, dict) and "serialized_bytes" in status:
            return int(status["serialized_bytes"])
    except Exception:
        pass
    try:
        return len(pickle.dumps(module, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


def _combined_footprint(agent: OrganismAgent) -> int:
    """Return ``ae_bytes + hn_bytes`` for the agent."""
    ae_bytes = _module_bytes(agent.experience_autoencoder)
    hn_bytes = _module_bytes(agent.identity_hypernetwork)
    return ae_bytes + hn_bytes


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------

CONDITIONS: tuple[str, ...] = ("A0", "A0b", "A1", "A2", "A3", "REF-lo")


def run(seed: int = 0, n_levels: int = 1,
        report_path: str | None = None) -> dict[str, Any]:
    """Run the bounded-growth consolidation experiment.

    Parameters
    ----------
    seed : curriculum seed (default 0).
    n_levels : number of curriculum levels to run (default 1 for speed).
    report_path : optional path to write a JSON report.

    Returns
    -------
    dict with per-condition footprints, behavior deltas, and ``m1_ratio``.
    """
    curriculum = _SubsetCurriculum(build_curriculum(seed=seed), n_levels=n_levels)
    suite = EvalSuite(curriculum)

    per_condition: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        agent = _build_agent(condition)
        result = suite.run(agent)
        combined = _combined_footprint(agent)
        ae_bytes = _module_bytes(agent.experience_autoencoder)
        hn_bytes = _module_bytes(agent.identity_hypernetwork)
        bpd = float(result.final_card.get("memory_bytes_per_behavior_delta", 0.0))
        per_condition[condition] = {
            "combined_footprint": combined,
            "ae_bytes": ae_bytes,
            "hn_bytes": hn_bytes,
            "consolidated_size": int(result.consolidated_size),
            "memory_bytes_per_behavior_delta": bpd,
            "transfer_score": float(result.final_card.get("transfer_score", 0.0)),
            "forgetting_score": float(result.final_card.get("forgetting_score", 0.0)),
            "consolidation_score": float(result.final_card.get("consolidation_score", 0.0)),
        }

    a0_combined = per_condition["A0"]["combined_footprint"]
    compact_candidates = [
        per_condition[c]["combined_footprint"]
        for c in ("A1", "A2", "A3")
        if c in per_condition
    ]
    if a0_combined > 0 and compact_candidates:
        m1_ratio = float(min(compact_candidates) / a0_combined)
    else:
        m1_ratio = float("nan")

    # Monotonicity check: A0b <= A0 (seed-regenerable excludes dense _A).
    a0 = per_condition["A0"]["combined_footprint"]
    a0b = per_condition["A0b"]["combined_footprint"]
    a1 = per_condition["A1"]["combined_footprint"]
    monotonic_a0_a0b = a0b <= a0
    monotonic_a0b_a1 = a1 <= a0  # A1 low-rank should be <= A0 dense

    report: dict[str, Any] = {
        "m1_ratio": m1_ratio,
        "a0_combined": a0_combined,
        "per_condition": per_condition,
        "monotonic_a0_a0b": monotonic_a0_a0b,
        "monotonic_a0b_a1_or_a0": monotonic_a0b_a1,
        "seed": seed,
        "n_levels": n_levels,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # Emit METRIC / ASI lines for autoresearch.sh.
    _emit_lines(report)

    if report_path:
        with open(report_path, "w") as fh:
            json.dump(report, fh, indent=2, default=str)

    return report


def _emit_lines(report: dict[str, Any]) -> None:
    """Print ``METRIC`` and ``ASI`` lines for autoresearch.sh consumption."""
    m1 = report["m1_ratio"]
    if isinstance(m1, float) and math.isnan(m1):
        m1_str = "nan"
    else:
        m1_str = f"{m1:.6f}"
    print(f"METRIC bounded_growth_m1_ratio={m1_str}")

    for cond, data in report["per_condition"].items():
        print(f"ASI {cond}_combined_footprint={data['combined_footprint']}")
        print(f"ASI {cond}_ae_bytes={data['ae_bytes']}")
        print(f"ASI {cond}_hn_bytes={data['hn_bytes']}")
        print(f"ASI {cond}_bytes_per_delta={data['memory_bytes_per_behavior_delta']:.6f}")

    print(f"ASI monotonic_a0_a0b={report['monotonic_a0_a0b']}")
    print(f"ASI monotonic_a0b_a1_or_a0={report['monotonic_a0b_a1_or_a0']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oczy.experiments.bounded_growth.bounded_growth_eval",
        description="Experiment 06: Bounded-Growth Consolidation",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Curriculum seed (default: 0)")
    parser.add_argument("--levels", type=int, default=1,
                        help="Number of curriculum levels (default: 1)")
    parser.add_argument("--report", type=str, default=None,
                        help="Optional path to write a JSON report")
    args = parser.parse_args(argv)

    run(seed=args.seed, n_levels=args.levels, report_path=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
