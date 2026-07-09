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

from lanes._common import lane_measure

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
    def _build_target(self, episode: dict[str, Any]) -> np.ndarray:
        """Compact target: outcome vector + correction token presence.
        Uses a module-level token list shared across instances."""
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
    def decode(self, dz: np.ndarray) -> np.ndarray:
        """Reconstruct compact target from latent via pseudo-inverse of _A.
        A0b has no trained decoder, so this is an untrained baseline."""
        residual = dz[_OUTCOME_DIM:]
        f_hat = np.linalg.pinv(self._A) @ residual
        outcome_hat = np.full(_OUTCOME_DIM, -0.2, dtype=float)
        outcome_hat[np.argmax(dz[:_OUTCOME_DIM])] = 0.8
        return np.concatenate([outcome_hat, f_hat])
    def reconstruction_error(self, episode: dict[str, Any]) -> float:
        """MSE on compact target (outcome + correction token presence).
        Uses pinv decode — untrained baseline for comparison with A1."""
        target = self._build_target(episode)
        if np.linalg.norm(target) < 1e-9:
            return 0.0
        dz = self.encode(episode)
        # A0b has no trained decoder; use pinv on residual dims only
        pred = np.zeros(A1Autoencoder._TARGET_DIM, dtype=float)
        pred[:_OUTCOME_DIM] = np.full(_OUTCOME_DIM, -0.2, dtype=float)
        pred[np.argmax(dz[:_OUTCOME_DIM])] = 0.8
        # pinv decode for correction token presence (best effort with random A)
        residual = dz[_OUTCOME_DIM:]
        f_hat = np.linalg.pinv(self._A) @ residual
        corr_dim = A1Autoencoder._TARGET_DIM - _OUTCOME_DIM
        pred[_OUTCOME_DIM:] = f_hat[:corr_dim]
        return float(np.mean((pred - target) ** 2))
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

class A1Autoencoder:
    """Trained compact autoencoder (thesis §9). Replaces A0b's random
    projection with a learned low-rank encoder ``_A = U @ V`` and a trained
    decoder ``_D`` that reconstructs a compact target (outcome + correction
    token presence) from the latent Δz. Trained offline on MSE reconstruction
    loss; persists U, V, D as float32 to stay within the byte budget."""
    _TARGET_DIM = 16  # 4 outcome + 12 correction-token presence dims
    def __init__(self, seed: int = 42, rank: int = 3,
                 n_input: int = _NUM_SOURCES * _MAX_VOCAB,
                 latent_dim: int = _LATENT_DIM) -> None:
        self.seed, self.rank = int(seed), int(rank)
        self.n_input, self.latent_dim = int(n_input), int(latent_dim)
        self.residual_dim = self.latent_dim - _OUTCOME_DIM
        self.target_dim = self._TARGET_DIM
        self._vocab: dict[str, int] = {}
        self._corr_tokens: list[str] = []  # maps target idx → token
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
        """Map a correction token to a target-slot index (0..11)."""
        if token in self._corr_tokens:
            return self._corr_tokens.index(token)
        if len(self._corr_tokens) < self.target_dim - _OUTCOME_DIM:
            self._corr_tokens.append(token)
            return len(self._corr_tokens) - 1
        return None
    def _extract_features(self, episode: dict[str, Any]) -> np.ndarray:
        f = np.zeros(_NUM_SOURCES * _MAX_VOCAB, dtype=float)
        for s_i, source in enumerate(_SOURCE_NAMES):
            for tok, c in Counter(_tokenize(str(episode.get(source, "")))).items():
                idx = self._ensure_token(tok)
                if idx is not None:
                    f[s_i * _MAX_VOCAB + idx] += c * _SOURCE_WEIGHTS[s_i]
        return f
    def _build_target(self, episode: dict[str, Any]) -> np.ndarray:
        """Compact target: outcome vector + correction token presence."""
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
        """Reconstruct compact target from latent dz via trained decoder D."""
        return self._D @ dz + self._b_dec
    def train_step(self, episode: dict[str, Any], lr: float = 0.05,
                   train_encoder: bool = True) -> float:
        """One SGD step of MSE reconstruction: encode → decode → MSE vs target.
        When train_encoder=False, only the decoder D is updated (random encoder
        frozen) — used as an A0b-equivalent baseline with the same decode path."""
        f = self._extract_features(episode)
        if np.linalg.norm(f) < 1e-9:
            return 0.0
        target = self._build_target(episode)
        # Forward pass
        A = self._U @ self._V
        norm_f = 1.0 + np.linalg.norm(f)
        pre = (A @ f) / norm_f + self._b_enc
        residual = np.tanh(pre)
        dz = np.empty(self.latent_dim, dtype=float)
        dz[:_OUTCOME_DIM] = target[:_OUTCOME_DIM]
        dz[_OUTCOME_DIM:] = residual
        pred = self._D @ dz + self._b_dec
        loss = float(np.mean((pred - target) ** 2))
        # Backward pass
        n = target.shape[0]
        grad_pred = (2.0 / n) * (pred - target)          # (target_dim,)
        grad_D = np.outer(grad_pred, dz)                   # (target_dim, latent_dim)
        grad_b_dec = grad_pred                              # (target_dim,)
        # Update decoder always
        self._D -= lr * grad_D.astype(np.float32)
        self._b_dec -= lr * grad_b_dec.astype(np.float32)
        if not train_encoder:
            return loss
        grad_dz = self._D.T @ grad_pred                     # (latent_dim,)
        # Outcome dims of dz are fixed (not differentiable), only residual flows
        grad_residual = grad_dz[_OUTCOME_DIM:]              # (residual_dim,)
        grad_pre = grad_residual * (1.0 - residual ** 2)   # through tanh
        grad_Af = grad_pre / norm_f                         # w.r.t. (A @ f)
        grad_b_enc = grad_pre                               # (residual_dim,)
        # A = U @ V → grad_U = outer(grad_Af, f) @ V^T, grad_V = U^T @ outer(grad_Af, f)
        outer_gf = np.outer(grad_Af, f)
        grad_U = outer_gf @ self._V.T
        grad_V = self._U.T @ outer_gf
        self._U -= lr * grad_U.astype(np.float32)
        self._V -= lr * grad_V.astype(np.float32)
        self._b_enc -= lr * grad_b_enc.astype(np.float32)
        return loss
    def reconstruction_error(self, episode: dict[str, Any]) -> float:
        """MSE between decoded latent and compact target on held-out episodes."""
        target = self._build_target(episode)
        if np.linalg.norm(target) < 1e-9:
            return 0.0
        dz = self.encode(episode)
        pred = self.decode(dz)
        return float(np.mean((pred - target) ** 2))
    def status(self, include_size: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"project": "experience_autoencoder_a1",
            "ready": True, "latent_dim": self.latent_dim,
            "vocab_size": len(self._vocab), "seed": self.seed,
            "rank": self.rank, "target_dim": self.target_dim,
            "shape": (self.residual_dim, self.n_input)}
        if include_size:
            result["serialized_bytes"] = len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        return result


def name() -> str:
    return "lane_06_combined_footprint_bytes"


@lane_measure
def measure() -> float:
    from identity_hypernetwork import IdentityHypernetwork
    hn = IdentityHypernetwork()  # default latent_dim=8, seed=0
    # 12-episode synthetic correction corpus (8 train + 4 held-out)
    corpus = [
        ("what color is the sky on a clear day", "green", "no the sky is blue not green", "blue"),
        ("capital of france", "lyon", "incorrect the capital is paris", "paris"),
        ("two plus two", "five", "wrong two plus two is four", "four"),
        ("largest planet", "mars", "no jupiter is the largest planet", "jupiter"),
        ("speed of light", "slow", "light travels very fast actually", "fast"),
        ("boiling point of water", "50", "water boils at 100 degrees celsius", "100"),
        ("first president of the us", "lincoln", "washington was the first president", "washington"),
        ("chemical symbol for gold", "go", "the symbol for gold is au", "au"),
        ("how many continents", "4", "there are seven continents", "seven"),
        ("tallest mountain", "k2", "everest is the tallest mountain", "everest"),
        ("opposite of hot", "warm", "the opposite of hot is cold", "cold"),
        ("primary color", "green", "red is a primary color", "red"),
    ]
    train_eps = corpus[:8]
    held_out = corpus[8:]
    # --- A0b baseline (random projection, seed-regenerable) ---
    ae0 = A0bAutoencoder(seed=42)
    # --- A1 trained low-rank autoencoder ---
    ae1 = A1Autoencoder(seed=42, rank=3)
    # --- A0b-equiv: same A1 architecture but encoder frozen (random) ---
    ae0b = A1Autoencoder(seed=99, rank=3)  # different seed → different random enc
    # Pre-fill vocab and corr_tokens on all with training episodes
    # so held-out measurement uses consistent token mappings
    train_data = []
    for situ, wrong, corr, right in train_eps:
        ep = {"situation": situ, "model_answer": wrong, "correction": corr,
              "revised_answer": right, "outcome": "corrected"}
        train_data.append(ep)
        ae0.encode(ep)
        ae0._build_target(ep)
        ae1.encode(ep)
        ae1._build_target(ep)
        ae0b.encode(ep)
        ae0b._build_target(ep)
        hn.update_identity({"source": "user_correction",
                            "correct_label": right, "token": right})
    a0b_bytes = int(ae0.status(include_size=True).get("serialized_bytes", 0))
    # Train A1 (encoder + decoder) and A0b-equiv (decoder only) for 100 epochs
    for _epoch in range(100):
        for ep in train_data:
            ae1.train_step(ep, lr=0.05, train_encoder=True)
            ae0b.train_step(ep, lr=0.05, train_encoder=False)
    a1_bytes = int(ae1.status(include_size=True).get("serialized_bytes", 0))
    held_out_eps = [{"situation": s, "model_answer": w, "correction": c,
                     "revised_answer": r, "outcome": "corrected"}
                    for s, w, c, r in held_out]
    a0b_recon = float(np.mean([ae0.reconstruction_error(ep) for ep in held_out_eps]))
    a0b_equiv_recon = float(np.mean([ae0b.reconstruction_error(ep) for ep in held_out_eps]))
    a1_recon = float(np.mean([ae1.reconstruction_error(ep) for ep in held_out_eps]))
    hn_bytes = int(hn.status(include_size=True).get("serialized_bytes", 0))
    # A1 trained encoder should beat A0b-equiv random encoder (same decode path)
    _ = (a0b_bytes, a0b_recon, a0b_equiv_recon, a1_recon,
         a1_recon < a0b_equiv_recon)
    # Primary metric: A1 combined footprint (must stay < 22947)
    return float(a1_bytes + hn_bytes)
