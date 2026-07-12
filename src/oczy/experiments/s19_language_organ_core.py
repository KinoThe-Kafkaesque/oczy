"""Research 19 core: shared cortex, two articulation arms, controls, scoring.

Implements the immutable Research/19 spec (research/19-lm-as-language-organ.md):

* A shared trainable cortex (<=64k persistent parameters) with a perception
  projection, a label head (Arm A), and a fixed-width latent coupler (Arm B).
* Direct online correction training on a frozen Qwen2.5-0.5B-Instruct LM.
* Arm A supplies the predicted label as a text prefix (parametric retrieval).
* Arm B injects a fixed 3x896 latent bank via ``inputs_embeds`` — no label,
  answer, correction, exemplar, or episode-ID text enters at probe time.
* C0-C7 conditions, model hashing, raw-trace lifecycle, parameter accounting.
* Calibrate-dev (DEV only) and evaluate (holdout/transfer/scope/specificity)
  are separated; evaluate fails closed unless manifest hash and human sign-off
  match.

This module is imported by :mod:`oczy.experiments.s19_language_organ` (CLI).
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from oczy.eval_v2.scoring import probe_matches
from oczy.experiments.organism_curriculum.dataset import (
    Episode,
    Probe,
    Stage,
    split_probes,
)
from oczy.lm._types import ReservedPosition
from oczy.lm.hf_driver import HFDriver

# ---------------------------------------------------------------------------
# Immutable configuration (frozen here per spec)
# ---------------------------------------------------------------------------

D_EMBD: int = 896           # Qwen2.5-0.5B-Instruct hidden size
D_CORTEX: int = 16          # cortex activation dimension
LATENT_TOKENS: int = 3      # fixed-width latent bank tokens
N_LABELS: int = 20          # stage-0 corrected labels
MAX_PARAMS: int = 64_000    # persistent parameter budget
LABEL_PREFIX_TEMPLATE: str = "{label}. "
SPECIFICITY_MARGIN_DEFAULT: float = 0.15
CONFIDENCE_THRESHOLD_DEFAULT: float = 0.5
MAX_GENERATE_TOKENS: int = 32
DEFAULT_SEEDS: int = 5
MIN_SEEDS: int = 3
SCHEMA_VERSION: str = "oczy/r19-calibration-manifest/v1"

# Parameter budget breakdown (for accounting / audit):
#   W_perceive  896 * 16     = 14336
#   W_label     16 * 20      = 320
#   b_label     20           = 20
#   W_coupler   16 * (3*896) = 43008
#   b_coupler   3*896        = 2688
#   warm_state  16           = 16
#   total                     = 60388  <= 64000
PARAM_BREAKDOWN: dict[str, int] = {
    "W_perceive": D_EMBD * D_CORTEX,
    "W_label": D_CORTEX * N_LABELS,
    "b_label": N_LABELS,
    "W_coupler": D_CORTEX * (LATENT_TOKENS * D_EMBD),
    "b_coupler": LATENT_TOKENS * D_EMBD,
    "warm_state": D_CORTEX,
}
TOTAL_PARAMS: int = sum(PARAM_BREAKDOWN.values())


# ---------------------------------------------------------------------------
# Label management
# ---------------------------------------------------------------------------


def extract_stage_labels(stage: Stage) -> list[str]:
    """Return the sorted-unique corrected labels from *stage* episodes.

    Stage 0 has exactly 20 unique labels; these form the label head's
    output space.  The sort is by first appearance to preserve a stable
    index assignment independent of set iteration order.
    """
    seen: dict[str, int] = {}
    for ep in stage.episodes:
        label = ep.corrected_label
        if label not in seen:
            seen[label] = len(seen)
    return list(seen.keys())


def build_label_index(labels: list[str]) -> dict[str, int]:
    """Map label text → integer index."""
    return {label: i for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Model hashing
# ---------------------------------------------------------------------------


def hash_model(driver: HFDriver) -> str:
    """Return a stable SHA-256 identity hash for the loaded model.

    Hashes model_id, the full config JSON, the SHA-256 of EVERY byte of
    every named parameter tensor, and the tokenizer vocab/file hash.
    This detects any weight change (bit-identical requirement) and any
    tokenizer change.
    """
    cfg = getattr(driver._model, "config", None)
    cfg_dict: dict[str, Any]
    if cfg is not None:
        cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else {"repr": repr(cfg)}
    else:
        cfg_dict = {}

    # Hash every byte of every named parameter tensor (bit-identical check).
    param_hashes: list[str] = []
    for name, param in sorted(driver._model.named_parameters()):
        data = param.detach().cpu().numpy().tobytes()
        param_hashes.append(f"{name}:{hashlib.sha256(data).hexdigest()}")

    # Hash the tokenizer vocab and special tokens.
    tokenizer = driver._tokenizer
    tokenizer_hash = ""
    try:
        vocab_items = sorted(tokenizer.get_vocab().items())
        vocab_payload = json.dumps(vocab_items, default=str)
        tokenizer_hash = hashlib.sha256(vocab_payload.encode()).hexdigest()
    except Exception:
        tokenizer_hash = hashlib.sha256(repr(tokenizer).encode()).hexdigest()

    payload = json.dumps(
        {
            "model_id": driver.model_id,
            "config": cfg_dict,
            "param_hashes": param_hashes,
            "tokenizer_hash": tokenizer_hash,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_eval_manifest() -> str:
    """Return SHA-256 over the eval/v2 MANIFEST.json contents."""
    from eval.v2 import get_data_dir  # type: ignore[missing-import]

    manifest_path = get_data_dir() / "MANIFEST.json"
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()

def hash_model_config(driver: HFDriver) -> str:
    """Return SHA-256 of the model config.json (serialised via config.to_dict)."""
    cfg = getattr(driver._model, "config", None)
    if cfg is not None and hasattr(cfg, "to_dict"):
        cfg_dict = cfg.to_dict()
    elif cfg is not None:
        cfg_dict = {"repr": repr(cfg)}
    else:
        cfg_dict = {}
    payload = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_model_safetensors(driver: HFDriver) -> str:
    """Return SHA-256 over every byte of every named parameter tensor.

    This is the bit-identical weight fingerprint — the same substance as
    ``hash_model`` but restricted to weight bytes only (no config/tokenizer).
    """
    param_hashes: list[str] = []
    for name, param in sorted(driver._model.named_parameters()):
        data = param.detach().cpu().numpy().tobytes()
        param_hashes.append(f"{name}:{hashlib.sha256(data).hexdigest()}")
    payload = json.dumps(param_hashes, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def hash_head_state(cortex: SharedCortex) -> tuple[str, int]:
    """Return (sha256, byte_count) for the head state (W_perceive, W_label,
    b_label, warm_state — everything except the coupler).
    """
    head_state = {
        "W_perceive": cortex.W_perceive.detach().cpu(),
        "W_label": cortex.W_label.detach().cpu(),
        "b_label": cortex.b_label.detach().cpu(),
        "warm_state": cortex.warm_state.detach().cpu(),
    }
    payload = pickle.dumps(head_state, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def hash_cortex_artifact(cortex: SharedCortex) -> tuple[str, int]:
    """Return (sha256, byte_count) for the full serialized cortex (head+coupler).

    This covers the entire persistent cortex artifact — all six parameter
    tensors — so evaluate can verify the complete artifact integrity.
    """
    state = cortex.state_dict()
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def hash_coupler_state(cortex: SharedCortex) -> tuple[str, int]:
    """Return (sha256, byte_count) for the coupler state artifact."""
    state = cortex.coupler_state()
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def derive_source_provenance(
    source_commit: str | None,
    source_archive: str | None,
) -> tuple[str, str]:
    """Derive (commit, archive_sha256) provenance, never fabricating.

    Priority:
      1. Explicit CLI args (source_commit, source_archive path).
      2. Environment variables OCZY_SOURCE_COMMIT / OCZY_SOURCE_ARCHIVE.
      3. ``"unavailable"`` sentinel — never invented.

    When an archive path is provided, its SHA-256 is computed from the file.
    When only the hash is available (OCZY_SOURCE_ARCHIVE_SHA256), that is used
    directly.
    """
    import os

    commit = source_commit or os.environ.get("OCZY_SOURCE_COMMIT", "")
    archive_path = source_archive or os.environ.get("OCZY_SOURCE_ARCHIVE", "")
    archive_sha = os.environ.get("OCZY_SOURCE_ARCHIVE_SHA256", "")

    if not commit:
        commit = "unavailable"
    if not archive_sha and archive_path:
        h = hashlib.sha256()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        archive_sha = h.hexdigest()
    if not archive_sha:
        archive_sha = "unavailable"

    return commit, archive_sha


# ---------------------------------------------------------------------------
# Shared cortex
# ---------------------------------------------------------------------------


@dataclass
class CortexConfig:
    """Configuration for the shared cortex."""

    d_embd: int = D_EMBD
    d_cortex: int = D_CORTEX
    latent_tokens: int = LATENT_TOKENS
    n_labels: int = N_LABELS
    lr: float = 0.01
    coupler_lr: float = 0.001
    label_lr: float = 0.05


class SharedCortex:
    """Shared trainable cortex: perception, label head, latent coupler.

    All parameters are float32 torch tensors.  The coupler (W_coupler,
    b_coupler) can be frozen after calibration while the rest are trained
    per-seed.  Total persistent parameters: 60388 (<= 64000).

    Parameter shapes:
        W_perceive  (d_embd, d_cortex)       — request feature projection
        W_label     (d_cortex, n_labels)     — label head
        b_label     (n_labels,)              — label head bias
        W_coupler   (d_cortex, latent*d_embd) — latent bank projection
        b_coupler   (latent*d_embd,)         — latent bank bias
        warm_state  (d_cortex,)              — persistent state bias
    """

    def __init__(self, config: CortexConfig | None = None, seed: int = 0) -> None:
        self.config = config or CortexConfig()
        c = self.config
        rng = torch.Generator().manual_seed(seed)

        # Perception projection
        self.W_perceive = torch.nn.Parameter(
            torch.randn(c.d_embd, c.d_cortex, generator=rng) * 0.05
        )
        # Label head
        self.W_label = torch.nn.Parameter(
            torch.randn(c.d_cortex, c.n_labels, generator=rng) * 0.1
        )
        self.b_label = torch.nn.Parameter(torch.zeros(c.n_labels))
        # Latent coupler
        self.W_coupler = torch.nn.Parameter(
            torch.randn(c.d_cortex, c.latent_tokens * c.d_embd, generator=rng) * 0.02
        )
        self.b_coupler = torch.nn.Parameter(torch.zeros(c.latent_tokens * c.d_embd))
        # Persistent state
        self.warm_state = torch.nn.Parameter(torch.zeros(c.d_cortex))

        self._coupler_frozen = False
        self._perception_frozen = False

    # -- freezing ----------------------------------------------------------

    def freeze_coupler(self) -> None:
        """Freeze the coupler parameters (after calibration)."""
        self.W_coupler.requires_grad_(False)
        self.b_coupler.requires_grad_(False)
        self._coupler_frozen = True

    def unfreeze_coupler(self) -> None:
        self.W_coupler.requires_grad_(True)
        self.b_coupler.requires_grad_(True)
        self._coupler_frozen = False

    @property
    def coupler_frozen(self) -> bool:
        return self._coupler_frozen

    def freeze_perception(self) -> None:
        """Freeze perception and label parameters (for coupler DEV training).

        During calibrate-dev, only the coupler is trained; W_perceive,
        W_label, b_label, and warm_state are frozen to prevent the coupler
        training from corrupting the perception/label representations.
        """
        self.W_perceive.requires_grad_(False)
        self.W_label.requires_grad_(False)
        self.b_label.requires_grad_(False)
        self.warm_state.requires_grad_(False)
        self._perception_frozen = True

    def unfreeze_perception(self) -> None:
        """Unfreeze perception and label parameters."""
        self.W_perceive.requires_grad_(True)
        self.W_label.requires_grad_(True)
        self.b_label.requires_grad_(True)
        self.warm_state.requires_grad_(True)
        self._perception_frozen = False

    @property
    def perception_frozen(self) -> bool:
        return self._perception_frozen

    def coupler_parameters(self) -> list[torch.nn.Parameter]:
        """Return only the coupler parameters (for coupler-only training)."""
        return [self.W_coupler, self.b_coupler]

    # -- forward ------------------------------------------------------------

    def perceive(self, request_features: np.ndarray) -> torch.Tensor:
        """Project request features to cortex activation.

        Args:
            request_features: (d_embd,) mean-pooled frozen LM features.

        Returns:
            (d_cortex,) cortex activation tensor.
        """
        x = torch.from_numpy(request_features).float()
        return x @ self.W_perceive + self.warm_state

    def predict_label(self, cortex_act: torch.Tensor) -> tuple[int, float]:
        """Predict label index and confidence (softmax prob).

        Returns:
            (label_index, confidence) where confidence is the max softmax
            probability.
        """
        logits = cortex_act @ self.W_label + self.b_label
        probs = F.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return int(idx.item()), float(conf.item())

    def compute_latent(self, cortex_act: torch.Tensor) -> torch.Tensor:
        """Compute the fixed-width latent bank.

        Returns:
            (latent_tokens, d_embd) latent bank tensor.
        """
        c = self.config
        latent = cortex_act @ self.W_coupler + self.b_coupler
        return latent.view(c.latent_tokens, c.d_embd)

    # -- training -----------------------------------------------------------

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return parameters that require gradients."""
        params = [self.W_perceive, self.W_label, self.b_label, self.warm_state]
        if not self._coupler_frozen:
            params.extend([self.W_coupler, self.b_coupler])
        return [p for p in params if p.requires_grad]

    def train_label_head(
        self,
        request_features: np.ndarray,
        label_idx: int,
    ) -> float:
        """One online correction step on the label head.

        Cross-entropy loss against the corrected label index.  Updates
        W_perceive, W_label, b_label, warm_state (and coupler if not frozen).
        """
        cortex_act = self.perceive(request_features)
        logits = cortex_act @ self.W_label + self.b_label
        target = torch.tensor([label_idx], dtype=torch.long)
        loss = F.cross_entropy(logits.unsqueeze(0), target)

        opt = torch.optim.SGD(
            self.trainable_parameters(),
            lr=self.config.label_lr,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def train_coupler(
        self,
        driver: HFDriver,
        request: str,
        corrected_response: str,
    ) -> float:
        """One online correction step on the coupler via LM steering.

        Computes the latent bank from cortex activation, prepends it to
        the request embeddings, forwards through the frozen LM, and
        computes cross-entropy on the corrected response tokens.

        Gradient flows: loss → LM (frozen) → inputs_embeds → latent →
        W_coupler, b_coupler → cortex_act → W_perceive.
        """
        c = self.config
        tokenizer = driver._tokenizer
        model = driver._model

        # Ensure LM params are frozen.
        for p in model.parameters():
            p.requires_grad_(False)

        features = driver.peek_embedding(request, last_token_only=False)
        cortex_act = self.perceive(features)

        # Latent bank (latent_tokens, d_embd).
        latent = self.compute_latent(cortex_act)  # (L, D)

        # Request token embeddings (frozen).
        input_ids = driver._tokenize(request)
        embed_fn = cast(Callable[..., torch.Tensor], cast(Any, model).get_input_embeddings())
        request_embeds = embed_fn(input_ids)  # (1, S, D)

        # Answer tokens for teacher forcing.
        answer_text = " " + corrected_response
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        if not answer_ids:
            return 0.0
        answer_tensor = torch.tensor([answer_ids], dtype=torch.long)
        answer_embeds = embed_fn(answer_tensor)  # (1, A, D)

        # Concatenate: [latent, request, answer]
        latent_batch = latent.unsqueeze(0)  # (1, L, D)
        inputs_embeds = torch.cat(
            [latent_batch, request_embeds, answer_embeds], dim=1
        )  # (1, L+S+A, D)

        # Forward through frozen LM.
        out = model(inputs_embeds=inputs_embeds, use_cache=False)

        # Loss on answer positions.
        # Logits at position (L + S - 1) predict first answer token.
        seq_len = request_embeds.shape[1]
        n_answer = len(answer_ids)
        start = c.latent_tokens + seq_len - 1
        end = start + n_answer
        logits = out.logits[0, start:end, :]  # (A, V)
        targets = torch.tensor(answer_ids, dtype=torch.long)
        loss = F.cross_entropy(logits, targets)

        opt = torch.optim.SGD(
            self.coupler_parameters(),
            lr=self.config.coupler_lr,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    # -- controls ----------------------------------------------------------

    def zero_state(self) -> None:
        """Zero the warm_state (C4: causal state test)."""
        with torch.no_grad():
            self.warm_state.zero_()

    def swap_state(self, other: SharedCortex) -> None:
        """Swap only warm_state with another cortex (C5: addressing test).

        Per the spec, the cortex *state* is swapped — only warm_state
        is the persistent dynamic state. W_perceive, W_label, b_label are
        *trained* parameters (not state); the coupler is frozen and shared.
        Only warm_state is copied from *other*.
        """
        with torch.no_grad():
            self.warm_state.copy_(other.warm_state)

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU state dict for persistence."""
        return {
            "W_perceive": self.W_perceive.detach().cpu(),
            "W_label": self.W_label.detach().cpu(),
            "b_label": self.b_label.detach().cpu(),
            "W_coupler": self.W_coupler.detach().cpu(),
            "b_coupler": self.b_coupler.detach().cpu(),
            "warm_state": self.warm_state.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load parameters from a state dict."""
        self.W_perceive.data = state["W_perceive"].float()
        self.W_label.data = state["W_label"].float()
        self.b_label.data = state["b_label"].float()
        self.W_coupler.data = state["W_coupler"].float()
        self.b_coupler.data = state["b_coupler"].float()
        self.warm_state.data = state["warm_state"].float()

    def coupler_state(self) -> dict[str, torch.Tensor]:
        """Return only the coupler parameters (for manifest storage)."""
        return {
            "W_coupler": self.W_coupler.detach().cpu(),
            "b_coupler": self.b_coupler.detach().cpu(),
        }

    def load_coupler(self, state: dict[str, torch.Tensor]) -> None:
        """Load coupler parameters from a state dict."""
        self.W_coupler.data = state["W_coupler"].float()
        self.b_coupler.data = state["b_coupler"].float()

    # -- accounting ---------------------------------------------------------

    def parameter_count(self) -> int:
        """Return total persistent parameter count."""
        return TOTAL_PARAMS

    def persistent_bytes(self) -> int:
        """Return serialized size in bytes (pickle of state dict)."""
        return len(pickle.dumps(self.state_dict(), protocol=pickle.HIGHEST_PROTOCOL))

    def coupler_hash(self) -> str:
        """Return SHA-256 hash of the coupler parameters."""
        state = self.coupler_state()
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Raw-trace lifecycle
# ---------------------------------------------------------------------------


class TraceStore:
    """Tracks raw correction traces and manages their deletion.

    The spec requires: 'delete all correction texts and transient optimizer
    examples, and verify raw-trace count zero' before evaluation.
    """

    def __init__(self) -> None:
        self._traces: list[dict[str, Any]] = []
        self._cached_features: dict[str, np.ndarray] = {}
        self._cached_embeddings: dict[tuple[str, bool], np.ndarray] = {}

    def add(
        self,
        episode_id: str,
        request: str,
        correction: str,
        response: str,
    ) -> None:
        """Record one raw correction trace."""
        self._traces.append(
            {
                "episode_id": episode_id,
                "request": request,
                "correction": correction,
                "response": response,
            }
        )

    def add_cached_feature(self, request: str, features: np.ndarray) -> None:
        """Register a cached request feature vector for later deletion."""
        self._cached_features[request] = features

    def add_cached_embedding(
        self, key: tuple[str, bool], embedding: np.ndarray
    ) -> None:
        """Register a cached embedding for later deletion."""
        self._cached_embeddings[key] = embedding

    def count(self) -> int:
        return len(self._traces)

    def cached_feature_count(self) -> int:
        return len(self._cached_features)

    def cached_embedding_count(self) -> int:
        return len(self._cached_embeddings)

    def delete_all(self) -> int:
        """Delete all raw traces and cached state.  Returns trace count deleted."""
        n = len(self._traces)
        self._traces.clear()
        self._cached_features.clear()
        self._cached_embeddings.clear()
        return n

    def verify_zero(self) -> bool:
        """Return True if all traces and cached state are zero."""
        return (
            len(self._traces) == 0
            and len(self._cached_features) == 0
            and len(self._cached_embeddings) == 0
        )


# ---------------------------------------------------------------------------
# Arm articulation
# ---------------------------------------------------------------------------


def arm_a_generate(
    driver: HFDriver,
    cortex: SharedCortex,
    request: str,
    labels: list[str],
    confidence_threshold: float,
    label_prefix_template: str = LABEL_PREFIX_TEMPLATE,
) -> tuple[str, dict[str, Any]]:
    """Arm A: predict label, supply as text prefix (parametric retrieval).

    Returns:
        (generated_text, audit_info)
    """
    features = driver.peek_embedding(request, last_token_only=False)
    cortex_act = cortex.perceive(features)
    label_idx, conf = cortex.predict_label(cortex_act)

    audit: dict[str, Any] = {
        "arm": "A",
        "label_idx": label_idx,
        "confidence": conf,
        "threshold": confidence_threshold,
        "abstained": conf < confidence_threshold,
        "prompt_text": request,
        "latent_bank_shape": None,
        "prefix_text": None,
    }

    if conf < confidence_threshold:
        # Abstain: fall through to unmodified LM.
        answer = driver.generate(request, max_tokens=MAX_GENERATE_TOKENS)
        audit["prefix_text"] = None
        return answer, audit

    label_text = labels[label_idx]
    prefix = label_prefix_template.format(label=label_text)
    audit["prefix_text"] = prefix

    driver.set_reserved_position(cast(Any, ReservedPosition(text=prefix)))
    try:
        answer = driver.generate(request, max_tokens=MAX_GENERATE_TOKENS)
    finally:
        driver.clear_reserved_position()
    return answer, audit


def arm_b_generate(
    driver: HFDriver,
    cortex: SharedCortex,
    request: str,
    confidence_threshold: float,
) -> tuple[str, dict[str, Any]]:
    """Arm B: inject latent bank via inputs_embeds (latent control).

    No label, answer, correction, exemplar, or episode-ID text enters the
    LM at probe time.  The only inputs are the request, frozen LM features
    of the request, query-conditioned cortex activations, and the
    fixed-width latent bank.

    Returns:
        (generated_text, audit_info)
    """
    c = cortex.config
    model = driver._model
    tokenizer = driver._tokenizer

    # 1. Request features (mean-pooled final hidden).
    features = driver.peek_embedding(request, last_token_only=False)

    # 2. Cortex activation (query-conditioned, from persistent params).
    cortex_act = cortex.perceive(features)

    # 3. Confidence check for abstain path.
    _, conf = cortex.predict_label(cortex_act)

    audit: dict[str, Any] = {
        "arm": "B",
        "confidence": conf,
        "threshold": confidence_threshold,
        "abstained": conf < confidence_threshold,
        "prompt_text": request,
        "latent_bank_shape": (c.latent_tokens, c.d_embd),
        "raw_trace_count": None,  # filled by caller
    }

    if conf < confidence_threshold:
        # Abstain: fall through to unmodified LM.
        answer = driver.generate(request, max_tokens=MAX_GENERATE_TOKENS)
        audit["latent_bank_shape"] = None
        return answer, audit

    # 4. Compute latent bank (fixed shape, no text content).
    latent = cortex.compute_latent(cortex_act)  # (L, D)

    # 5. Get request token embeddings (frozen).
    input_ids = driver._tokenize(request)
    embed_fn = cast(Callable[..., torch.Tensor], cast(Any, model).get_input_embeddings())
    request_embeds = embed_fn(input_ids)  # (1, S, D)

    # 6. Concatenate: [latent, request]
    latent_batch = latent.detach().unsqueeze(0)  # (1, L, D)
    inputs_embeds = torch.cat([latent_batch, request_embeds], dim=1)

    # 7. Forward and greedy decode.
    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, use_cache=True)
        past = out.past_key_values
        next_token = int(out.logits[0, -1, :].argmax().item())
        generated_ids: list[int] = [next_token]
        eos_id = tokenizer.eos_token_id

        for _ in range(MAX_GENERATE_TOKENS - 1):
            if next_token == eos_id:
                break
            token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            out = model(
                input_ids=token_tensor,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_token = int(out.logits[0, -1, :].argmax().item())
            generated_ids.append(next_token)

    answer = tokenizer.decode(generated_ids) if generated_ids else ""
    return answer, audit


def vanilla_generate(
    driver: HFDriver,
    request: str,
) -> tuple[str, dict[str, Any]]:
    """Vanilla baseline: unmodified frozen LM."""
    answer = driver.generate(request, max_tokens=MAX_GENERATE_TOKENS)
    audit: dict[str, Any] = {
        "arm": "vanilla",
        "prompt_text": request,
        "latent_bank_shape": None,
    }
    return answer, audit


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _probe_id(episode: Episode, probe: Probe) -> str:
    return f"{episode.id}|{probe.request}|{probe.category}"


def score_probes(
    driver: HFDriver,
    stage: Stage,
    probe_ids: set[str],
    arm: str,
    cortex: SharedCortex | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    label_prefix_template: str = LABEL_PREFIX_TEMPLATE,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """Score a set of probes under one arm.

    Returns a dict with accuracy, total, and per-probe audit records.
    """
    if arm in ("A", "B"):
        if cortex is None:
            raise ValueError(f"Arm {arm} requires a cortex")
    if arm == "A" and labels is None:
        raise ValueError("Arm A requires labels")
    results: list[bool] = []
    audits: list[dict[str, Any]] = []

    for ep in stage.episodes:
        for probe in ep.probes:
            pid = _probe_id(ep, probe)
            if pid not in probe_ids:
                continue

            if arm == "vanilla":
                answer, audit = vanilla_generate(driver, probe.request)
            elif arm == "A":
                if cortex is None or labels is None:
                    raise ValueError("Arm A requires cortex and labels")
                answer, audit = arm_a_generate(
                    driver, cortex, probe.request, labels,
                    confidence_threshold, label_prefix_template,
                )
            elif arm == "B":
                if cortex is None:
                    raise ValueError("Arm B requires a cortex")
                answer, audit = arm_b_generate(
                    driver, cortex, probe.request, confidence_threshold,
                )
            else:
                raise ValueError(f"Unknown arm: {arm}")

            correct = probe_matches(answer, probe, ep)
            results.append(correct)

            audit.update({
                "probe_id": pid,
                "episode_id": ep.id,
                "correct": correct,
                "raw_trace_count": trace_store.count() if trace_store else None,
            })
            audits.append(audit)

    total = len(results)
    correct_count = sum(results)
    accuracy = correct_count / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "audits": audits,
    }


def score_specificity(
    driver: HFDriver,
    other_stages: tuple[Stage, ...],
    arm: str,
    cortex: SharedCortex | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    label_prefix_template: str = LABEL_PREFIX_TEMPLATE,
    trace_store: TraceStore | None = None,
    use_holdout: bool = True,
) -> dict[str, Any]:
    """Score specificity on untaught stages' probes.

    When use_holdout=True (evaluate phase), uses holdout probes.
    When use_holdout=False (calibrate-dev phase), uses DEV probes only
    to respect the DEV/holdout firewall.

    Returns accuracy on the union of all other stages' probes.
    """
    all_results: list[bool] = []
    all_audits: list[dict[str, Any]] = []

    for stage in other_stages:
        dev_ids, holdout_ids = split_probes(stage, fraction=0.3, salt="v2.2")
        probe_ids = holdout_ids if use_holdout else dev_ids
        result = score_probes(
            driver, stage, probe_ids, arm, cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        all_results.extend([a["correct"] for a in result["audits"]])
        all_audits.extend(result["audits"])

    total = len(all_results)
    correct_count = sum(all_results)
    accuracy = correct_count / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total,
        "audits": all_audits,
    }


# ---------------------------------------------------------------------------
# Teaching
# ---------------------------------------------------------------------------


def teach_cortex(
    driver: HFDriver,
    cortex: SharedCortex,
    stage: Stage,
    labels: list[str],
    label_index: dict[str, int],
    seed: int,
    trace_store: TraceStore,
    permuted_labels: bool = False,
) -> dict[str, Any]:
    """Teach the cortex with stage-0 corrections in seed-shuffled order.

    For each correction:
      1. Train label head (cross-entropy on corrected_label).

    The coupler is NOT trained here — it was frozen during calibrate-dev.
    Only W_perceive, W_label, b_label, and warm_state are updated.

    If *permuted_labels* is True (C6), the label indices are permuted
    during teaching to test feedback semantics. The corrected_response
    is still recorded in the trace store (same as non-permuted), but the
    label head learns wrong label→request mappings.

    All raw correction texts are recorded in *trace_store*.
    """

    rng = random.Random(seed)
    teach_order = list(stage.episodes)
    rng.shuffle(teach_order)

    # Build a fixed permutation for C6.
    if permuted_labels:
        perm = list(range(len(labels)))
        rng.shuffle(perm)
    else:
        perm = list(range(len(labels)))

    label_losses: list[float] = []

    for ep in teach_order:
        # Record raw trace.
        trace_store.add(
            episode_id=ep.id,
            request=ep.initial_request,
            correction=ep.correction_utterance,
            response=ep.corrected_response,
        )

        # Get request features.
        features = driver.peek_embedding(ep.initial_request, last_token_only=False)
        trace_store.add_cached_feature(ep.initial_request, features)

        # Determine label index (possibly permuted for C6).
        true_idx = label_index[ep.corrected_label]
        teach_idx = perm[true_idx] if permuted_labels else true_idx

        # Train label head only (coupler is frozen).
        label_loss = cortex.train_label_head(features, teach_idx)
        label_losses.append(label_loss)

    return {
        "n_episodes": len(teach_order),
        "label_loss_mean": float(np.mean(label_losses)) if label_losses else 0.0,
        "permuted": permuted_labels,
    }


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------


def run_condition(
    driver: HFDriver,
    condition: str,
    stage: Stage,
    other_stages: tuple[Stage, ...],
    dev_ids: set[str],
    holdout_ids: set[str],
    cortex: SharedCortex | None,
    labels: list[str] | None,
    confidence_threshold: float,
    label_prefix_template: str,
    trace_store: TraceStore,
    swapped_cortex: SharedCortex | None = None,
) -> dict[str, Any]:
    """Run one condition (C0-C7) and return results.

    Conditions:
        C0 — vanilla baseline (no cortex)
        C1 — cortex architecture, update disabled (random init, no training)
        C2 — Arm A: trained head + label prefix
        C3 — Arm B: trained head + latent control
        C4 — C3 with cortex state zeroed
        C5 — C3 with cortex state swapped
        C6 — C3 with permuted labels during teaching (handled in teach)
        C7 — retrieval baseline (external bar, not attached to C3)
    """
    result: dict[str, Any] = {"condition": condition, "dev_probe_count": len(dev_ids)}

    if condition == "C0":
        # Vanilla baseline.
        holdout = score_probes(driver, stage, holdout_ids, "vanilla")
        transfer = _score_transfer(driver, other_stages, "vanilla")
        scope = _score_scope(driver, other_stages, "vanilla")
        specificity = score_specificity(
            driver, other_stages, "vanilla", trace_store=trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": 0,
        })
        return result

    if condition == "C1":
        if cortex is None:
            raise ValueError("C1 requires a cortex")
        # Cortex architecture, no update — random init, no training.
        # Score with Arm B (latent control) but with random cortex.
        holdout = score_probes(
            driver, stage, holdout_ids, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C2":
        if cortex is None:
            raise ValueError("C2 requires a cortex")
        # Arm A: trained head + label prefix.
        holdout = score_probes(
            driver, stage, holdout_ids, "A", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "A", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "A", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "A", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C3":
        if cortex is None:
            raise ValueError("C3 requires a cortex")
        # Arm B: trained head + latent control (primary).
        holdout = score_probes(
            driver, stage, holdout_ids, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C4":
        if cortex is None:
            raise ValueError("C4 requires a cortex")
        # C3 with cortex state zeroed.
        cortex.zero_state()
        holdout = score_probes(
            driver, stage, holdout_ids, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C5":
        if cortex is None:
            raise ValueError("C5 requires a cortex")
        # C3 with cortex state swapped.
        if swapped_cortex is None:
            raise ValueError("C5 requires swapped_cortex")
        cortex.swap_state(swapped_cortex)
        holdout = score_probes(
            driver, stage, holdout_ids, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C6":
        if cortex is None:
            raise ValueError("C6 requires a cortex")
        # C3 with permuted labels during teaching.
        # The cortex was already trained with permuted labels in teach_cortex.
        # Score with Arm B.
        holdout = score_probes(
            driver, stage, holdout_ids, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        transfer = _score_transfer(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        scope = _score_scope(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        specificity = score_specificity(
            driver, other_stages, "B", cortex, labels,
            confidence_threshold, label_prefix_template, trace_store,
        )
        result.update({
            "holdout_acc": holdout["accuracy"],
            "transfer_acc": transfer["accuracy"],
            "scope_acc": scope["accuracy"],
            "specificity_acc": specificity["accuracy"],
            "persistent_bytes": cortex.persistent_bytes(),
        })
        return result

    if condition == "C7":
        # S3.M2a retrieval baseline (external bar only).
        # C7 must use the existing S3.M2a retrieval baseline through a real
        # adapter.  If no adapter is available, return a blocking validity
        # failure rather than a substitute baseline.
        adapter_result = _try_s3m2a_retrieval_adapter()
        if adapter_result is None:
            result.update({
                "holdout_acc": 0.0,
                "transfer_acc": 0.0,
                "scope_acc": 0.0,
                "specificity_acc": 0.0,
                "persistent_bytes": 0,
                "validity_blocked": True,
                "validity_error": (
                    "C7 requires the S3.M2a retrieval baseline adapter; "
                    "no adapter is available"
                ),
            })
        else:
            result.update({
                "holdout_acc": adapter_result["holdout_acc"],
                "transfer_acc": adapter_result["transfer_acc"],
                "scope_acc": 0.0,
                "specificity_acc": 0.0,
                "persistent_bytes": 0,
                "validity_blocked": False,
            })
        return result

    raise ValueError(f"Unknown condition: {condition}")


def _score_transfer(
    driver: HFDriver,
    other_stages: tuple[Stage, ...],
    arm: str,
    cortex: SharedCortex | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    label_prefix_template: str = LABEL_PREFIX_TEMPLATE,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """Score the complete untaught stage-1 transfer battery."""
    # Stage 1 is the transfer stage.
    stage1 = next((s for s in other_stages if "transfer" in s.name.lower()), None)
    if stage1 is None:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "audits": []}

    all_probe_ids: set[str] = set()
    for ep in stage1.episodes:
        for probe in ep.probes:
            all_probe_ids.add(_probe_id(ep, probe))

    return score_probes(
        driver, stage1, all_probe_ids, arm, cortex, labels,
        confidence_threshold, label_prefix_template, trace_store,
    )


def _score_scope(
    driver: HFDriver,
    other_stages: tuple[Stage, ...],
    arm: str,
    cortex: SharedCortex | None = None,
    labels: list[str] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT,
    label_prefix_template: str = LABEL_PREFIX_TEMPLATE,
    trace_store: TraceStore | None = None,
) -> dict[str, Any]:
    """Score stage-2 holdout scope probes."""
    stage2 = next((s for s in other_stages if "scope" in s.name.lower()), None)
    if stage2 is None:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "audits": []}

    _, holdout_ids = split_probes(stage2, fraction=0.3, salt="v2.2")
    # Only score scope probes.
    scope_ids: set[str] = set()
    for ep in stage2.episodes:
        for probe in ep.probes:
            pid = _probe_id(ep, probe)
            if pid in holdout_ids and probe.category == "scope":
                scope_ids.add(pid)

    return score_probes(
        driver, stage2, scope_ids, arm, cortex, labels,
        confidence_threshold, label_prefix_template, trace_store,
    )


def _try_s3m2a_retrieval_adapter() -> dict[str, float] | None:
    """Attempt to run the S3.M2a retrieval baseline through a real adapter.

    The S3.M2a baseline lives in :mod:`oczy.experiments.organ_additive_retrieval`
    and uses :class:`RetrievalOrganism`, which is architecturally incompatible
    with the R19 condition runner (different organism class, different
    teaching/scoring loop).  A real adapter would need to bridge the R19
    probe/stage format to the S3.M2a organism format.

    Returns ``None`` if no adapter is available, signalling a blocking
    validity failure.  C7 must NOT substitute a new nearest-neighbour
    baseline — it must use the real S3.M2a adapter or block.
    """
    # No real adapter exists yet.  Returning None blocks evaluation rather
    # than substituting a new baseline, per the R19 audit contract.
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, lower, upper) 95% CI using normal approximation."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean, mean
    std = float(np.std(arr, ddof=1))
    n = len(arr)
    se = std / math.sqrt(max(n, 1))
    lower = mean - 1.96 * se
    upper = mean + 1.96 * se
    return mean, lower, upper


def ci_excludes_zero(ci: tuple[float, float, float]) -> bool:
    """Return True if the 95% CI excludes zero."""
    return ci[1] > 0 or ci[2] < 0


def compute_phase0_distributions(stage: Stage) -> dict[str, dict[str, int]]:
    """Compute Phase-0 episode/probe distributions for the manifest.

    Returns a dict with ``labels``, ``domains``, and ``probe_categories``
    keys, each mapping to a count distribution.  These are carried in the
    calibration manifest so that evaluate can verify the same Phase-0
    distributions were used.
    """
    label_dist: dict[str, int] = {}
    domain_dist: dict[str, int] = {}
    probe_cat_dist: dict[str, int] = {}
    for ep in stage.episodes:
        label_dist[ep.corrected_label] = label_dist.get(ep.corrected_label, 0) + 1
        domain_dist[ep.domain] = domain_dist.get(ep.domain, 0) + 1
        for probe in ep.probes:
            probe_cat_dist[probe.category] = probe_cat_dist.get(probe.category, 0) + 1
    return {
        "labels": label_dist,
        "domains": domain_dist,
        "probe_categories": probe_cat_dist,
    }


# ---------------------------------------------------------------------------
# Calibration manifest
# ---------------------------------------------------------------------------


def _str_field(
    d: dict[str, Any], key: str, alias: str | None = None, *, default: str = ""
) -> str:
    """Extract a required str, distinguishing missing aliases from explicit null.

    Missing keys resolve to *default*; explicit ``None`` raises ``ValueError``
    so :meth:`CalibrationManifest.required_fields_present` never sees a null
    where a typed value is required.
    """
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required str field")
        if isinstance(v, str):
            return v
        raise ValueError(f"{key}: expected str, got {type(v).__name__}")
    if alias is not None and alias in d:
        v = d[alias]
        if v is None:
            raise ValueError(f"{alias}: explicit null not allowed for legacy str alias")
        if isinstance(v, str):
            return v
        raise ValueError(f"{alias}: expected str, got {type(v).__name__}")
    return default


def _int_field(
    d: dict[str, Any], key: str, alias: str | None = None, *, default: int = 0
) -> int:
    """Extract a required int, distinguishing missing aliases from explicit null."""
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required int field")
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise ValueError(f"{key}: expected int, got {type(v).__name__}")
    if alias is not None and alias in d:
        v = d[alias]
        if v is None:
            raise ValueError(f"{alias}: explicit null not allowed for legacy int alias")
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        raise ValueError(f"{alias}: expected int, got {type(v).__name__}")
    return default


def _float_field(
    d: dict[str, Any], key: str, alias: str | None = None, *, default: float = 0.0
) -> float:
    """Extract a required float, distinguishing missing aliases from explicit null."""
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required float field")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        raise ValueError(f"{key}: expected float, got {type(v).__name__}")
    if alias is not None and alias in d:
        v = d[alias]
        if v is None:
            raise ValueError(f"{alias}: explicit null not allowed for legacy float alias")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        raise ValueError(f"{alias}: expected float, got {type(v).__name__}")
    return default


def _bool_field(
    d: dict[str, Any], key: str, alias: str | None = None, *, default: bool = False
) -> bool:
    """Extract a required bool, distinguishing missing aliases from explicit null."""
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required bool field")
        if isinstance(v, bool):
            return v
        raise ValueError(f"{key}: expected bool, got {type(v).__name__}")
    if alias is not None and alias in d:
        v = d[alias]
        if v is None:
            raise ValueError(f"{alias}: explicit null not allowed for legacy bool alias")
        if isinstance(v, bool):
            return v
        raise ValueError(f"{alias}: expected bool, got {type(v).__name__}")
    return default


def _optional_str_field(
    d: dict[str, Any], key: str, alias: str | None = None
) -> str | None:
    """Extract an optional str that may legitimately be null."""
    if key in d:
        v: Any = d[key]
        if v is None:
            return None
        if isinstance(v, str):
            return v
        raise ValueError(f"{key}: expected str or null, got {type(v).__name__}")
    if alias is not None and alias in d:
        v = d[alias]
        if v is None:
            return None
        if isinstance(v, str):
            return v
        raise ValueError(f"{alias}: expected str or null, got {type(v).__name__}")
    return None


def _dict_str_int_field(
    d: dict[str, Any], key: str, *, default: dict[str, int]
) -> dict[str, int]:
    """Extract a required dict[str, int], distinguishing missing from null.

    Missing key resolves to *default*; explicit ``None`` or a non-dict
    container raises ``ValueError`` so the manifest fails closed before
    evaluate.  Values must be ints (bool rejected).
    """
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required dict field")
        if not isinstance(v, dict):
            raise ValueError(f"{key}: expected dict, got {type(v).__name__}")
        for dk, dv in v.items():
            if not isinstance(dk, str):
                raise ValueError(f"{key}: dict keys must be str, got {type(dk).__name__}")
            if not isinstance(dv, int) or isinstance(dv, bool):
                raise ValueError(f"{key}: dict values must be int, got {type(dv).__name__}")
        return v
    return dict(default)


def _list_int_shape_field(
    d: dict[str, Any], key: str, *, expected: tuple[int, ...], default: list[int]
) -> list[int]:
    """Extract a required list[int] with a fixed expected shape.

    Missing key resolves to *default*; explicit ``None``, a non-list
    container, non-int elements, or a shape mismatch raises ``ValueError``
    so the manifest fails closed before evaluate.  bool elements are
    rejected (bool is a subclass of int).
    """
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required list field")
        if not isinstance(v, list):
            raise ValueError(f"{key}: expected list, got {type(v).__name__}")
        for el in v:
            if not isinstance(el, int) or isinstance(el, bool):
                raise ValueError(f"{key}: list elements must be int, got {type(el).__name__}")
        if tuple(v) != expected:
            raise ValueError(f"{key}: expected shape {list(expected)}, got {v}")
        return list(v)
    return list(default)


def _list_str_field(
    d: dict[str, Any], key: str, *, default: list[str]
) -> list[str]:
    """Extract a required list[str], distinguishing missing from null.

    Missing key resolves to *default*; explicit ``None`` or a non-list
    container raises ``ValueError``.  Element type is checked in
    :meth:`required_fields_present` to allow an empty default through
    ``from_dict`` while still failing closed at validation time.
    """
    if key in d:
        v: Any = d[key]
        if v is None:
            raise ValueError(f"{key}: explicit null not allowed for required list field")
        if not isinstance(v, list):
            raise ValueError(f"{key}: expected list, got {type(v).__name__}")
        return v
    return list(default)

@dataclass
class CalibrationManifest:
    """Flat calibration manifest produced by calibrate-dev, consumed by evaluate.

    Every field is emitted as a top-level key in ``to_dict()`` — no nested
    objects.  The ``manifest_sha256`` is SHA-256 of canonical JSON
    (``sort_keys=True``, ``separators=(',', ':')``) over **all** fields
    except ``created_at`` and ``manifest_sha256`` itself, so re-running
    calibrate-dev at a different time with identical calibration values
    produces an identical hash.

    Required contract fields (all present in every emitted dict):
    +  - schema_version (const ``oczy/r19-calibration-manifest/v1``)
    +  - created_at (ISO timestamp, excluded from hash)
    +  - source provenance (commit + archive_sha256, or explicit unavailable)
    +  - eval provenance (version + manifest_sha256)
    +  - model provenance (repo_id, revision, config/safetensors hashes)
    +  - architecture dimensions (d_embd, d_cortex, latent_tokens, max_labels, arm_b_input_mode)
    +  - parameters (total, budget, breakdown)
    +  - fixed_latent_shape
    +  - proposed thresholds (confidence_threshold, specificity_margin)
    +  - cortex artifact (sha256 + bytes, covers head+coupler)
    +  - coupler/head state (sha256 + bytes)
    +  - label phrasing (frozen + labels)
    +  - dev distribution (split, repeatability, confidence, specificity)
    +  - split (salt + fraction)
    +  - c7 retrieval baseline (reference, available, blocked_reason)
    +  - trace deletion audit (raw_traces_deleted, raw_trace_count, embedding_cache_cleared, optimizer_state_deleted)
    +  - articulation audit (prompt, latent shape, raw traces, LM hash, persistent bytes, banned-content booleans)
    +  - signoff (thresholds_signed_off, human_signoff_id, oracle_ceiling, dev_articulation_gate, meta_test_conflation_ok)
    +  - holdout_accessed (const false)
    +  - manifest_sha256
    """

    # -- schema / timestamp ------------------------------------------------
    schema_version: str = SCHEMA_VERSION
    created_at: str = ""

    # -- source provenance -------------------------------------------------
    source_commit: str = ""
    source_archive_sha256: str = ""

    # -- eval provenance ---------------------------------------------------
    eval_version: str = ""
    eval_manifest_sha256: str = ""

    # -- model provenance --------------------------------------------------
    model_repo_id: str = ""
    model_revision: str = ""
    model_config_sha256: str = ""
    model_safetensors_sha256: str = ""
    model_params_requires_grad: bool = False

    # -- architecture ------------------------------------------------------
    d_embd: int = D_EMBD
    d_cortex: int = D_CORTEX
    latent_tokens: int = LATENT_TOKENS
    max_labels: int = N_LABELS
    arm_b_input_mode: str = "inputs_embeds"

    # -- parameters --------------------------------------------------------
    parameter_total: int = TOTAL_PARAMS
    parameter_budget: int = MAX_PARAMS
    parameter_breakdown: dict[str, int] = field(default_factory=lambda: dict(PARAM_BREAKDOWN))

    # -- fixed latent shape ------------------------------------------------
    fixed_latent_shape: list[int] = field(default_factory=lambda: [LATENT_TOKENS, D_EMBD])

    # -- proposed thresholds -----------------------------------------------
    proposed_confidence_threshold: float = CONFIDENCE_THRESHOLD_DEFAULT
    proposed_specificity_margin: float = SPECIFICITY_MARGIN_DEFAULT

    # -- cortex artifact (full serialized head+coupler) --------------------
    cortex_artifact_sha256: str = ""
    cortex_artifact_bytes: int = 0
    cortex_artifact_path: str = ""

    # -- coupler / head state artifacts ------------------------------------
    coupler_sha256: str = ""
    coupler_bytes: int = 0
    head_sha256: str = ""
    head_bytes: int = 0

    # -- label phrasing ----------------------------------------------------
    label_phrasing_frozen: bool = True
    labels: list[str] = field(default_factory=list)

    # -- dev distribution --------------------------------------------------
    dev_split: str = "dev"
    dev_repeatability_std: float = 0.0
    dev_confidence_mean: float = 0.0
    dev_confidence_std: float = 0.0
    dev_confidence_min: float = 0.0
    dev_confidence_max: float = 0.0
    dev_specificity_acc: float = 0.0
    dev_holdout_ids_discarded: bool = True

    # -- split -------------------------------------------------------------
    split_salt: str = "v2.2"
    split_fraction: float = 0.3

    # -- c7 retrieval baseline ---------------------------------------------
    c7_reference: str = ""
    c7_available: bool = False
    c7_blocked_reason: str | None = None

    # -- trace deletion audit ----------------------------------------------
    trace_raw_traces_deleted: bool = False
    trace_raw_trace_count: int = 0
    trace_embedding_cache_cleared: bool = False
    trace_optimizer_state_deleted: bool = False

    # -- articulation audit (representative Arm-B record) ------------------
    articulation_prompt_text: str = ""
    articulation_latent_bank_shape: list[int] = field(default_factory=lambda: [LATENT_TOKENS, D_EMBD])
    articulation_raw_trace_count: int = 0
    articulation_language_organ_hash: str = ""
    articulation_persistent_cortex_bytes: int = 0
    articulation_banned_label_text_absent: bool = True
    articulation_banned_corrected_response_absent: bool = True
    articulation_banned_correction_utterance_absent: bool = True
    articulation_banned_expected_answer_absent: bool = True

    # -- signoff -----------------------------------------------------------
    signoff_thresholds_signed_off: bool = False
    signoff_human_signoff_id: str = ""
    signoff_oracle_ceiling: float = 0.0
    signoff_dev_articulation_gate: bool = False
    signoff_meta_test_conflation_ok: bool = False

    # -- holdout firewall --------------------------------------------------
    holdout_accessed: bool = False

    # -- manifest hash (excluded from hash computation) --------------------
    manifest_sha256: str = ""

    # Fields excluded from the canonical hash payload.
    _HASH_EXCLUDED: tuple[str, ...] = field(default=("created_at", "manifest_sha256"), repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the flat manifest dict with a computed ``manifest_sha256``."""
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source_commit": self.source_commit,
            "source_archive_sha256": self.source_archive_sha256,
            "eval_version": self.eval_version,
            "eval_manifest_sha256": self.eval_manifest_sha256,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "model_config_sha256": self.model_config_sha256,
            "model_safetensors_sha256": self.model_safetensors_sha256,
            "model_params_requires_grad": self.model_params_requires_grad,
            "d_embd": self.d_embd,
            "d_cortex": self.d_cortex,
            "latent_tokens": self.latent_tokens,
            "max_labels": self.max_labels,
            "arm_b_input_mode": self.arm_b_input_mode,
            "parameter_total": self.parameter_total,
            "parameter_budget": self.parameter_budget,
            "parameter_breakdown": dict(self.parameter_breakdown),
            "fixed_latent_shape": list(self.fixed_latent_shape),
            "proposed_confidence_threshold": self.proposed_confidence_threshold,
            "proposed_specificity_margin": self.proposed_specificity_margin,
            "cortex_artifact_sha256": self.cortex_artifact_sha256,
            "cortex_artifact_bytes": self.cortex_artifact_bytes,
            "cortex_artifact_path": self.cortex_artifact_path,
            "coupler_sha256": self.coupler_sha256,
            "coupler_bytes": self.coupler_bytes,
            "head_sha256": self.head_sha256,
            "head_bytes": self.head_bytes,
            "label_phrasing_frozen": self.label_phrasing_frozen,
            "labels": list(self.labels),
            "dev_split": self.dev_split,
            "dev_repeatability_std": self.dev_repeatability_std,
            "dev_confidence_mean": self.dev_confidence_mean,
            "dev_confidence_std": self.dev_confidence_std,
            "dev_confidence_min": self.dev_confidence_min,
            "dev_confidence_max": self.dev_confidence_max,
            "dev_specificity_acc": self.dev_specificity_acc,
            "dev_holdout_ids_discarded": self.dev_holdout_ids_discarded,
            "split_salt": self.split_salt,
            "split_fraction": self.split_fraction,
            "c7_reference": self.c7_reference,
            "c7_available": self.c7_available,
            "c7_blocked_reason": self.c7_blocked_reason,
            "trace_raw_traces_deleted": self.trace_raw_traces_deleted,
            "trace_raw_trace_count": self.trace_raw_trace_count,
            "trace_embedding_cache_cleared": self.trace_embedding_cache_cleared,
            "trace_optimizer_state_deleted": self.trace_optimizer_state_deleted,
            "articulation_prompt_text": self.articulation_prompt_text,
            "articulation_latent_bank_shape": list(self.articulation_latent_bank_shape),
            "articulation_raw_trace_count": self.articulation_raw_trace_count,
            "articulation_language_organ_hash": self.articulation_language_organ_hash,
            "articulation_persistent_cortex_bytes": self.articulation_persistent_cortex_bytes,
            "articulation_banned_label_text_absent": self.articulation_banned_label_text_absent,
            "articulation_banned_corrected_response_absent": self.articulation_banned_corrected_response_absent,
            "articulation_banned_correction_utterance_absent": self.articulation_banned_correction_utterance_absent,
            "articulation_banned_expected_answer_absent": self.articulation_banned_expected_answer_absent,
            "signoff_thresholds_signed_off": self.signoff_thresholds_signed_off,
            "signoff_human_signoff_id": self.signoff_human_signoff_id,
            "signoff_oracle_ceiling": self.signoff_oracle_ceiling,
            "signoff_dev_articulation_gate": self.signoff_dev_articulation_gate,
            "signoff_meta_test_conflation_ok": self.signoff_meta_test_conflation_ok,
            "holdout_accessed": self.holdout_accessed,
        }
        # Canonical hash: sorted keys, compact separators, exclude
        # created_at and manifest_sha256.
        hash_payload = {k: v for k, v in d.items() if k not in self._HASH_EXCLUDED}
        canonical = json.dumps(hash_payload, sort_keys=True, default=str, separators=(",", ":"))
        d["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return d

    def compute_hash(self) -> str:
        """Return the manifest hash (over all fields except excluded)."""
        return self.to_dict()["manifest_sha256"]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationManifest:
        """Reconstruct from a flat dict, tolerating older field names.

        Typed helpers distinguish missing aliases (→ typed default) from
        explicit ``None`` (→ ``ValueError`` for required fields).  The
        optional ``c7_blocked_reason`` field preserves ``None``.
        """
        return cls(
            schema_version=_str_field(d, "schema_version", default=SCHEMA_VERSION),
            created_at=_str_field(d, "created_at", "calibration_timestamp"),
            source_commit=_str_field(d, "source_commit"),
            source_archive_sha256=_str_field(d, "source_archive_sha256"),
            eval_version=_str_field(d, "eval_version"),
            eval_manifest_sha256=_str_field(d, "eval_manifest_sha256", "eval_manifest_hash"),
            model_repo_id=_str_field(d, "model_repo_id", "model_id"),
            model_revision=_str_field(d, "model_revision"),
            model_config_sha256=_str_field(d, "model_config_sha256"),
            model_safetensors_sha256=_str_field(d, "model_safetensors_sha256", "model_hash"),
            model_params_requires_grad=_bool_field(d, "model_params_requires_grad"),
            d_embd=_int_field(d, "d_embd", default=D_EMBD),
            d_cortex=_int_field(d, "d_cortex", default=D_CORTEX),
            latent_tokens=_int_field(d, "latent_tokens", default=LATENT_TOKENS),
            max_labels=_int_field(d, "max_labels", "n_labels", default=N_LABELS),
            arm_b_input_mode=_str_field(d, "arm_b_input_mode", default="inputs_embeds"),
            parameter_total=_int_field(d, "parameter_total", "parameter_count", default=TOTAL_PARAMS),
            parameter_budget=_int_field(d, "parameter_budget", default=MAX_PARAMS),
            parameter_breakdown=_dict_str_int_field(d, "parameter_breakdown", default=dict(PARAM_BREAKDOWN)),
            fixed_latent_shape=_list_int_shape_field(d, "fixed_latent_shape", expected=(LATENT_TOKENS, D_EMBD), default=[LATENT_TOKENS, D_EMBD]),
            proposed_confidence_threshold=_float_field(d, "proposed_confidence_threshold", "confidence_threshold", default=CONFIDENCE_THRESHOLD_DEFAULT),
            proposed_specificity_margin=_float_field(d, "proposed_specificity_margin", "specificity_margin", default=SPECIFICITY_MARGIN_DEFAULT),
            cortex_artifact_sha256=_str_field(d, "cortex_artifact_sha256"),
            cortex_artifact_bytes=_int_field(d, "cortex_artifact_bytes"),
            cortex_artifact_path=_str_field(d, "cortex_artifact_path"),
            coupler_sha256=_str_field(d, "coupler_sha256", "coupler_hash"),
            coupler_bytes=_int_field(d, "coupler_bytes"),
            head_sha256=_str_field(d, "head_sha256"),
            head_bytes=_int_field(d, "head_bytes"),
            label_phrasing_frozen=_bool_field(d, "label_phrasing_frozen", default=True),
            labels=_list_str_field(d, "labels", default=[]),
            dev_split=_str_field(d, "dev_split", default="dev"),
            dev_repeatability_std=_float_field(d, "dev_repeatability_std"),
            dev_confidence_mean=_float_field(d, "dev_confidence_mean"),
            dev_confidence_std=_float_field(d, "dev_confidence_std"),
            dev_confidence_min=_float_field(d, "dev_confidence_min"),
            dev_confidence_max=_float_field(d, "dev_confidence_max"),
            dev_specificity_acc=_float_field(d, "dev_specificity_acc"),
            dev_holdout_ids_discarded=_bool_field(d, "dev_holdout_ids_discarded", default=True),
            split_salt=_str_field(d, "split_salt", default="v2.2"),
            split_fraction=_float_field(d, "split_fraction", default=0.3),
            c7_reference=_str_field(d, "c7_reference"),
            c7_available=_bool_field(d, "c7_available"),
            c7_blocked_reason=_optional_str_field(d, "c7_blocked_reason"),
            trace_raw_traces_deleted=_bool_field(d, "trace_raw_traces_deleted"),
            trace_raw_trace_count=_int_field(d, "trace_raw_trace_count"),
            trace_embedding_cache_cleared=_bool_field(d, "trace_embedding_cache_cleared"),
            trace_optimizer_state_deleted=_bool_field(d, "trace_optimizer_state_deleted"),
            articulation_prompt_text=_str_field(d, "articulation_prompt_text"),
            articulation_latent_bank_shape=_list_int_shape_field(d, "articulation_latent_bank_shape", expected=(LATENT_TOKENS, D_EMBD), default=[LATENT_TOKENS, D_EMBD]),
            articulation_raw_trace_count=_int_field(d, "articulation_raw_trace_count"),
            articulation_language_organ_hash=_str_field(d, "articulation_language_organ_hash"),
            articulation_persistent_cortex_bytes=_int_field(d, "articulation_persistent_cortex_bytes"),
            articulation_banned_label_text_absent=_bool_field(d, "articulation_banned_label_text_absent", default=True),
            articulation_banned_corrected_response_absent=_bool_field(d, "articulation_banned_corrected_response_absent", default=True),
            articulation_banned_correction_utterance_absent=_bool_field(d, "articulation_banned_correction_utterance_absent", default=True),
            articulation_banned_expected_answer_absent=_bool_field(d, "articulation_banned_expected_answer_absent", default=True),
            signoff_thresholds_signed_off=_bool_field(d, "signoff_thresholds_signed_off", "thresholds_signed_off"),
            signoff_human_signoff_id=_str_field(d, "signoff_human_signoff_id", "human_signoff_id"),
            signoff_oracle_ceiling=_float_field(d, "signoff_oracle_ceiling", "oracle_ceiling"),
            signoff_dev_articulation_gate=_bool_field(d, "signoff_dev_articulation_gate", "dev_articulation_gate"),
            signoff_meta_test_conflation_ok=_bool_field(d, "signoff_meta_test_conflation_ok", "meta_test_conflation_ok"),
            holdout_accessed=_bool_field(d, "holdout_accessed"),
            manifest_sha256=_str_field(d, "manifest_sha256", "manifest_hash"),
        )

    def verify_hash(self) -> bool:
        """Return True if the stored manifest_sha256 matches a recomputation."""
        return self.to_dict()["manifest_sha256"] == self.manifest_sha256

    def required_fields_present(self) -> bool:
        """Return True if every contract-required field is present and non-default.

        Evaluate calls this to fail closed on incomplete manifests.
        """
        required_nonempty = (
            "schema_version",
            "source_commit",
            "source_archive_sha256",
            "eval_version",
            "eval_manifest_sha256",
            "model_repo_id",
            "model_revision",
            "model_config_sha256",
            "model_safetensors_sha256",
            "cortex_artifact_sha256",
            "coupler_sha256",
            "head_sha256",
            "articulation_language_organ_hash",
            "c7_reference",
        )
        for f in required_nonempty:
            val = getattr(self, f)
            if not val:
                return False
        if self.cortex_artifact_bytes <= 0:
            return False
        if self.coupler_bytes <= 0:
            return False
        if self.head_bytes <= 0:
            return False
        # -- required collection fields (must match schema exactly) ----------
        if not isinstance(self.labels, list) or not self.labels:
            return False
        if not all(isinstance(x, str) for x in self.labels):
            return False
        if not isinstance(self.parameter_breakdown, dict):
            return False
        if not isinstance(self.fixed_latent_shape, list):
            return False
        if list(self.fixed_latent_shape) != [LATENT_TOKENS, D_EMBD]:
            return False
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in self.fixed_latent_shape):
            return False
        if not isinstance(self.articulation_latent_bank_shape, list):
            return False
        if list(self.articulation_latent_bank_shape) != [LATENT_TOKENS, D_EMBD]:
            return False
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in self.articulation_latent_bank_shape):
            return False
        if self.holdout_accessed:
            return False
        return True


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def compute_verdicts(
    c3_results: list[dict[str, Any]],
    c2_results: list[dict[str, Any]],
    c4_results: list[dict[str, Any]],
    c5_results: list[dict[str, Any]],
    c6_results: list[dict[str, Any]],
    c1_results: list[dict[str, Any]],
    specificity_margin: float,
    validity_pass: bool,
) -> dict[str, str]:
    """Compute H-LATENT and H-LABEL verdicts per spec.

    Accept H-LATENT only if ALL of:
      - C3 retention_delta > 0 with 95% CI excluding zero
      - C3 transfer_delta > 0 with 95% CI excluding zero
      - causal_state_delta > 0 and feedback_semantics_delta > 0, each with
        95% CI excluding zero
      - state_addressing_delta > 0 with 95% CI excluding zero
      - specificity within DEV-frozen equivalence margin
      - all validity gates pass

    Accept H-LABEL separately if C2 passes retention, transfer, specificity.
    """
    verdicts: dict[str, str] = {}

    if not validity_pass:
        verdicts["H_LATENT"] = "BLOCKED"
        verdicts["H_LABEL"] = "BLOCKED"
        return verdicts

    # C3 deltas.
    c3_retention = [r["holdout_acc"] - c1["holdout_acc"] for r, c1 in zip(c3_results, c1_results, strict=True)]
    c3_transfer = [r["transfer_acc"] - c1["transfer_acc"] for r, c1 in zip(c3_results, c1_results, strict=True)]
    causal_state = [r["holdout_acc"] - c4["holdout_acc"] for r, c4 in zip(c3_results, c4_results, strict=True)]
    feedback_sem = [r["holdout_acc"] - c6["holdout_acc"] for r, c6 in zip(c3_results, c6_results, strict=True)]
    addressing = [r["holdout_acc"] - c5["holdout_acc"] for r, c5 in zip(c3_results, c5_results, strict=True)]
    c3_specificity = [r["specificity_acc"] - c1["specificity_acc"] for r, c1 in zip(c3_results, c1_results, strict=True)]

    retention_ci = mean_ci(c3_retention)
    transfer_ci = mean_ci(c3_transfer)
    causal_ci = mean_ci(causal_state)
    feedback_ci = mean_ci(feedback_sem)
    addressing_ci = mean_ci(addressing)
    specificity_ci = mean_ci(c3_specificity)

    h_latent_accept = (
        ci_excludes_zero(retention_ci) and retention_ci[0] > 0
        and ci_excludes_zero(transfer_ci) and transfer_ci[0] > 0
        and ci_excludes_zero(causal_ci) and causal_ci[0] > 0
        and ci_excludes_zero(feedback_ci) and feedback_ci[0] > 0
        and ci_excludes_zero(addressing_ci) and addressing_ci[0] > 0
        and abs(specificity_ci[0]) <= specificity_margin
    )
    verdicts["H_LATENT"] = "ACCEPT" if h_latent_accept else "REFUTE"

    # C2 deltas (H-LABEL).
    c2_retention = [r["holdout_acc"] - c1["holdout_acc"] for r, c1 in zip(c2_results, c1_results, strict=True)]
    c2_transfer = [r["transfer_acc"] - c1["transfer_acc"] for r, c1 in zip(c2_results, c1_results, strict=True)]
    c2_specificity = [r["specificity_acc"] - c1["specificity_acc"] for r, c1 in zip(c2_results, c1_results, strict=True)]

    c2_retention_ci = mean_ci(c2_retention)
    c2_transfer_ci = mean_ci(c2_transfer)
    c2_specificity_ci = mean_ci(c2_specificity)

    h_label_accept = (
        ci_excludes_zero(c2_retention_ci) and c2_retention_ci[0] > 0
        and ci_excludes_zero(c2_transfer_ci) and c2_transfer_ci[0] > 0
        and abs(c2_specificity_ci[0]) <= specificity_margin
    )
    verdicts["H_LABEL"] = "ACCEPT" if h_label_accept else "REFUTE"

    return verdicts


# ---------------------------------------------------------------------------
# Audit emission
# ---------------------------------------------------------------------------


def build_articulation_audit(
    condition: str,
    arm: str,
    prompt_text: str,
    latent_bank_shape: tuple[int, ...] | None,
    raw_trace_count: int,
    model_hash: str,
    persistent_bytes: int,
    banned_label_text: str | None = None,
    banned_correction_text: str | None = None,
    banned_answer_text: str | None = None,
    banned_episode_id: str | None = None,
) -> dict[str, Any]:
    """Build a machine-checkable articulation audit record per spec.

    The runner must emit an articulation audit containing:
      - prompt text
      - latent-bank shape
      - raw-trace count
      - language-organ hash
      - persistent cortex bytes
      - banned-content fields (label, correction, answer, episode ID text
        that must never appear in the prompt or latent construction)
    """
    return {
        "condition": condition,
        "arm": arm,
        "prompt_text": prompt_text,
        "latent_bank_shape": latent_bank_shape,
        "raw_trace_count": raw_trace_count,
        "language_organ_hash": model_hash,
        "persistent_cortex_bytes": persistent_bytes,
        "banned_label_text": banned_label_text,
        "banned_correction_text": banned_correction_text,
        "banned_answer_text": banned_answer_text,
        "banned_episode_id": banned_episode_id,
    }


def verify_no_text_injection(audits: list[dict[str, Any]]) -> bool:
    """Verify that no Arm B audit contains label/answer/correction text.

    Checks that Arm B prompts contain only the request text and that no
    label, corrected_response, correction_utterance, or expected answer
    text leaks into the prompt. Also verifies the latent bank shape is
    fixed (latent_tokens, d_embd) or None (abstained).
    """
    # Forbidden text fragments that must never appear in Arm B prompts.
    # These are populated from the audit records themselves.
    forbidden_fragments: set[str] = set()
    for audit in audits:
        for key in ("label_text", "corrected_response", "correction_utterance", "expected", "banned_label_text", "banned_correction_text", "banned_answer_text", "banned_episode_id"):
            val = audit.get(key)
            if val and isinstance(val, str) and len(val.strip()) > 2:
                forbidden_fragments.add(val.strip().lower())

    for audit in audits:
        if audit.get("arm") != "B":
            continue
        # Latent bank shape must be fixed or None (abstained).
        shape = audit.get("latent_bank_shape")
        if shape is not None and shape != (LATENT_TOKENS, D_EMBD):
            return False
        # Content check: no forbidden text in the prompt.
        prompt = audit.get("prompt_text", "").lower()
        for fragment in forbidden_fragments:
            if fragment in prompt:
                return False
    return True


def verify_fixed_latent_width(audits: list[dict[str, Any]]) -> bool:
    """Verify that the latent bank shape is independent of episode count."""
    for audit in audits:
        if audit.get("arm") != "B":
            continue
        shape = audit.get("latent_bank_shape")
        if shape is not None and shape != (LATENT_TOKENS, D_EMBD):
            return False
    return True


def verify_raw_traces_deleted(trace_store: TraceStore) -> bool:
    """Verify that all raw traces have been deleted."""
    return trace_store.verify_zero()


def verify_frozen_lm(hash_before: str, hash_after: str) -> bool:
    """Verify that the LM hash is identical before and after the run."""
    return hash_before == hash_after


def verify_parameter_budget(cortex: SharedCortex) -> bool:
    """Verify that the cortex parameter count is within budget."""
    return cortex.parameter_count() <= MAX_PARAMS


def verify_no_episode_id_conditioning(audits: list[dict[str, Any]]) -> bool:
    """Verify that no episode IDs enter the prompt or latent construction."""
    # Arm B prompts should be only the request text.
    # Episode IDs appear only in audit metadata, not in prompt_text.
    for audit in audits:
        if audit.get("arm") != "B":
            continue
        # The prompt text should not contain episode ID patterns.
        prompt = audit.get("prompt_text", "")
        if "s0_" in prompt or "s1_" in prompt or "s2_" in prompt:
            return False
    return True


def compute_oracle_ceiling(
    driver: HFDriver,
    stage: Stage,
    probe_ids: set[str],
) -> float:
    """Compute the oracle ceiling: accuracy when the corrected_response
    is directly prepended to the probe request as a text prefix.

    This is the upper bound for any text-injection approach and must
    exceed the cortex's performance for the hypothesis to be meaningful.
    """
    correct = 0
    total = 0
    for ep in stage.episodes:
        for probe in ep.probes:
            pid = _probe_id(ep, probe)
            if pid not in probe_ids:
                continue
            prefix = ep.corrected_response + " "
            driver.set_reserved_position(cast(Any, ReservedPosition(text=prefix)))
            try:
                answer = driver.generate(probe.request, max_tokens=MAX_GENERATE_TOKENS)
            finally:
                driver.clear_reserved_position()
            correct += int(probe_matches(answer, probe, ep))
            total += 1
    return correct / total if total else 0.0


def check_dev_articulation_gate(
    driver: HFDriver,
    cortex: SharedCortex,
    stage: Stage,
    dev_ids: set[str],
    labels: list[str],
    confidence_threshold: float,
    label_prefix_template: str,
    trace_store: TraceStore,
) -> bool:
    """Check that Arm B articulation on DEV probes exceeds the no-update
    baseline (C1) by at least a small margin. This is the DEV articulation
    gate: the coupler must produce a measurable improvement on DEV before
    it is worth evaluating on holdout.

    Returns True if the gate passes (Arm B DEV accuracy > C1 DEV accuracy).
    """
    # Arm B on DEV.
    arm_b_result = score_probes(
        driver, stage, dev_ids, "B", cortex, labels,
        confidence_threshold, label_prefix_template, trace_store,
    )
    # C1 (random cortex) on DEV.
    random_cortex = SharedCortex(config=cortex.config, seed=999)
    if cortex.coupler_frozen:
        random_cortex.load_coupler(cortex.coupler_state())
        random_cortex.freeze_coupler()
    c1_result = score_probes(
        driver, stage, dev_ids, "B", random_cortex, labels,
        confidence_threshold, label_prefix_template, trace_store,
    )
    return arm_b_result["accuracy"] > c1_result["accuracy"]


def check_meta_test_conflation(
    c3_results: list[dict[str, Any]],
    c2_results: list[dict[str, Any]],
) -> bool:
    """Verify that C3 (Arm B) and C2 (Arm A) results are not identical
    across all seeds, which would indicate the two arms are conflated
    (e.g., both falling through to the same vanilla path).

    Returns True if at least one seed shows a difference between C3 and C2.
    """
    for c3, c2 in zip(c3_results, c2_results, strict=True):
        if abs(c3["holdout_acc"] - c2["holdout_acc"]) > 1e-6:
            return True
    return False
