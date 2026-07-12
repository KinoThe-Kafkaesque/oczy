"""Differentiable fast-weight cortex for Research/20 (DEV).

Implements the exact write/consolidate/read/couple equations specified in
``experiments/09-meta-trained-cortex-frozen-language-organ/README.md`` and
the DEV plan at ``agent://PlanR20Dev``.

This module owns **only** the differentiable cortex math. It imports PyTorch
and the shared contracts module — nothing else. No driver, task generator,
optimizer, artifact writer, or CLI is imported here.

Design rules enforced:
- F/S are exactly ``[B, 64, 64]``.
- Every transition returns **new** tensors (no in-place state mutation).
- No ``.data``, ``detach``, optimizer, ``backward``, ``train_step``,
  task/event storage, cache, save method, or task-ID argument.
- Local seeded ``torch.Generator`` for initialization; the process-global
  RNG is never mutated.
- All developmental parameter groups retain gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from oczy.experiments.meta_cortex.contracts import (
    CORTEX_DIM,
    ModelConfig,
)

__all__ = [
    "CortexState",
    "EventFeatureBatch",
    "WriteResult",
    "ConsolidationResult",
    "MetaCortex",
]

# ---------------------------------------------------------------------------
# Runtime types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CortexState:
    """Cortex fast/slow state, both ``[B, 64, 64]``.

    ``fast`` (F) is reset at task start and cleared after consolidation.
    ``slow`` (S) is persistent after consolidation.
    """

    fast: torch.Tensor
    slow: torch.Tensor


@dataclass(frozen=True, slots=True)
class EventFeatureBatch:
    """Per-event feature tensor ``[B, 4, D]`` in role order.

    Role order is: observation, attempt, correction, outcome.
    """

    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Result of a single writer step."""

    state: CortexState
    key: torch.Tensor
    value: torch.Tensor
    eta: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Result of a single consolidation step."""

    state: CortexState
    gate: torch.Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8


def _l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2-normalize along *dim* with epsilon for numerical stability."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + _EPS)


def _validate_state(state: CortexState, *, name: str = "state") -> None:
    """Validate F/S shapes, dtypes, and finiteness."""
    for label, tensor in (("fast", state.fast), ("slow", state.slow)):
        if tensor.ndim != 3:
            raise ValueError(
                f"{name}.{label} must be 3-D [B,64,64], got shape {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != CORTEX_DIM or tensor.shape[2] != CORTEX_DIM:
            raise ValueError(
                f"{name}.{label} must be [B,{CORTEX_DIM},{CORTEX_DIM}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"{name}.{label} must be float32, got {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}.{label} contains nonfinite values")


def _validate_feature_tensor(
    values: torch.Tensor,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    feature_dim: int,
) -> None:
    """Validate a feature tensor's shape, dtype, and finiteness."""
    if values.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(values.shape)}"
        )
    if values.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {values.dtype}")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains nonfinite values")
    if values.shape[-1] != feature_dim:
        raise ValueError(
            f"{name} feature dim {values.shape[-1]} != model feature_dim {feature_dim}"
        )


# ---------------------------------------------------------------------------
# MetaCortex
# ---------------------------------------------------------------------------


class MetaCortex(nn.Module):
    """Differentiable fast-weight cortex (Research/20 DEV).

    Parameters (all trainable, all float32)::

        feature_projection   Linear(D, C)
        role_embeddings       Parameter[5, C]   (obs/attempt/correction/outcome/query)
        event_fusion          Linear(4C, C)
        writer_hidden         Linear(2C, 2C)
        writer_key            Linear(2C, C)
        writer_value          Linear(2C, C)
        writer_eta            Linear(2C, 1)
        writer_decay          Linear(2C, 1)
        consolidation_probe   Parameter[C]
        consolidation_hidden  Linear(2C, C)
        consolidation_gate    Linear(C, 1)
        reader_hidden         Linear(3C, 2C)
        reader_out            Linear(2C, C)
        slot_embeddings       Parameter[L, C]
        layer_norm            LayerNorm(C)
        coupler_output        Linear(C, D)       (shared across slots)
        coupler_log_gain      Parameter[1]

    Total: ``N(D,L,C) = 2*C*D + D + 22*C^2 + 23*C + L*C + 4``.
    At C=64: ``129*D + 64*L + 91_588``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        D = config.feature_dim
        C = config.d_cortex
        L = config.bank_width

        # Local seeded generator for Parameter tensors — never touches the
        # global RNG.  ``nn.Linear`` layers use the global RNG internally,
        # so we fork it (save/restore) to avoid mutating the process state.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(20260709)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260709)

            # 1. Shared feature projection and role embeddings.
            self.feature_projection = nn.Linear(D, C)
            # Role constants: observation, attempt, correction, outcome, query.
            self.role_embeddings = nn.Parameter(
                torch.randn(5, C, generator=gen) * 0.02
            )

            # 2. Event fusion: 4 role-projected features -> fused event vector.
            self.event_fusion = nn.Linear(4 * C, C)

            # 3. Writer core + heads.
            self.writer_hidden = nn.Linear(2 * C, 2 * C)
            self.writer_key = nn.Linear(2 * C, C)
            self.writer_value = nn.Linear(2 * C, C)
            self.writer_eta = nn.Linear(2 * C, 1)
            self.writer_decay = nn.Linear(2 * C, 1)

            # 4. Consolidator.
            self.consolidation_probe = nn.Parameter(torch.randn(C, generator=gen) * 0.02)
            self.consolidation_hidden = nn.Linear(2 * C, C)
            self.consolidation_gate = nn.Linear(C, 1)

            # 5. Reader.
            self.reader_hidden = nn.Linear(3 * C, 2 * C)
            self.reader_out = nn.Linear(2 * C, C)

            # 6. Coupler.
            self.slot_embeddings = nn.Parameter(
                torch.randn(L, C, generator=gen) * 0.02
            )
            self.layer_norm = nn.LayerNorm(C)
            self.coupler_output = nn.Linear(C, D)
            self.coupler_log_gain = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> CortexState:
        """Return a zero-initialized cortex state ``[B, 64, 64]``."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if dtype != torch.float32:
            raise ValueError(f"dtype must be float32, got {dtype}")
        shape = (batch_size, CORTEX_DIM, CORTEX_DIM)
        fast = torch.zeros(shape, device=device, dtype=dtype)
        slow = torch.zeros(shape, device=device, dtype=dtype)
        return CortexState(fast=fast, slow=slow)

    def write(self, state: CortexState, event_features: EventFeatureBatch) -> WriteResult:
        """Apply one learned writer step.

        Implements::

            Z = tanh(feature_projection(X) + role_embeddings[0:4])       [B,4,64]
            e = tanh(event_fusion(flatten(Z)))                           [B,64]
            f_ctx = F @ l2_normalize(e)                                  [B,64]
            h = silu(writer_hidden(concat(e, f_ctx)))                    [B,128]
            k = l2_normalize(tanh(key(h)))                               [B,64]
            v = tanh(value(h))                                           [B,64]
            eta = sigmoid(eta_head(h)).view(B,1,1)                       [B,1,1]
            lambda = sigmoid(decay_head(h)).view(B,1,1)                  [B,1,1]
            F_next = lambda * F + eta * (v[..., :, None] @ k[..., None, :]) [B,64,64]
            S_next = S                                                   [B,64,64]
        """
        _validate_state(state, name="state")
        D = self.config.feature_dim
        C = self.config.d_cortex
        values = event_features.values
        if values.ndim != 3:
            raise ValueError(
                f"event_features must be 3-D [B,4,D], got {tuple(values.shape)}"
            )
        B = values.shape[0]
        _validate_feature_tensor(
            values,
            expected_shape=(B, 4, D),
            name="event_features",
            feature_dim=D,
        )
        # Device/dtype consistency.
        if values.device != state.fast.device:
            raise ValueError(
                f"event_features device {values.device} != state device {state.fast.device}"
            )
        if values.dtype != state.fast.dtype:
            raise ValueError(
                f"event_features dtype {values.dtype} != state dtype {state.fast.dtype}"
            )
        if B != state.fast.shape[0]:
            raise ValueError(
                f"event_features batch {B} != state batch {state.fast.shape[0]}"
            )

        F_mat = state.fast  # [B,64,64]
        S_mat = state.slow  # [B,64,64]

        # Z = tanh(feature_projection(X) + role_embeddings[0:4])  [B,4,64]
        proj = self.feature_projection(values)  # [B,4,64]
        roles = self.role_embeddings[:4]  # [4,64]
        Z = torch.tanh(proj + roles.unsqueeze(0))  # [B,4,64]

        # e = tanh(event_fusion(flatten(Z)))  [B,64]
        Z_flat = Z.reshape(B, 4 * C)  # [B, 4*64]
        e = torch.tanh(self.event_fusion(Z_flat))  # [B,64]

        # f_ctx = F @ l2_normalize(e)  [B,64]
        e_norm = _l2_normalize(e, dim=-1)  # [B,64]
        f_ctx = torch.bmm(F_mat, e_norm.unsqueeze(-1)).squeeze(-1)  # [B,64]

        # h = silu(writer_hidden(concat(e, f_ctx)))  [B,128]
        h_in = torch.cat([e, f_ctx], dim=-1)  # [B,128]
        h = F.silu(self.writer_hidden(h_in))  # [B,128]

        # k = l2_normalize(tanh(key(h)))  [B,64]
        k = _l2_normalize(torch.tanh(self.writer_key(h)), dim=-1)  # [B,64]
        # v = tanh(value(h))  [B,64]
        v = torch.tanh(self.writer_value(h))  # [B,64]
        # eta = sigmoid(eta_head(h))  [B,1]
        eta_scalar = torch.sigmoid(self.writer_eta(h))  # [B,1]
        # lambda = sigmoid(decay_head(h))  [B,1]
        decay_scalar = torch.sigmoid(self.writer_decay(h))  # [B,1]

        eta = eta_scalar.view(B, 1, 1)  # [B,1,1]
        lam = decay_scalar.view(B, 1, 1)  # [B,1,1]

        # outer(v, k) = v[..., :, None] @ k[..., None, :]  [B,64,64]
        outer = torch.bmm(v.unsqueeze(-1), k.unsqueeze(1))  # [B,64,64]
        # F_next = lambda * F + eta * outer(v, k)
        F_next = lam * F_mat + eta * outer  # [B,64,64]
        # S unchanged (but return a new state object).
        S_next = S_mat

        new_state = CortexState(fast=F_next, slow=S_next)
        return WriteResult(
            state=new_state,
            key=k,
            value=v,
            eta=eta_scalar,
            decay=decay_scalar,
        )

    def consolidate(self, state: CortexState) -> ConsolidationResult:
        """Apply one learned consolidation step.

        Implements::

            p = l2_normalize(consolidation_probe)                     [64]
            f_probe = F @ p; s_probe = S @ p                          [B,64] each
            h_g = silu(consolidation_hidden(concat(f_probe, s_probe))) [B,64]
            g = sigmoid(consolidation_gate(h_g)).view(B,1,1)          [B,1,1]
            S_next = (1-g) * S + g * F                                [B,64,64]
            F_next = zeros_like(F)                                    [B,64,64]
        """
        _validate_state(state, name="state")
        F_mat = state.fast  # [B,64,64]
        S_mat = state.slow  # [B,64,64]
        B = F_mat.shape[0]

        # p = l2_normalize(consolidation_probe)  [64]
        p = _l2_normalize(self.consolidation_probe, dim=-1)  # [64]

        # f_probe = F @ p  [B,64]  (matmul broadcasts [B,64,64] @ [64] -> [B,64])
        f_probe = torch.matmul(F_mat, p)  # [B,64]
        # s_probe = S @ p  [B,64]
        s_probe = torch.matmul(S_mat, p)  # [B,64]

        # h_g = silu(consolidation_hidden(concat(f_probe, s_probe)))  [B,64]
        h_g_in = torch.cat([f_probe, s_probe], dim=-1)  # [B,128]
        h_g = F.silu(self.consolidation_hidden(h_g_in))  # [B,64]

        # g = sigmoid(consolidation_gate(h_g))  [B,1]
        g_scalar = torch.sigmoid(self.consolidation_gate(h_g))  # [B,1]
        g = g_scalar.view(B, 1, 1)  # [B,1,1]

        # S_next = (1-g) * S + g * F
        S_next = (1.0 - g) * S_mat + g * F_mat  # [B,64,64]
        # F_next = zeros_like(F)
        F_next = torch.zeros_like(F_mat)  # [B,64,64]

        new_state = CortexState(fast=F_next, slow=S_next)
        return ConsolidationResult(state=new_state, gate=g_scalar)

    def read(self, state: CortexState, query_features: torch.Tensor) -> torch.Tensor:
        """Query-conditioned read from F and S.

        Implements::

            q = l2_normalize(tanh(feature_projection(x_q) + query_role)) [B,64]
            r_f = F @ q; r_s = S @ q                                    [B,64] each
            r = tanh(reader_out(silu(reader_hidden(concat(q,r_f,r_s))))) [B,64]
        """
        _validate_state(state, name="state")
        D = self.config.feature_dim
        if query_features.ndim != 2:
            raise ValueError(
                f"query_features must be 2-D [B,D], got {tuple(query_features.shape)}"
            )
        B_q = query_features.shape[0]
        _validate_feature_tensor(
            query_features,
            expected_shape=(B_q, D),
            name="query_features",
            feature_dim=D,
        )
        if query_features.device != state.fast.device:
            raise ValueError(
                f"query_features device {query_features.device} != state device {state.fast.device}"
            )
        if query_features.dtype != state.fast.dtype:
            raise ValueError(
                f"query_features dtype {query_features.dtype} != state dtype {state.fast.dtype}"
            )
        if B_q != state.fast.shape[0]:
            raise ValueError(
                f"query_features batch {B_q} != state batch {state.fast.shape[0]}"
            )

        F_mat = state.fast  # [B,64,64]
        S_mat = state.slow  # [B,64,64]

        # q = l2_normalize(tanh(feature_projection(x_q) + query_role))  [B,64]
        query_role = self.role_embeddings[4]  # [64]
        q = torch.tanh(self.feature_projection(query_features) + query_role.unsqueeze(0))  # [B,64]
        q = _l2_normalize(q, dim=-1)  # [B,64]

        # r_f = F @ q  [B,64]
        r_f = torch.bmm(F_mat, q.unsqueeze(-1)).squeeze(-1)  # [B,64]
        # r_s = S @ q  [B,64]
        r_s = torch.bmm(S_mat, q.unsqueeze(-1)).squeeze(-1)  # [B,64]

        # r = tanh(reader_out(silu(reader_hidden(concat(q,r_f,r_s)))))  [B,64]
        r_in = torch.cat([q, r_f, r_s], dim=-1)  # [B,192]
        r = torch.tanh(self.reader_out(F.silu(self.reader_hidden(r_in))))  # [B,64]
        return r

    def couple(self, readout: torch.Tensor) -> torch.Tensor:
        """Map a readout to a fixed-width soft bank ``[B, L, D]``.

        Implements::

            slot_h = tanh(LayerNorm(r[:,None,:] + slot_embeddings))     [B,L,64]
            soft_bank = softplus(coupler_log_gain) * coupler_output(slot_h) [B,L,D]
        """
        if readout.ndim != 2:
            raise ValueError(
                f"readout must be 2-D [B,64], got {tuple(readout.shape)}"
            )
        if readout.shape[1] != CORTEX_DIM:
            raise ValueError(
                f"readout dim {readout.shape[1]} != cortex dim {CORTEX_DIM}"
            )
        if readout.dtype != torch.float32:
            raise ValueError(f"readout must be float32, got {readout.dtype}")
        if not torch.isfinite(readout).all():
            raise ValueError("readout contains nonfinite values")

        # slot_h = tanh(LayerNorm(r[:,None,:] + slot_embeddings))  [B,L,64]
        r_expanded = readout.unsqueeze(1)  # [B,1,64]
        slot_h = torch.tanh(self.layer_norm(r_expanded + self.slot_embeddings.unsqueeze(0)))  # [B,L,64]

        # soft_bank = softplus(coupler_log_gain) * coupler_output(slot_h)  [B,L,D]
        gain = F.softplus(self.coupler_log_gain)  # scalar
        soft_bank = gain * self.coupler_output(slot_h)  # [B,L,D]
        return soft_bank

    def zero_state(self, state: CortexState) -> CortexState:
        """Return a new state with F and S set to zero (same shape/device/dtype)."""
        _validate_state(state, name="state")
        return CortexState(
            fast=torch.zeros_like(state.fast),
            slow=torch.zeros_like(state.slow),
        )

    def swap_state(self, state: CortexState, donor: CortexState) -> CortexState:
        """Return a new state with F/S taken from *donor*.

        Validates that donor shapes, device, and dtype match *state*.
        Copies state only — never parameters.
        """
        _validate_state(state, name="state")
        _validate_state(donor, name="donor")
        if state.fast.shape != donor.fast.shape:
            raise ValueError(
                f"state shape {tuple(state.fast.shape)} != donor shape {tuple(donor.fast.shape)}"
            )
        if state.fast.device != donor.fast.device:
            raise ValueError(
                f"state device {state.fast.device} != donor device {donor.fast.device}"
            )
        if state.fast.dtype != donor.fast.dtype:
            raise ValueError(
                f"state dtype {state.fast.dtype} != donor dtype {donor.fast.dtype}"
            )
        return CortexState(
            fast=donor.fast.clone(),
            slow=donor.slow.clone(),
        )

    def parameter_breakdown(self) -> dict[str, int]:
        """Return exact per-component parameter counts.

        Total: ``N(D,L,C) = 2*C*D + D + 22*C^2 + 23*C + L*C + 4``.
        At C=64: ``129*D + 64*L + 91_588``.
        """
        D = self.config.feature_dim
        C = self.config.d_cortex
        L = self.config.bank_width
        return {
            # feature_projection: Linear(D, C) — weight + bias
            "feature_projection": D * C + C,
            # role_embeddings: Parameter[5, C]
            "role_embeddings": 5 * C,
            # event_fusion: Linear(4C, C) — weight + bias
            "event_fusion": 4 * C * C + C,
            # writer_hidden: Linear(2C, 2C) — weight + bias
            "writer_hidden": 2 * C * 2 * C + 2 * C,
            # writer_key: Linear(2C, C) — weight + bias
            "writer_key": 2 * C * C + C,
            # writer_value: Linear(2C, C) — weight + bias
            "writer_value": 2 * C * C + C,
            # writer_eta: Linear(2C, 1) — weight + bias
            "writer_eta": 2 * C * 1 + 1,
            # writer_decay: Linear(2C, 1) — weight + bias
            "writer_decay": 2 * C * 1 + 1,
            # consolidation_probe: Parameter[C]
            "consolidation_probe": C,
            # consolidation_hidden: Linear(2C, C) — weight + bias
            "consolidation_hidden": 2 * C * C + C,
            # consolidation_gate: Linear(C, 1) — weight + bias
            "consolidation_gate": C * 1 + 1,
            # reader_hidden: Linear(3C, 2C) — weight + bias
            "reader_hidden": 3 * C * 2 * C + 2 * C,
            # reader_out: Linear(2C, C) — weight + bias
            "reader_out": 2 * C * C + C,
            # slot_embeddings: Parameter[L, C]
            "slot_embeddings": L * C,
            # layer_norm: LayerNorm(C) — weight + bias
            "layer_norm": 2 * C,
            # coupler_output: Linear(C, D) — weight + bias
            "coupler_output": C * D + D,
            # coupler_log_gain: Parameter[1]
            "coupler_log_gain": 1,
        }

    def parameter_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(self.parameter_breakdown().values())

    def logical_persistent_bytes(self, state: CortexState) -> int:
        """Return logical persistent bytes for *state*.

        Before consolidation: F + S = ``2 * 64 * 64 * 4 = 32_768``.
        After consolidation (F must be zero): S only = ``64 * 64 * 4 = 16_384``.
        """
        _validate_state(state, name="state")
        bytes_per_element = 4  # float32
        s_bytes = CORTEX_DIM * CORTEX_DIM * bytes_per_element
        # If F is exactly zero, only S is persistent.
        if torch.count_nonzero(state.fast).item() == 0:
            return s_bytes
        return 2 * s_bytes
