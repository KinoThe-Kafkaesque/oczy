"""R24-local differentiable fast/slow cortex.

The R24 toy experiment deliberately uses a much smaller cortex than R20.  A
single task owns two ``[16, 16]`` float32 matrices: a differentiable fast
matrix and a slow matrix.  Exactly three interaction features are written
before consolidation, and a query-conditioned reader projects the 16-wide
cortex readout to the decoder's 64-wide conditioning vector.

This module is self-contained within R24.  It stores neither task identifiers
nor event history, performs no optimizer step, and never mutates a supplied
state tensor in place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as nnf

FEATURE_DIM = 64
STATE_DIM = 16
READOUT_DIM = 64
EVENT_WRITES = 3
PARAMETER_BUDGET = 10_240
PERSISTENT_BYTES_PER_TASK = STATE_DIM * STATE_DIM * 4
_EPS = 1e-8

__all__ = [
    "FEATURE_DIM",
    "STATE_DIM",
    "READOUT_DIM",
    "EVENT_WRITES",
    "PARAMETER_BUDGET",
    "PERSISTENT_BYTES_PER_TASK",
    "TinyCortexConfig",
    "CortexState",
    "EventFeatureBatch",
    "WriteResult",
    "ThreeWriteResult",
    "ConsolidationResult",
    "TinyCortexV3",
    "TinyCortex",
]


@dataclass(frozen=True, slots=True)
class TinyCortexConfig:
    """Fixed R24-v3 cortex dimensions.

    The fields are explicit for artifact/config reporting, but R24-v3 does not
    allow changing them without creating a new experiment version.
    """

    feature_dim: int = FEATURE_DIM
    state_dim: int = STATE_DIM
    readout_dim: int = READOUT_DIM
    event_writes: int = EVENT_WRITES

    def __post_init__(self) -> None:
        expected = {
            "feature_dim": FEATURE_DIM,
            "state_dim": STATE_DIM,
            "readout_dim": READOUT_DIM,
            "event_writes": EVENT_WRITES,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must be {value} for R24-v3")


@dataclass(frozen=True, slots=True)
class CortexState:
    """Fast and slow cortex matrices, each ``[B, 16, 16]`` float32."""

    fast: torch.Tensor
    slow: torch.Tensor


@dataclass(frozen=True, slots=True)
class EventFeatureBatch:
    """A tensor wrapper used by :meth:`write` or :meth:`write_three`.

    ``write`` requires ``values`` to be ``[B, 64]``.  ``write_three`` requires
    ``[B, 3, 64]``.  Raw tensors are accepted as well so orchestration code
    need not allocate a wrapper.
    """

    values: torch.Tensor


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Diagnostics and new state from one fast-weight write."""

    state: CortexState
    key: torch.Tensor
    value: torch.Tensor
    eta: torch.Tensor
    decay: torch.Tensor


@dataclass(frozen=True, slots=True)
class ThreeWriteResult:
    """Diagnostics and new state after exactly three sequential writes."""

    state: CortexState
    keys: torch.Tensor
    values: torch.Tensor
    etas: torch.Tensor
    decays: torch.Tensor


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """New state and learned gate from one consolidation."""

    state: CortexState
    gate: torch.Tensor


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / (tensor.norm(p=2, dim=-1, keepdim=True) + _EPS)


def _unwrap_features(features: torch.Tensor | EventFeatureBatch) -> torch.Tensor:
    if isinstance(features, EventFeatureBatch):
        return features.values
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a Tensor or EventFeatureBatch")
    return features


def _validate_state(state: CortexState, *, name: str = "state") -> None:
    if not isinstance(state, CortexState):
        raise TypeError(f"{name} must be CortexState")
    for label, tensor in (("fast", state.fast), ("slow", state.slow)):
        if tensor.ndim != 3 or tensor.shape[1:] != (STATE_DIM, STATE_DIM):
            raise ValueError(
                f"{name}.{label} must have shape [B,{STATE_DIM},{STATE_DIM}], "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < 1:
            raise ValueError(f"{name}.{label} batch size must be at least one")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name}.{label} must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}.{label} contains non-finite values")
    if state.fast.shape != state.slow.shape:
        raise ValueError(f"{name}.fast and {name}.slow shapes must match")
    if state.fast.device != state.slow.device:
        raise ValueError(f"{name}.fast and {name}.slow devices must match")
    if state.fast.dtype != state.slow.dtype:
        raise ValueError(f"{name}.fast and {name}.slow dtypes must match")


class TinyCortexV3(nn.Module):
    """A 6,995-parameter differentiable cortex for R24-v3.

    A single write implements the scaled R20 fast-weight equation::

        e = normalize(tanh(feature_projection(x) + event_role))
        f_ctx = F @ e
        h = silu(writer_hidden(concat(e, f_ctx)))
        k = normalize(tanh(writer_key(h)))
        v = tanh(writer_value(h))
        eta = sigmoid(writer_eta(h)); decay = sigmoid(writer_decay(h))
        F_next = decay * F + eta * outer(v, k)

    ``write_three`` applies that equation sequentially to the registered three
    inner-loop events.  Consolidation and reading are fully differentiable.
    """

    def __init__(
        self,
        config: TinyCortexConfig | None = None,
        *,
        init_seed: int = 20260809,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else TinyCortexConfig()
        if not isinstance(self.config, TinyCortexConfig):
            raise TypeError("config must be TinyCortexConfig")
        if isinstance(init_seed, bool) or not isinstance(init_seed, int):
            raise TypeError("init_seed must be an integer")
        if not 0 <= init_seed < 2**63:
            raise ValueError("init_seed must be in [0, 2**63)")
        self.init_seed = init_seed

        # nn.Linear initializes from the process RNG.  A forked RNG scope
        # makes construction deterministic without changing the caller's RNG
        # state.  CUDA RNGs are untouched because all parameters start on CPU.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.feature_projection = nn.Linear(FEATURE_DIM, STATE_DIM)
            # Roles are event and query respectively.
            self.role_embeddings = nn.Parameter(torch.randn(2, STATE_DIM) * 0.02)

            self.writer_hidden = nn.Linear(2 * STATE_DIM, 2 * STATE_DIM)
            self.writer_key = nn.Linear(2 * STATE_DIM, STATE_DIM)
            self.writer_value = nn.Linear(2 * STATE_DIM, STATE_DIM)
            self.writer_eta = nn.Linear(2 * STATE_DIM, 1)
            self.writer_decay = nn.Linear(2 * STATE_DIM, 1)

            self.consolidation_probe = nn.Parameter(torch.randn(STATE_DIM) * 0.02)
            self.consolidation_hidden = nn.Linear(2 * STATE_DIM, STATE_DIM)
            self.consolidation_gate = nn.Linear(STATE_DIM, 1)

            self.reader_hidden = nn.Linear(3 * STATE_DIM, 2 * STATE_DIM)
            self.reader_out = nn.Linear(2 * STATE_DIM, STATE_DIM)
            self.readout_projection = nn.Linear(STATE_DIM, READOUT_DIM)

        if self.parameter_count() > PARAMETER_BUDGET:
            raise RuntimeError("R24-v3 cortex exceeds its registered parameter budget")

    def _parameter_device(self) -> torch.device:
        return self.feature_projection.weight.device

    def _validate_features(
        self,
        features: torch.Tensor,
        *,
        state: CortexState,
        expected_shape: tuple[int, ...],
        name: str,
    ) -> None:
        if features.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(features.shape)}")
        if features.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {features.dtype}")
        if features.device != state.fast.device:
            raise ValueError(
                f"{name} device {features.device} does not match state device {state.fast.device}"
            )
        if features.device != self._parameter_device():
            raise ValueError(
                f"{name} device {features.device} does not match model device "
                f"{self._parameter_device()}"
            )
        if not torch.isfinite(features).all():
            raise ValueError(f"{name} contains non-finite values")

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> CortexState:
        """Return bit-zero fast and slow state without changing any RNG."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be at least one")
        if dtype != torch.float32:
            raise ValueError(f"dtype must be float32, got {dtype}")
        resolved_device = self._parameter_device() if device is None else torch.device(device)
        shape = (batch_size, STATE_DIM, STATE_DIM)
        return CortexState(
            fast=torch.zeros(shape, device=resolved_device, dtype=dtype),
            slow=torch.zeros(shape, device=resolved_device, dtype=dtype),
        )

    def write(
        self,
        state: CortexState,
        event_features: torch.Tensor | EventFeatureBatch,
    ) -> WriteResult:
        """Return one learned fast-weight update, leaving ``state`` untouched."""
        _validate_state(state)
        values = _unwrap_features(event_features)
        batch_size = state.fast.shape[0]
        self._validate_features(
            values,
            state=state,
            expected_shape=(batch_size, FEATURE_DIM),
            name="event_features",
        )

        event = torch.tanh(self.feature_projection(values) + self.role_embeddings[0])
        event = _l2_normalize(event)
        fast_context = torch.bmm(state.fast, event.unsqueeze(-1)).squeeze(-1)
        hidden = nnf.silu(self.writer_hidden(torch.cat((event, fast_context), dim=-1)))
        key = _l2_normalize(torch.tanh(self.writer_key(hidden)))
        value = torch.tanh(self.writer_value(hidden))
        eta = torch.sigmoid(self.writer_eta(hidden))
        decay = torch.sigmoid(self.writer_decay(hidden))
        outer = torch.bmm(value.unsqueeze(-1), key.unsqueeze(1))
        fast_next = decay.view(batch_size, 1, 1) * state.fast + eta.view(
            batch_size, 1, 1
        ) * outer
        next_state = CortexState(fast=fast_next, slow=state.slow.clone())
        return WriteResult(
            state=next_state,
            key=key,
            value=value,
            eta=eta,
            decay=decay,
        )

    def write_three(
        self,
        state: CortexState,
        event_features: torch.Tensor | EventFeatureBatch,
    ) -> ThreeWriteResult:
        """Sequentially apply exactly three event writes.

        ``event_features`` is ``[B, 3, 64]`` in interaction order.  Stacked
        diagnostics keep that same event axis.
        """
        _validate_state(state)
        values = _unwrap_features(event_features)
        batch_size = state.fast.shape[0]
        self._validate_features(
            values,
            state=state,
            expected_shape=(batch_size, EVENT_WRITES, FEATURE_DIM),
            name="event_features",
        )

        current = state
        results: list[WriteResult] = []
        for event_index in range(EVENT_WRITES):
            result = self.write(current, values[:, event_index, :])
            results.append(result)
            current = result.state
        return ThreeWriteResult(
            state=current,
            keys=torch.stack([result.key for result in results], dim=1),
            values=torch.stack([result.value for result in results], dim=1),
            etas=torch.stack([result.eta for result in results], dim=1),
            decays=torch.stack([result.decay for result in results], dim=1),
        )

    def write_events(
        self,
        state: CortexState,
        event_features: torch.Tensor | EventFeatureBatch,
    ) -> ThreeWriteResult:
        """Named alias for the registered exact-three event unroll."""
        return self.write_three(state, event_features)

    def consolidate(self, state: CortexState) -> ConsolidationResult:
        """Blend fast into slow and return an exactly bit-zero fast matrix."""
        _validate_state(state)
        batch_size = state.fast.shape[0]
        probe = _l2_normalize(self.consolidation_probe)
        fast_probe = torch.matmul(state.fast, probe)
        slow_probe = torch.matmul(state.slow, probe)
        hidden = nnf.silu(
            self.consolidation_hidden(torch.cat((fast_probe, slow_probe), dim=-1))
        )
        gate = torch.sigmoid(self.consolidation_gate(hidden))
        gate_matrix = gate.view(batch_size, 1, 1)
        slow_next = (1.0 - gate_matrix) * state.slow + gate_matrix * state.fast
        next_state = CortexState(
            fast=torch.zeros_like(state.fast),
            slow=slow_next,
        )
        return ConsolidationResult(state=next_state, gate=gate)

    def read_state(self, state: CortexState, query_features: torch.Tensor) -> torch.Tensor:
        """Return the query-conditioned 16-wide cortex readout."""
        _validate_state(state)
        batch_size = state.fast.shape[0]
        self._validate_features(
            query_features,
            state=state,
            expected_shape=(batch_size, FEATURE_DIM),
            name="query_features",
        )
        query = torch.tanh(
            self.feature_projection(query_features) + self.role_embeddings[1]
        )
        query = _l2_normalize(query)
        fast_read = torch.bmm(state.fast, query.unsqueeze(-1)).squeeze(-1)
        slow_read = torch.bmm(state.slow, query.unsqueeze(-1)).squeeze(-1)
        hidden = nnf.silu(
            self.reader_hidden(torch.cat((query, fast_read, slow_read), dim=-1))
        )
        return torch.tanh(self.reader_out(hidden))

    def project(self, state_readout: torch.Tensor) -> torch.Tensor:
        """Project a finite ``[B,16]`` readout to decoder conditioning ``r[64]``."""
        if state_readout.ndim != 2 or state_readout.shape[1] != STATE_DIM:
            raise ValueError(
                f"state_readout must have shape [B,{STATE_DIM}], "
                f"got {tuple(state_readout.shape)}"
            )
        if state_readout.shape[0] < 1:
            raise ValueError("state_readout batch size must be at least one")
        if state_readout.dtype != torch.float32:
            raise ValueError(f"state_readout must be float32, got {state_readout.dtype}")
        if state_readout.device != self._parameter_device():
            raise ValueError(
                f"state_readout device {state_readout.device} does not match model device "
                f"{self._parameter_device()}"
            )
        if not torch.isfinite(state_readout).all():
            raise ValueError("state_readout contains non-finite values")
        return torch.tanh(self.readout_projection(state_readout))

    def read(self, state: CortexState, query_features: torch.Tensor) -> torch.Tensor:
        """Read cortex state for a query and return decoder vector ``[B,64]``."""
        return self.project(self.read_state(state, query_features))

    def forward(self, state: CortexState, query_features: torch.Tensor) -> torch.Tensor:
        return self.read(state, query_features)

    def zero_state(self, state: CortexState) -> CortexState:
        """Return new bit-zero F/S tensors; parameters and input are unchanged."""
        _validate_state(state)
        return CortexState(
            fast=torch.zeros_like(state.fast),
            slow=torch.zeros_like(state.slow),
        )

    def swap_state(self, state: CortexState, donor: CortexState) -> CortexState:
        """Return a cloned donor state after validating compatibility."""
        _validate_state(state)
        _validate_state(donor, name="donor")
        if state.fast.shape != donor.fast.shape:
            raise ValueError(
                f"state shape {tuple(state.fast.shape)} does not match donor shape "
                f"{tuple(donor.fast.shape)}"
            )
        if state.fast.device != donor.fast.device:
            raise ValueError(
                f"state device {state.fast.device} does not match donor device "
                f"{donor.fast.device}"
            )
        if state.fast.dtype != donor.fast.dtype:
            raise ValueError(
                f"state dtype {state.fast.dtype} does not match donor dtype "
                f"{donor.fast.dtype}"
            )
        return CortexState(fast=donor.fast.clone(), slow=donor.slow.clone())

    def parameter_breakdown(self) -> dict[str, int]:
        """Return exact trainable parameter counts by architectural group."""
        return {
            "feature_projection": FEATURE_DIM * STATE_DIM + STATE_DIM,
            "role_embeddings": 2 * STATE_DIM,
            "writer_hidden": 2 * STATE_DIM * 2 * STATE_DIM + 2 * STATE_DIM,
            "writer_key": 2 * STATE_DIM * STATE_DIM + STATE_DIM,
            "writer_value": 2 * STATE_DIM * STATE_DIM + STATE_DIM,
            "writer_eta": 2 * STATE_DIM + 1,
            "writer_decay": 2 * STATE_DIM + 1,
            "consolidation_probe": STATE_DIM,
            "consolidation_hidden": 2 * STATE_DIM * STATE_DIM + STATE_DIM,
            "consolidation_gate": STATE_DIM + 1,
            "reader_hidden": 3 * STATE_DIM * 2 * STATE_DIM + 2 * STATE_DIM,
            "reader_out": 2 * STATE_DIM * STATE_DIM + STATE_DIM,
            "readout_projection": STATE_DIM * READOUT_DIM + READOUT_DIM,
        }

    def parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters (6,995)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_hash(self) -> str:
        """Return a deterministic SHA-256 over named parameter values."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            cpu_tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(cpu_tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            digest.update(cpu_tensor.numpy().tobytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def logical_persistent_bytes(self, state: CortexState) -> int:
        """Return logical bytes per task: S only after F becomes bit-zero."""
        _validate_state(state)
        if torch.count_nonzero(state.fast).item() == 0:
            return PERSISTENT_BYTES_PER_TASK
        return 2 * PERSISTENT_BYTES_PER_TASK

    def post_consolidation_bytes_per_task(self, state: CortexState) -> int:
        """Validate a consolidated state and return its 1,024-byte footprint."""
        _validate_state(state)
        if torch.count_nonzero(state.fast).item() != 0:
            raise ValueError("post-consolidation fast state must be bit-zero")
        return PERSISTENT_BYTES_PER_TASK


TinyCortex = TinyCortexV3
