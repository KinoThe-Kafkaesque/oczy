"""KV-cortex: small-dim neuromodulator that steers LM forward dynamics.

Contract (binding verified 2026-06-23, see
experiments_logs/2026-06-23_cortex_kv_contract.md):

    cortex state   : vector w of dim d_cortex, split into cold_state
                     (loaded at boot, written by consolidate()) and
                     warm_state (in-memory, mutable per turn).
    perception     : warm_state absorbs a hidden-state delta from the
                     LM's last-token residual each turn.
    articulation   : the LM driver reads emit_cvec(layer_idx) per layer
                     and applies it via llama_set_adapter_cvec. This is
                     the C-side control-vector surface: a per-layer
                     steering vector added to the residual stream at
                     layers [L, L+1). It persists across generate() calls
                     and clears cleanly with NULL data.

Frozen LM, mutable cortex. Per-layer projectors are fixed-random at
init and Hebbian-trainable via train_step(). Consolidation is explicit
until implicit triggers are wired.

Per-turn cost on a 1.2B-param LM with d_cortex=128 is sub-2 ms (one
d_cortex x d_embd matmul + n_layers d_embd x d_cortex projections),
so the cortex can keep up with token streaming.

Shape note: ``proj_c`` has shape ``(n_layers, d_embd, d_cortex)`` and
consumes ~7 MB at the defaults (28 * 2048 * 128 float32). That feels
right for an "intent projector per layer" footprint; can shrink by
sharing one projector across all layers if memory becomes a constraint.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class KVCortexConfig:
    """Sizes and rates for the KV cortex.

    Defaults match LFM2.5-1.2B-Instruct: d_embd 2048, n_layers 28.
    d_head is no longer used (the cvec surface wants per-layer n_embd-dim
    vectors, not (k, v) KV-slot tensors).
    """

    d_cortex: int = 128          # warm/cold state dimensionality
    d_embd: int = 2048           # input dim from LM hidden state AND output dim per layer
    n_layers: int = 28           # how many per-layer steering cvecs the cortex emits
    seed: int = 0

    # Plasticity. Two scalars today; will become a learned alpha_ij matrix
    # the day the cortex gets differentiable plasticity (experiments.txt #4).
    alpha_warm: float = 0.02          # normal-token update rate
    alpha_correction: float = 5.0     # neuromodulator-gated update rate

    # Consolidation.
    consolidate_replay_threshold: int = 3
    consolidate_slow_step: float = 0.05  # cold = (1-s)*cold + s*warm
    consolidate_replay_step: float = 0.10

    # Steering mode. ``proj_random`` (default): per-layer cvecs come from
    # ``proj_c @ warm_state``. ``raw_hidden``: per-layer cvecs come from a
    # single ``last_correction_hidden`` vector broadcast across all layers,
    # scaled by the cortex state's correction-norm component. The latter
    # produces semantically-aligned cvecs at the cost of per-layer
    # expressivity; it's a stage toward SVD-initialised proj_c (see
    # GOALS.md "meaningful cvec steering" sub-goal).
    steering_mode: str = "proj_random"


class KVCortex:
    """Small two-speed neuromodulator whose per-layer cvecs steer the LM.

    Lifecycle:

        boot         warm_state := cold_state.copy()
        per-turn     observe(hidden, correction_signal) -> updated warm_state
        per-layer    emit_cvec(layer_idx) -> ndarray[n_embd]
                     passed to llama_set_adapter_cvec(ctx, cvec, n_embd,
                                                       n_embd, L, L+1)
        explicit     consolidate(replays)      -> mutates cold_state
        reset        reset_warm_from_cold()    -> re-boot mid-session

    The cortex never materialises a label string. Its output is a vector
    per layer (dim n_embd), intended to be applied via the LM's
    ``llama_set_adapter_cvec`` control-vector surface — not written into
    the KV cache directly. The driver shim (``oczy_lm.cvec_driver``) owns
    the ctypes boundary; this cortex emits numpy arrays only.
    """

    def __init__(self, config: KVCortexConfig | None = None) -> None:
        self.config = config or KVCortexConfig()
        c = self.config
        self.rng = np.random.default_rng(c.seed)

        # Two-speed state. Cold survives restart; warm does not.
        self.cold_state: np.ndarray = np.zeros(c.d_cortex, dtype=np.float32)
        self.warm_state: np.ndarray = self.cold_state.copy()

        # Perception projector: hidden (d_embd) -> cortex (d_cortex).
        # Fixed-random init at 1/sqrt(d_embd) scale (matches fast-weight
        # programmer convention). Hebbian-trained later via train_step().
        proj_hidden = self.rng.standard_normal(
            (c.d_cortex, c.d_embd)
        ).astype(np.float32) / np.sqrt(c.d_embd)
        self.proj_hidden: np.ndarray = proj_hidden

        # Per-layer articulation projectors: warm (d_cortex) -> cvec (d_embd).
        # Each layer gets its own projector over the SAME intent vector.
        # Shape: (n_layers, d_embd, d_cortex). /sqrt(d_cortex) keeps the
        # projected vector bounded around unit scale.
        self.proj_c: np.ndarray = (
            self.rng.standard_normal((c.n_layers, c.d_embd, c.d_cortex))
            .astype(np.float32)
            / np.sqrt(c.d_cortex)
        )

        self.alpha_warm: float = c.alpha_warm
        self.alpha_correction: float = c.alpha_correction

        # Counts exposed via status(); also gate consolidation decisions later.
        self.update_count: int = 0
        self.correction_count: int = 0
        self.consolidate_count: int = 0

        # Last hidden absorbed under a high correction_signal. Used when
        # ``config.steering_mode == "raw_hidden"`` so the cvec emitted is
        # aligned with a real LM residual rather than a random projection
        # of the warm_state. See GOALS.md "meaningful cvec steering".
        self.last_correction_hidden: np.ndarray = np.zeros(c.d_embd, dtype=np.float32)

        # Cached per-layer cvecs. Recomputed only when warm_state changes,
        # so emit_cvec() for a steady-state cortex is a tuple lookup.
        self._cvec_payloads: list[np.ndarray] = [
            np.zeros(c.d_embd, dtype=np.float32)
            for _ in range(c.n_layers)
        ]
        self._dirty: bool = True

    # ------------------------------------------------------------------
    # Warm path (called per token / per turn from the LM driver)
    # ------------------------------------------------------------------
    def observe(
        self,
        lm_hidden: np.ndarray,
        correction_signal: float = 0.0,
    ) -> np.ndarray:
        """Absorb one LM hidden state into warm_state.

        Args:
            lm_hidden: ndarray (d_embd,) — the layer-L residual the cortex
                is bound to. The driver is responsible for picking which
                layer's hidden to pass; the cortex does not care.
            correction_signal: scalar in [0, 1]. 0 = ordinary token,
                1 = explicit correction / high surprise. Drives the
                plasticity rate (linear blend between alpha_warm and
                alpha_correction).

        Returns:
            A copy of the updated warm_state (d_cortex,). The driver may
            ignore the return value; it is provided for inspection.
        """
        h = np.asarray(lm_hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.config.d_embd:
            raise ValueError(
                "lm_hidden dim %d != config.d_embd %d"
                % (h.shape[0], self.config.d_embd)
            )

        # Linear blend between the two plasticity regimes. When
        # differentiable plasticity lands, this scalar becomes a learned
        # alpha_ij matrix and correction_signal is its gate.
        plasticity = (
            self.alpha_warm * (1.0 - correction_signal)
            + self.alpha_correction * correction_signal
        )
        plasticity = float(np.clip(plasticity, 0.0, 1.0))

        # Project hidden -> cortex direction, tanh-bounded so the warm
        # state stays bounded for free.
        delta = np.tanh(self.proj_hidden @ h).astype(np.float32)

        # Exponential moving update: warm is a faded trace of recent deltas.
        self.warm_state = (
            (1.0 - plasticity) * self.warm_state + plasticity * delta
        ).astype(np.float32)
        self.update_count += 1
        if correction_signal > 0.5:
            self.correction_count += 1
            # Capture the LM hidden itself for raw_hidden steering mode.
            # The vector is semantically aligned with what the user just
            # said, so emitting it back as a cvec steers the LM toward
            # the same region rather than off-manifold.
            self.last_correction_hidden = h.copy()

        self._dirty = True
        return self.warm_state.copy()

    def emit_cvec(self, layer_idx: int) -> np.ndarray:
        """Return the per-layer steering vector (n_embd,) for layer ``layer_idx``.

        The driver binds this to ``llama_set_adapter_cvec(ctx, cvec_ptr,
        n_embd, n_embd, layer_idx, layer_idx+1)``. The vector represents
        the cortex's current intent projected into that layer's residual
        space — an activation-space bias the LM's forward pass consults
        when computing attention at that depth.

        Cached: returns a view-stable reference without recomputation if
        the warm state has not changed since the last ``observe``.
        """
        if not 0 <= layer_idx < self.config.n_layers:
            raise IndexError(
                "layer_idx %d out of range [0, %d)"
                % (layer_idx, self.config.n_layers)
            )
        if self._dirty:
            self._recompute_payloads()
        return self._cvec_payloads[layer_idx]

    def emit_all_cvecs(self) -> list[np.ndarray]:
        """Return one steering vector per layer, ready for the driver.

        Convenience for CortexAgent: call once after ``observe`` to grab
        every layer's cvec, then feed each to the driver.
        """
        if self._dirty:
            self._recompute_payloads()
        return list(self._cvec_payloads)

    def _recompute_payloads(self) -> None:
        """Project warm_state through every per-layer projector.

        In ``proj_random`` mode: einsum ``proj_c @ warm_state`` to
        (n_layers, d_embd), per-layer contiguous.

        In ``raw_hidden`` mode: broadcast a single
        ``last_correction_hidden`` vector across all layers, scaled by
        the cortex's overall correction magnitude (the L2 norm of warm_state
        projected onto the "correction direction" pi/2 from origin, which
        we approximate with the full warm norm for simplicity). Per-layer
        expressivity is lost in this mode but the cvec is guaranteed
        semantically aligned to a real LM residual.
        """
        if self.config.steering_mode == "raw_hidden":
            # Use the warm_state's norm as the amplitude. This couples the
            # cortex's learned intent magnitude to the cvec's steering
            # strength, while keeping the cvec's DIRECTION equal to a real
            # LM hidden -- so amplitude-varied steering lands in the
            # request-aligned residual basin rather than a random one.
            amp = float(np.linalg.norm(self.warm_state))
            unit_h = self.last_correction_hidden / max(
                float(np.linalg.norm(self.last_correction_hidden)), 1e-6
            )
            v = (unit_h * amp).astype(np.float32)
            self._cvec_payloads = [
                np.ascontiguousarray(v.copy())
                for _ in range(self.config.n_layers)
            ]
            self._dirty = False
            return

        # ``proj_random`` (default).
        w = self.warm_state                            # (d_cortex,)
        projected = np.einsum(
            "lec,c->le", self.proj_c, w
        ).astype(np.float32)                           # (n_layers, d_embd)
        self._cvec_payloads = [
            np.ascontiguousarray(projected[i])
            for i in range(self.config.n_layers)
        ]
        self._dirty = False

    # Forward passthrough for callers that want flat forward(hidden) -> intent
    def forward(self, lm_hidden: np.ndarray, correction_signal: float = 0.0) -> np.ndarray:
        """observe() + return warm_state. Convenience for the cortex-as-fn view."""
        return self.observe(lm_hidden, correction_signal=correction_signal)

    # ------------------------------------------------------------------
    # Cold path
    # ------------------------------------------------------------------
    def consolidate(self, replays: list[np.ndarray] | None = None) -> None:
        """Move warm state into cold state.

        Two effects, both gated:

          1. Slow nudge: cold drifts a small step toward warm (the EMA
             analogue of consolidation). Always runs.

          2. Replay absorption: if `replays` (a list of d_embd vectors the
             hippocampus wants to replay) is supplied AND enough replays
             accumulate, those replays are projected through `proj_hidden`,
             averaged, and absorbed into cold_state with a separate rate.

        This is the only method that writes to cold_state. After it runs,
        the next cold boot starts from the updated value.
        """
        c = self.config

        # Replay absorption (only if enough replays crossed the threshold).
        if replays is not None and len(replays) >= c.consolidate_replay_threshold:
            stacked = np.stack(replays, axis=0)            # (R, d_embd)
            deltas = self.proj_hidden @ stacked.T          # (d_cortex, R)
            avg_delta = np.mean(np.tanh(deltas), axis=1).astype(np.float32)
            self.cold_state = (
                self.cold_state + c.consolidate_replay_step * avg_delta
            ).astype(np.float32)

        # Slow EMA nudge.
        self.cold_state = (
            (1.0 - c.consolidate_slow_step) * self.cold_state
            + c.consolidate_slow_step * self.warm_state
        ).astype(np.float32)

        self.consolidate_count += 1

    def reset_warm_from_cold(self) -> None:
        """Cold boot: warm_state := cold_state.copy().

        Called at session start, after a long idle period, or when the
        cortex is being detached from one LM driver and attached to
        another. The warm state is forgotten; cold state survives.
        """
        self.warm_state = self.cold_state.copy()
        self._dirty = True

    def reset_warm_to_zeros(self) -> None:
        """Erase warm state entirely while keeping cold state.

        Used for mid-session context resets (e.g., topic change) so the
        cortex stops steering toward stale intent but cold identity is
        preserved.
        """
        self.warm_state = np.zeros_like(self.warm_state)
        self._dirty = True

    # ------------------------------------------------------------------
    # Passive Hebbian training of the perception projector
    # ------------------------------------------------------------------
    def train_step(self, lm_hidden: np.ndarray, lr: float = 0.001) -> float:
        """One Hebbian update on `proj_hidden`.

        Reinforces whatever activation pattern cortex produced for the
        observed hidden state. Same family as the ExperienceAutoencoder's
        train_step: rank-1 outer-product update followed by per-row L2
        renormalisation.

        Returns the pre-update cortex signal norm (a proxy for "how much
        this input activated the cortex") so callers can monitor
        convergence.
        """
        h = np.asarray(lm_hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.config.d_embd:
            raise ValueError(
                "lm_hidden dim %d != config.d_embd %d"
                % (h.shape[0], self.config.d_embd)
            )
        signal = self.proj_hidden @ h
        bounded = np.tanh(signal)

        self.proj_hidden += lr * np.outer(bounded, h)
        # Per-row renormalisation keeps the projector stable across
        # thousands of updates without an explicit regulariser.
        norms = np.linalg.norm(self.proj_hidden, axis=1, keepdims=True)
        self.proj_hidden /= np.where(norms == 0, 1.0, norms)
        return float(np.linalg.norm(signal))

    # ------------------------------------------------------------------
    # Introspection / persistence
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Cross-organ status contract.

        Reports warm/cold norms and drift so the driver and harness can
        observe cortex behaviour without poking numpy arrays directly.
        `serialized_bytes` is the canonical pickle size; `record_count`
        tracks corrections absorbed (the meaningful learning signal).
        """
        return {
            "project": "plastic_cortex.kv",
            "d_cortex": self.config.d_cortex,
            "n_layers": self.config.n_layers,
            "warm_norm": float(np.linalg.norm(self.warm_state)),
            "cold_norm": float(np.linalg.norm(self.cold_state)),
            "warm_cold_drift": float(
                np.linalg.norm(self.warm_state - self.cold_state)
            ),
            "update_count": self.update_count,
            "correction_count": self.correction_count,
            "consolidate_count": self.consolidate_count,
            "serialized_bytes": len(
                pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
            ),
            "record_count": self.correction_count,
        }

    # Pickle: the projectors (proj_hidden, proj_k, proj_v) ARE the learned
    # state. RNG state is intentionally not serialised: it is deterministic
    # given the seed, and we want loaded cortex to behave predictably.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path | str) -> "KVCortex":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)