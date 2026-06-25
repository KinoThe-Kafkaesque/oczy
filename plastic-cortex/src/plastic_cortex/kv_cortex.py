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

import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

STATE_VERSION = 1


@dataclass
class KVCortexConfig:
    """Sizes and rates for the KV cortex.

    Defaults match LFM2.5-1.2B-Instruct: d_embd 2048, n_layers 28.
    d_head is no longer used (the cvec surface wants per-layer n_embd-dim
    vectors, not (k, v) KV-slot tensors).
    """

    d_cortex: int = 128  # warm/cold state dimensionality
    d_embd: int = 2048  # input dim from LM hidden state AND output dim per layer
    n_layers: int = 28  # how many per-layer steering cvecs the cortex emits
    seed: int = 0

    # Plasticity. Two scalars today; will become a learned alpha_ij matrix
    # the day the cortex gets differentiable plasticity (experiments.txt #4).
    alpha_warm: float = 0.02  # normal-token update rate
    alpha_correction: float = 5.0  # neuromodulator-gated update rate

    # Consolidation.
    consolidate_replay_threshold: int = 3
    consolidate_slow_step: float = 0.05  # cold = (1-s)*cold + s*warm
    consolidate_replay_step: float = 0.10
    max_consolidation_strength: float = 10.0  # cap on per-call strength multiplier

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
        proj_hidden = self.rng.standard_normal((c.d_cortex, c.d_embd)).astype(np.float32) / np.sqrt(
            c.d_embd
        )
        self.proj_hidden: np.ndarray = proj_hidden

        # Per-layer articulation projectors: warm (d_cortex) -> cvec (d_embd).
        # Each layer gets its own projector over the SAME intent vector.
        # Shape: (n_layers, d_embd, d_cortex). /sqrt(d_cortex) keeps the
        # projected vector bounded around unit scale.
        self.proj_c: np.ndarray = self.rng.standard_normal(
            (c.n_layers, c.d_embd, c.d_cortex)
        ).astype(np.float32) / np.sqrt(c.d_cortex)

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
            np.zeros(c.d_embd, dtype=np.float32) for _ in range(c.n_layers)
        ]
        # Optional articulation projector shared by every layer.
        # Set by ``init_proj_c_from_svd(shared=True)`` so the driver can
        # push one uniform cvec instead of stacking per-layer arrays.
        self.proj_c_shared: np.ndarray | None = None
        # Single flat contiguous buffer for the uniform cvec path.
        self._cvec_payloads_flat: np.ndarray | None = None
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
                "lm_hidden dim %d != config.d_embd %d" % (h.shape[0], self.config.d_embd)
            )

        # Linear blend between the two plasticity regimes. When
        # differentiable plasticity lands, this scalar becomes a learned
        # alpha_ij matrix and correction_signal is its gate.
        plasticity = (
            self.alpha_warm * (1.0 - correction_signal) + self.alpha_correction * correction_signal
        )
        plasticity = float(np.clip(plasticity, 0.0, 1.0))

        # Project hidden -> cortex direction, tanh-bounded so the warm
        # state stays bounded for free.
        delta = np.tanh(self.proj_hidden @ h).astype(np.float32)

        # Exponential moving update: warm is a faded trace of recent deltas.
        self.warm_state = ((1.0 - plasticity) * self.warm_state + plasticity * delta).astype(
            np.float32
        )
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
                "layer_idx %d out of range [0, %d)" % (layer_idx, self.config.n_layers)
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

    def has_uniform_proj_c(self) -> bool:
        """Return True when all layers share a single ``proj_c_shared`` matrix.

        The uniform path lets the driver push one cvec to every layer at
        once, avoiding per-layer stack/concatenation on every articulation.
        """
        return self.proj_c_shared is not None

    def emit_all_cvecs_flat(self) -> np.ndarray:
        """Return a single contiguous flat buffer of all per-layer cvecs.

        Shape is ``(n_layers * n_embd,)`` and can be passed directly to
        ``LlamaCVecDriver.set_cvecs_flat``. Requires that payloads have
        been recomputed; returns ``ValueError`` if no flat buffer is
        available (e.g., ``raw_hidden`` mode or non-shared proj_random
        before recomputation).
        """
        if self._dirty:
            self._recompute_payloads()
        if self._cvec_payloads_flat is None:
            raise ValueError("flat cvec buffer not available in current mode")
        return self._cvec_payloads_flat

    def emit_uniform_cvec(self) -> np.ndarray:
        """Return the single steering vector used for every layer.

        Requires ``has_uniform_proj_c()``; raises ``ValueError`` otherwise.
        The result is the same contiguous buffer stored in
        ``_cvec_payloads_flat``, so the driver can pass it straight through.
        """
        if not self.has_uniform_proj_c():
            raise ValueError(
                "emit_uniform_cvec() requires a shared projector; "
                "use emit_cvec(layer_idx) or emit_all_cvecs() instead"
            )
        if self._dirty:
            self._recompute_payloads()
        return self._cvec_payloads_flat

    def _recompute_payloads(self) -> None:
        """Project warm_state through the active projector(s).

        In uniform/shared mode: one ``proj_c_shared @ warm_state`` vector is
        computed, cached as a single flat contiguous vector, and referenced
        from every per-layer payload slot.  This lets the articulation path
        push one cvec to the driver via ``set_cvec_uniform`` instead of
        concatenating a per-layer stack.

        In ``raw_hidden`` mode: broadcast a single ``last_correction_hidden``
        vector across all layers, scaled by the cortex's overall correction
        magnitude.

        In ``proj_random`` mode (legacy per-layer): einsum
        ``proj_c @ warm_state`` to ``(n_layers, d_embd)``.
        """
        if self.proj_c_shared is not None:
            vec = (self.proj_c_shared @ self.warm_state).astype(np.float32)
            shared_payload = np.ascontiguousarray(vec)
            self._cvec_payloads = [shared_payload for _ in range(self.config.n_layers)]
            self._cvec_payloads_flat = shared_payload
            self._dirty = False
            return

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
                np.ascontiguousarray(v.copy()) for _ in range(self.config.n_layers)
            ]
            self._cvec_payloads_flat = None
            self._dirty = False
            return

        # ``proj_random`` (default): one distinct projector per layer.
        w = self.warm_state  # (d_cortex,)
        projected = np.einsum("lec,c->le", self.proj_c, w).astype(np.float32)  # (n_layers, d_embd)
        self._cvec_payloads = [
            np.ascontiguousarray(projected[i]) for i in range(self.config.n_layers)
        ]
        self._cvec_payloads_flat = np.ascontiguousarray(projected).reshape(-1)
        self._dirty = False

    # Forward passthrough for callers that want flat forward(hidden) -> intent
    def forward(self, lm_hidden: np.ndarray, correction_signal: float = 0.0) -> np.ndarray:
        """observe() + return warm_state. Convenience for the cortex-as-fn view."""
        return self.observe(lm_hidden, correction_signal=correction_signal)

    # ------------------------------------------------------------------
    # Cold path
    def consolidate(
        self,
        replays: list[np.ndarray] | None = None,
        strength: float = 1.0,
    ) -> None:
        """Move warm state into cold state.

        Two effects, both gated:

          1. Slow nudge: cold drifts a small step toward warm (the EMA
             analogue of consolidation). Always runs.

          2. Replay absorption: if `replays` (a list of d_embd vectors the
             hippocampus wants to replay) is supplied AND enough replays
             accumulate, those replays are projected through `proj_hidden`,
             averaged, and absorbed into cold_state with a separate rate.

        The `strength` multiplier scales both update steps for this call,
        capped at ``config.max_consolidation_strength``. Use it to make a
        high-pressure consolidation episode write more aggressively into
        cold state without changing the baseline slow rates.

        This is the only method that writes to cold_state. After it runs,
        the next cold boot starts from the updated value.
        """
        c = self.config
        strength = float(np.clip(strength, 0.0, c.max_consolidation_strength))

        # Replay absorption (only if enough replays crossed the threshold).
        if replays is not None and len(replays) >= c.consolidate_replay_threshold:
            stacked = np.stack(replays, axis=0)  # (R, d_embd)
            deltas = self.proj_hidden @ stacked.T  # (d_cortex, R)
            avg_delta = np.mean(np.tanh(deltas), axis=1).astype(np.float32)
            self.cold_state = (
                self.cold_state
                + np.clip(c.consolidate_replay_step * strength, 0.0, 1.0) * avg_delta
            ).astype(np.float32)

        # Slow EMA nudge.
        effective_slow = np.clip(c.consolidate_slow_step * strength, 0.0, 1.0)
        self.cold_state = (
            (1.0 - effective_slow) * self.cold_state + effective_slow * self.warm_state
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
    # SVD-initialised articulation projector
    # ------------------------------------------------------------------
    def init_proj_c_from_svd(
        self,
        hiddens: np.ndarray,
        shared: bool = True,
    ) -> None:
        """Re-initialise ``proj_c`` / ``proj_c_shared`` from correction SVD.

        WHY: ``proj_random`` init draws cvecs from a random subspace, so
        the cortex's steering direction lives in noise rather than in
        correction-aligned structure. Initialising the projector from a
        real SVD basis makes ``proj_c @ warm_state`` land in the
        correction subspace by construction. Because the projector is part
        of the persisted cortex state, the steering direction survives cold
        boot -- unlike ``raw_hidden`` mode, whose direction lives in the
        transient ``last_correction_hidden`` field that reload discards.

        By default the same SVD basis is stored once in
        ``proj_c_shared`` and reused for every layer (the *uniform* path).
        Callers that need per-layer expressivity can pass ``shared=False``
        to keep the legacy stacked ``proj_c`` of shape
        ``(n_layers, d_embd, d_cortex)``.

        NOTE: ``hiddens`` should come from the same LM and the same
        pooling path (``peek_embedding(last_token_only=False)``) that
        ``perceive()`` feeds to ``observe()``. Layer-L (mid-network)
        hidden extraction (Goal 2) is not a prerequisite for this
        method's contract; final-layer hiddens work because they match
        the cortex's runtime input distribution.

        Args:
            hiddens: ndarray shape ``(N, d_embd)`` or ``(N, ...)`` with
                last dim ``d_embd``. ``N`` should be >= ``d_cortex`` so
                the SVD yields ``d_cortex`` non-degenerate right
                singular vectors.
            shared: when True (default), store one shared projector used
                by every layer; when False, broadcast it into the legacy
                per-layer ``proj_c`` stack.
        """
        h = np.asarray(hiddens, dtype=np.float32).reshape(np.asarray(hiddens).shape[0], -1)
        if h.shape[0] < self.config.d_cortex:
            raise ValueError(
                "need N >= d_cortex for non-degenerate SVD; got N=%d, "
                "d_cortex=%d" % (h.shape[0], self.config.d_cortex)
            )
        if h.shape[1] != self.config.d_embd:
            raise ValueError(
                "hiddens last dim %d != config.d_embd %d" % (h.shape[1], self.config.d_embd)
            )
        centered = h - h.mean(axis=0, keepdims=True)
        # full_matrices=False gives Vt of shape (min(N,d_embd), d_embd).
        # We take the top d_cortex rows -- the leading right singular
        # vectors -- as the projector's basis.
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        basis = Vt[: self.config.d_cortex]  # (d_cortex, d_embd)
        # Slab shape (d_embd, d_cortex): each column is one singular
        # vector, scaled by 1/sqrt(d_cortex) to match proj_random's
        # bound convention so emit_cvec magnitudes are comparable.
        slab = (basis.T / np.sqrt(self.config.d_cortex)).astype(np.float32)
        if shared:
            self.proj_c_shared = slab
            self.proj_c = None
        else:
            self.proj_c = np.stack(
                [slab.copy() for _ in range(self.config.n_layers)],
                axis=0,
            )
            self.proj_c_shared = None
        # No warm_state change: this rewrites the projector only. The
        # next emit_cvec() will regenerate payloads from the current
        # warm_state.
        self._cvec_payloads_flat = None
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
                "lm_hidden dim %d != config.d_embd %d" % (h.shape[0], self.config.d_embd)
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
    def status(self, include_size: bool = False) -> dict[str, Any]:
        """Cross-organ status contract.

        Reports warm/cold norms and drift so the driver and harness can
        observe cortex behaviour without poking numpy arrays directly.
        `serialized_bytes` is only computed when ``include_size=True`` to
        avoid expensive pickle calls in hot loops.
        """
        result = {
            "project": "plastic_cortex.kv",
            "d_cortex": self.config.d_cortex,
            "n_layers": self.config.n_layers,
            "warm_norm": float(np.linalg.norm(self.warm_state)),
            "cold_norm": float(np.linalg.norm(self.cold_state)),
            "warm_cold_drift": float(np.linalg.norm(self.warm_state - self.cold_state)),
            "update_count": self.update_count,
            "correction_count": self.correction_count,
            "consolidate_count": self.consolidate_count,
            "record_count": self.correction_count,
        }
        if include_size:
            result["serialized_bytes"] = len(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        return result

    # Pickle: the projectors (proj_hidden, proj_k, proj_v) ARE the learned
    # state. RNG state is intentionally not serialised: it is deterministic
    # given the seed, and we want loaded cortex to behave predictably.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Migration for pickles saved before the shared/uniform projector
        # work (proj_c_shared, _cvec_payloads_flat) was introduced.
        if "proj_c_shared" not in self.__dict__:
            self.__dict__["proj_c_shared"] = None
        if "_cvec_payloads_flat" not in self.__dict__:
            self.__dict__["_cvec_payloads_flat"] = None
        # Force a payload rebuild on first read; loaded state is cold anyway.
        self.__dict__["_dirty"] = True

    # Legacy pickle save/load are kept for backward compatibility.
    # Prefer ``save_state_dict`` / ``load_state_dict`` for a stable,
    # versioned, non-pickle format.

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

    # ------------------------------------------------------------------
    # Versioned, non-pickle state persistence (preferred over pickle).
    # ------------------------------------------------------------------
    def save_state_dict(self, path: Path | str) -> None:
        """Persist state to ``path/`` as ``manifest.json`` + ``arrays.npz``.

        ``manifest.json`` carries format version, class name, config,
        scalar state, and array metadata.  ``arrays.npz`` contains every
        numpy array in ``self.__dict__`` (including optional shared
        projector buffers and per-layer cvec caches).  This format is
        stable across Python versions and avoids pickle.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        list_counts: dict[str, int] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                list_counts[key] = len(value)
                for idx, arr in enumerate(value):
                    arrays[f"{key}_{idx}"] = arr

        scalars: dict[str, Any] = {}
        array_names = set(arrays.keys())
        for key, value in self.__dict__.items():
            if key in {"config", "rng"}:
                continue
            if key in array_names:
                continue
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                continue
            try:
                json.dumps(value)
                scalars[key] = value
            except (TypeError, ValueError):
                warnings.warn(
                    f"Skipping non-JSON-serializable KVCortex field {key!r} "
                    "in save_state_dict; it will be reconstructed from defaults.",
                    stacklevel=2,
                )

        manifest: dict[str, Any] = {
            "version": STATE_VERSION,
            "class": self.__class__.__name__,
            "config": asdict(self.config),
            "arrays": {
                name: {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                for name, arr in arrays.items()
            },
            "list_counts": list_counts,
            "scalars": scalars,
        }

        np.savez(path / "arrays.npz", **arrays)
        tmp_manifest = path / "manifest.json.tmp"
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_manifest.replace(path / "manifest.json")

    @classmethod
    def load_state_dict(cls, path: Path | str) -> KVCortex:
        """Reconstruct a ``KVCortex`` from its versioned directory.

        Validates version and class, rebuilds from config, restores arrays,
        and applies migration logic for any older format version.
        """
        path = Path(path)
        with (path / "manifest.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)

        if manifest.get("class") != "KVCortex":
            raise ValueError(
                f"Expected class 'KVCortex' in state dict, got {manifest.get('class')!r}"
            )

        version = manifest.get("version")
        if not isinstance(version, int) or version < 1:
            raise ValueError(
                f"KVCortex state dict version must be >= 1, got {version!r}"
            )


        config = KVCortexConfig(**manifest["config"])
        instance = cls(config)

        arrays = np.load(path / "arrays.npz", allow_pickle=False)
        list_buffers: dict[str, list[tuple[int, np.ndarray]]] = {}
        for name, arr in arrays.items():
            arr = np.array(arr)  # copy to a writable ndarray
            if "_" in name:
                base, tail = name.rsplit("_", 1)
                if tail.isdigit():
                    list_buffers.setdefault(base, []).append((int(tail), arr))
                    continue
            setattr(instance, name, arr)

        for base, items in list_buffers.items():
            items.sort(key=lambda x: x[0])
            setattr(instance, base, [arr for _, arr in items])

        for key, value in manifest.get("scalars", {}).items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        instance._migrate_state_after_load(version)
        return instance

    def _migrate_state_after_load(self, loaded_version: int) -> None:
        """__setstate__-style migration for non-pickle state loads.

        Apply transformations needed to bring a state dict saved at
        ``loaded_version`` up to the current ``STATE_VERSION``.
        """
        if loaded_version < STATE_VERSION:
            # No structural migrations needed yet; placeholder branch keeps
            # the migration door open for future versions.
            pass

        # Fields introduced after the pickle-only era.
        if "proj_c_shared" not in self.__dict__:
            self.proj_c_shared = None
        if "_cvec_payloads_flat" not in self.__dict__:
            self._cvec_payloads_flat = None
        # Force payload rebuild; loaded state is conceptually cold.
        self._dirty = True
