#!/usr/bin/env python3
"""Phase-1 autoresearch harness: KVCortex metabolism-loop compounding drift.

Drives a deterministic KVCortex through K=20 correction + consolidate
cycles using fixed mock hidden vectors (no LM, no network, no disk),
then emits metabolism metrics compatible with the autoresearch parser.

This operationalizes project 05 (`research/05-metabolism-loop-closure.md`):
the question is whether repeated corrections COMPOUND into cold-state drift
or merely OVERWRITE it. Random-walk null at K=20 is ~0.22; target >=0.6.

Emitted lines (the autoresearch parser reads `METRIC <name>=<value>`):

  METRIC compounding_index=<v>        primary:Σ step‖ / Σ‖step‖ in (0, 1]
  METRIC cold_state_final_norm=<v>   cold_state‖ after K cycles
  METRIC cold_norm_slope=<v>           OLS slope ofcold_state‖ over K+1 points
  METRIC replay_branch_fires=<n>     number of cycles where the replay absorption
                                     path actually fired (>=3 replays)

Determinism: KVCortex seeded, mock hiddens are a pure function of a literal
string (no RNG, no time-of-day). The same checkout produces the same numbers
every run.
"""

from __future__ import annotations

import sys

import numpy as np

from plastic_cortex.kv_cortex import KVCortex, KVCortexConfig


K_CYCLES = 20
N_CORRECTION_HIDDENS = 16  # for SVD-init; must be >= d_cortex
CONSOLIDATE_STRENGTH = 10.0  # hit the max_consolidation_strength cap


def _mock_hidden(text: str, n_embd: int) -> np.ndarray:
    """Deterministic sparse 'mock driver' hidden vector.

    Pure function of the literal string -- mirrors `multi_fact_stressor`'s
    `_MockDriver.peek_embedding` so the cortex sees well-separated
    embeddings without any LM dependency.
    """
    idx = sum(ord(c) for c in text) % n_embd
    h = np.zeros(n_embd, dtype=np.float32)
    h[idx] = 1.0
    h[(idx + 1) % n_embd] = float(len(text)) * 0.05
    return h


def run() -> int:
    config = KVCortexConfig(
        d_cortex=8,
        d_embd=64,
        n_layers=4,
        seed=42,
        max_consolidation_strength=CONSOLIDATE_STRENGTH,
    )
    cortex = KVCortex(config)

    # SVD-initialise proj_c from correction-aligned hiddens (commit a748758):
    # this is the proven-good init that puts the steering direction in the
    # correction subspace rather than random noise.
    svd_texts = [f"correction-{i:02d}" for i in range(N_CORRECTION_HIDDENS)]
    svd_hiddens = np.stack(
        [_mock_hidden(t, config.d_embd) for t in svd_texts], axis=0
    )
    cortex.init_proj_c_from_svd(svd_hiddens, shared=True)

    # Three distinct replay vectors so the additive replay branch fires on
    # every cycle (consolidate_replay_threshold == 3 by default).
    replays = [
        _mock_hidden("concept-A-replay", config.d_embd),
        _mock_hidden("concept-B-replay", config.d_embd),
        _mock_hidden("concept-C-replay", config.d_embd),
    ]

    cold_norms: list[float] = [float(np.linalg.norm(cortex.cold_state))]
    step_norms: list[float] = []
    replay_fires = 0

    for k in range(K_CYCLES):
        # Per-cycle correction hidden mirrors the live CortexAgent flow:
        # observe(correction_hidden, correction_signal=1.0) -> consolidate.
        h = _mock_hidden(f"correction-cycle-{k:02d}", config.d_embd)
        cortex.observe(h, correction_signal=1.0)
        cold_before = cortex.cold_state.copy()
        cortex.consolidate(replays=replays, strength=CONSOLIDATE_STRENGTH)
        delta = cortex.cold_state - cold_before
        step_norms.append(float(np.linalg.norm(delta)))
        cold_norms.append(float(np.linalg.norm(cortex.cold_state)))
        if len(replays) >= config.consolidate_replay_threshold:
            replay_fires += 1

    final_cold_norm = cold_norms[-1]
    sum_step_norms = float(sum(step_norms))
    compounding_index = (
        final_cold_norm / sum_step_norms if sum_step_norms > 0 else 0.0
    )
    ks = np.arange(len(cold_norms), dtype=np.float64)
    cold_norm_slope = float(
        np.polyfit(ks, np.asarray(cold_norms, dtype=np.float64), 1)[0]
    )

    print(f"METRIC compounding_index={compounding_index:.6f}")
    print(f"METRIC cold_state_final_norm={final_cold_norm:.6f}")
    print(f"METRIC cold_norm_slope={cold_norm_slope:.6f}")
    print(f"METRIC replay_branch_fires={replay_fires}")
    print(
        f"# initial_cold_norm={cold_norms[0]:.6f} "
        f"final_cold_norm={final_cold_norm:.6f} "
        f"sum_step_norms={sum_step_norms:.6f} "
        f"cycles={K_CYCLES} replay_threshold={config.consolidate_replay_threshold}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(run())