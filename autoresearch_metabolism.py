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

    # Each cycle draws 3 replays from a pool of N_CONCEPTS that ROTATES
    # per cycle (sliding window of 3 concepts advancing by 1 each cycle).
    # This mirrors how CortexAgent.consolidate() pulls representative_hidden
    # from the hippocampus: the replay bank slides as new turns flow in,
    # so replay_avg is correlated-but-varying per cycle -- not the same
    # direction every cycle.
    #
    # Segment 4 generalization stress-test: replays are drawn STOCHASTICALLY
    # per cycle via a deterministic LCG, not via a sliding window. Cycles k
    # and k+1 may share 0, 1, 2, or 3 of their 3 replay concepts (probability
    # of full disjoint is 5/8 * 4/8 * 3/8 = 15/512 ~= 3%). This is a
    # fundamentally different correlation structure from segments 2-3
    # (sliding window guarantees 2-of-3 overlap).
    #
    # Purpose: H1 (skip slow EMA when replay fires) + H2 (Hebbian train_step
    # on each replay) was tuned on segment 2 (3-of-5 window) and verified on
    # segment 3 (3-of-8 window). Both segments have the sliding-window
    # structure in common. This segment removes that structure to test
    # whether the fixes depend on it.
    #
    # Pre-registered plan: run baseline only. NOT iterating cortex-side
    # changes on this segment regardless of result -- that would be the
    # gaming trap the playbook names. Either H1+H2 holds above target
    # (stronger conclusion) or it doesn't (measured generalization limit,
    # reported honestly).
    n_concepts = 8
    concept_texts = [f"concept-{i:02d}" for i in range(n_concepts)]
    concept_hiddens = [
        _mock_hidden(t, config.d_embd) for t in concept_texts
    ]
    # Deterministic LCG (Numerical Recipes constants) so the same checkout
    # produces the same per-cycle replay picks every run. Seed fixed.
    lcg_state = 0x12345678

    cold_norms: list[float] = [float(np.linalg.norm(cortex.cold_state))]
    step_norms: list[float] = []
    replay_fires = 0

    for k in range(K_CYCLES):
        # Per-cycle correction hidden mirrors the live CortexAgent flow:
        # observe(correction_hidden, correction_signal=1.0) -> consolidate.
        h = _mock_hidden(f"correction-cycle-{k:02d}", config.d_embd)
        cortex.observe(h, correction_signal=1.0)
        # Segment 4: stochastic 3-of-8 replay draw via LCG. No guaranteed
        # overlap with the previous cycle's replays. Picks are sampled
        # without replacement from the 8-element pool to preserve the
        # 3-replay threshold contract.
        picked: list[int] = []
        while len(picked) < 3:
            lcg_state = (1664525 * lcg_state + 1013904223) & 0xFFFFFFFF
            idx = lcg_state % n_concepts
            if idx not in picked:
                picked.append(idx)
        replays = [concept_hiddens[idx] for idx in picked]
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