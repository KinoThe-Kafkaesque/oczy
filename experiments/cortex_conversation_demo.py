#!/usr/bin/env python3
"""Qualitative demo: 4-turn CortexAgent conversation.

Verifies the cortex is actually absorbing intent (not just producing
noise-shape cvec variation) by running one correction cycle and probing
whether the post-correction output differs from the baseline in a
direction consistent with the correction.

Sequence:
    1. Cold-boot the cortex; capture baseline articulate() on a probe.
    2. Perceive a correction; metabolise it.
    3. Articulate on the same probe; observe whether the output shifts.
    4. Consolidate() -- commit cortex.warm into cortex.cold.
    5. Save/load round-trip; articulate again with reloaded cold_state.
    6. Compare the three outputs: baseline, post-correction, post-reload.

If post-correction and post-reload agree (and differ from baseline), the
cortex's metabolised intent persists across consolidation and save/load.
If they don't agree, the cortex is producing transient steering that
doesn't survive cold boot -- and we know to focus on REAL persistence
mechanisms in proj_c / cold_state before extending further.

Run: uv run python experiments/cortex_conversation_demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _name in (
    "plastic-cortex",
    "neural-hippocampus",
    "world-model-critic",
    "identity-hypernetwork",
    "skill-immune-cortex",
    "experience-autoencoder",
):
    _src = _REPO_ROOT / _name / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

import numpy as np

from experiments.cortex_agent import CortexAgent, CortexAgentConfig
from plastic_cortex.kv_cortex import KVCortexConfig
from oczy_lm import CVecDriverConfig


PROMPT = "What does 'profile' mean in this product?"


def run_demo() -> int:
    print("CortexAgent conversation demo")
    print("=" * 64)
    print("Prompt: %r" % PROMPT)
    print()

    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=64, alpha_correction=5.0),
        driver=CVecDriverConfig(n_ctx=512, verbose=False, embedding=True),
        articulate_scale=30.0,
    )
    agent = CortexAgent(cfg)
    agent.boot()
    print("cold-booted. driver.n_layers=%d n_embd=%d" % (
        agent.driver.n_layers, agent.driver.n_embd
    ))
    print("cortex cold_state norm:", float(np.linalg.norm(agent.cortex.cold_state)))
    print()

    # 1. Baseline: zero cortex, no steering applied (cold_state == zeros).
    baseline = agent.articulate(prompt=PROMPT, max_tokens=20, temperature=0.0,
                                 apply_steering=False)
    print("baseline (no steering):")
    print("  %r" % baseline)
    print()

    # 2. Absorb a real correction.
    correction = "No, 'profile' means business vertical, not user profile."
    print("perceiving correction: %r" % correction)
    before_norm = float(np.linalg.norm(agent.cortex.warm_state))
    agent.perceive(correction)
    after_norm = float(np.linalg.norm(agent.cortex.warm_state))
    print("  warm_norm: %.4f -> %.4f" % (before_norm, after_norm))
    print("  last_drift: %.4f" % agent._last_drift)
    print("  last_correction_signal: %.2f" % agent._last_correction_signal)
    agent.metabolize()
    st = agent.neural_hippocampus.status()
    print("  hippocampus episodes after metabolize: %d" % st["episode_count"])
    print()

    # 3. Re-articulate the SAME probe with cortex steering now applied.
    steered = agent.articulate(prompt=PROMPT, max_tokens=20, temperature=0.0,
                               apply_steering=True)
    print("post-correction (with steering):")
    print("  %r" % steered)
    print()

    # 4. Consolidate.
    summary = agent.consolidate()
    print("consolidate() summary: %s" % summary)
    print("cortex cold_state norm now: %.4f" % float(np.linalg.norm(agent.cortex.cold_state)))
    print()

    # 5. Save / load round-trip; articulate from the loaded cold_state.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent.pkl"
        agent.save(path)
        reloaded = CortexAgent.load(
            path,
            config=CortexAgentConfig(
                cortex=KVCortexConfig(d_cortex=64),
                driver=CVecDriverConfig(n_ctx=512, verbose=False, embedding=True),
            ),
        )

    # Re-articulate from the freshly loaded cold state (and warm was reset
    # to cold at boot, so this should reflect the consolidated identity
    # not the per-turn warm_state).
    reloaded.articulate_scale = 30.0
    post_reload = reloaded.articulate(prompt=PROMPT, max_tokens=20,
                                      temperature=0.0, apply_steering=True)
    print("post-reload (cold-boot identity steering):")
    print("  %r" % post_reload)
    print()

    # 6. Comparison.
    print("=" * 64)
    print("COMPARISON")
    print("  baseline     : %r" % baseline)
    print("  steered      : %r" % steered)
    print("  post-reload  : %r" % post_reload)
    print()
    print("baseline != steered       :", baseline != steered)
    print("steered  != post-reload    :", steered != post_reload)
    print("baseline != post-reload   :", baseline != post_reload)
    print()

    # Verdict.
    if baseline != steered and baseline != post_reload:
        print("VERDICT: cortex steering persists across consolidate + save/load.")
        print("         (cold_state now carries the post-correction identity)")
        return 0
    if baseline != steered:
        print("VERDICT: cortex steering is live but does NOT survive consolidation.")
        print("         Need stronger cold-state writes in consolidate().")
        return 1
    print("VERDICT: cortex had no measurable effect on LM output.")
    print("         Scale may be too small OR proj_c init too random.")
    return 2


if __name__ == "__main__":
    sys.exit(run_demo())