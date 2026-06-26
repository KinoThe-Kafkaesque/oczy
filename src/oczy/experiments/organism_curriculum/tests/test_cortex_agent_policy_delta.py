"""Regression test for real CortexAgent policy head in curriculum runner."""

from __future__ import annotations

import warnings

import numpy as np

from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
from oczy.experiments.organism import OrganismAgent
from oczy.experiments.organism_curriculum.dataset import build_curriculum
from oczy.experiments.organism_curriculum.run_curriculum import (
    _MockDriver,
    run_stage,
)
from oczy.experiments.organism_curriculum.tests.test_shim_policy_delta import (
    _corrected_key,
)
from plastic_cortex.kv_cortex import KVCortexConfig


def test_cortex_agent_policy_margin_delta_positive() -> None:
    """A real CortexAgent with a mock driver should improve policy margin."""
    stages = build_curriculum(stage_names=("stage_0_grounding",))
    assert stages
    stage = stages[0]

    config = {
        "use_cortex_policy": True,
        "use_value_baseline": True,
        # Disable acceptance reward so the test isolates the symmetric
        # correction policy signal; otherwise the first (wrong) answer is
        # reinforced before the correction can penalise it.
        "use_acceptance_policy_reward": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        agent = OrganismAgent(config)

    driver = _MockDriver(n_embd=8, n_layers=2)
    cfg = CortexAgentConfig(
        cortex=KVCortexConfig(d_cortex=4),
        use_policy_head=True,
        policy_learning_rate=0.001,
    )
    cortex = CortexAgent(cfg, driver=driver)
    cortex.boot()


    # Zero-initialize the policy head so updates consistently increase the
    # corrected-vs-wrong margin without random-initialization noise.
    cortex._policy_W = np.zeros(
        cfg.cortex.d_cortex + driver.n_embd, dtype=np.float64
    )
    cortex._policy_b = 0.0

    agent.cortex_agent = cortex

    result = run_stage(agent, stage, adapter=None, instrument_policy=True)

    margin_deltas: list[float] = []
    scored = 0
    for er in result.episode_results:
        if not (er.policy_score_before and er.policy_score_after):
            continue
        scored += 1
        before = er.policy_score_before
        after = er.policy_score_after
        wrong = er.first_answer
        corrected = er.corrected_response

        corrected_key = _corrected_key(corrected, wrong, before, after)
        before_wrong = before.get(wrong, 0.0)
        after_wrong = after.get(wrong, 0.0)
        before_corrected = (
            before.get(corrected_key, before_wrong) if corrected_key else before_wrong
        )
        after_corrected = (
            after.get(corrected_key, after_wrong) if corrected_key else after_wrong
        )

        margin_before = before_corrected - before_wrong
        margin_after = after_corrected - after_wrong
        margin_deltas.append(margin_after - margin_before)

    assert scored > 0
    assert margin_deltas
    avg = sum(margin_deltas) / len(margin_deltas)
    assert avg > 0.0, f"expected positive margin delta, got {avg}"
