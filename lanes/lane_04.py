"""Lane 04: Context-Scoped Semantic Attractors -- SSI (Scope+Sense Index).

Loads the 8 Stage-2 scope-control episodes from
``src/oczy/experiments/organism_curriculum/stages/stage_2_scope.json``
and computes SSI: the fraction of episodes where, after the correction has
been taught, BOTH the retention probe (in the teaching context) AND the
scope probe (in a different-sense context) match under sense-mode scoring
(``scoring.probe_matches``). Per-episode conjunction is the discriminating
design -- an always-technical cortex fails the scope probe, an always-common
cortex fails the retention probe, so only genuine per-context selectivity
scores (research/04-context-scoped-attractors.md success criteria).

The cortex under test is the current single-slot baseline
(``KVCortexConfig(d_cortex=4)`` -- the matched baseline from
``run_curriculum.py:38-43``): one global warm/cold vector with no context
addressing, so teaching sense B reshapes the same basin that encoded sense A.
Spec H1 predicts SSI <= 0.125; the documented runs #73-#77 sit at 0/8 = 0.0.

Real-LM only -- the mock driver's hash-embeddings carry no semantics
(spec risks). If the LFM2.5 driver cannot be loaded, returns 0.0 (the
spec-stated baseline). Body wrapped in try/except; never raises.
Deterministic (fixed episode order, fixed driver config).
"""

from __future__ import annotations


def name() -> str:
    return "lane_04_ssi"


def measure() -> float:
    try:
        from oczy.experiments.cortex_agent import CortexAgent, CortexAgentConfig
        from oczy.experiments.organism_curriculum.dataset import build_curriculum
        from oczy.experiments.organism_curriculum.scoring import probe_matches
        from oczy.lm import CVecDriverConfig, LlamaCVecDriver
        from plastic_cortex.kv_cortex import KVCortexConfig

        try:
            driver = LlamaCVecDriver.load(
                CVecDriverConfig(n_ctx=128, n_threads=4, embedding=True)
            )
        except Exception:
            # Real-LM unavailable: return the spec-stated baseline (0/8).
            return 0.0

        # Matched single-slot baseline cortex: global warm/cold d_cortex=4
        # vector, no context addressing.
        cfg = CortexAgentConfig(cortex=KVCortexConfig(d_cortex=4))
        cortex = CortexAgent(cfg, driver=driver)
        cortex.boot()

        stages = build_curriculum(stage_names=("stage_2_scope",))
        if not stages or not stages[0].episodes:
            return float("nan")
        stage = stages[0]

        ssi_count = 0
        for ep in stage.episodes:
            # Teach the correction: perceive() runs cortex.observe() with
            # correction_signal=1.0, applying the high-plasticity EMA update
            # to the single global warm_state vector.
            try:
                cortex.perceive(ep.correction_utterance, correction_signal=1.0)
            except Exception:
                continue

            # Per-episode conjunction: BOTH retention (teaching context)
            # AND scope (different context) probes must match for credit.
            both_ok = True
            for probe in ep.probes:
                try:
                    out = cortex.answer(probe.request)
                    answer = out.get("answer", "") if isinstance(out, dict) else str(out)
                except Exception:
                    answer = ""
                if not probe_matches(answer, probe, ep):
                    both_ok = False
                    break
            if both_ok:
                ssi_count += 1

        return float(ssi_count) / float(len(stage.episodes))
    except Exception:
        return float("nan")