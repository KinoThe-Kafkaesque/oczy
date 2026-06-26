# Experiment Summary — 2026-06-25

Seed: 0 | Sense matching: enabled | Consolidation: enabled

| Agent | Uptake | Transfer | Scope | Forget | Consol | Identity | Mem/Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ZeroMemoryAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 69.0 |
| ContextOnlyAgent | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 684.75 |
| FastOnlyAgent | 0.6667 | 0.1667 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 12.0 |
| HippocampusOnlyAgent | 0.6667 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.25 |
| IdentityOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 493.0 |
| OrganismAgent | 0.6667 | 0.2500 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 68636.5 |

## Key observations

- FastOnlyAgent remains the most memory-efficient learner (12 bytes/Δ).
- OrganismAgent achieves the highest transfer (0.25) but pays a very large
  memory cost, so its composite behavior-delta-per-byte score is still low.
- HippocampusOnly and IdentityOnly baselines continue to struggle with transfer
  in the simple word-association domain, showing the standalone organs are not
  yet compressing lessons efficiently.
- The central bottleneck remains consolidation: the experience autoencoder and
  identity hypernetwork store large serialized objects rather than compact
  adapters.
- Recent autoresearch work on the codebase-QA benchmark reached
  `code_qa_accuracy=1.0` and `cortex_agent_recall_accuracy=1.0`.
- Consolidation-uptake probes showed boot-persistent *domain* shift via cvec
  steering, but exact-token uptake failed until a soft-prompt prefix was used.
  See `experiments_logs/2026-06-25_prefix_steering_poc.md` for the prefix
  steering proof of concept.

## Session delta (2026-06-25 continuation)

Commits since previous summary:
1. `8ee8d8e` — Make hippocampus replay tensor-native: hidden vectors stored with traces,\n+   `consolidate()` uses mean-cluster hidden replays, `cold_drift=0.324` in manual probe.
2. `87a7779` — Optimize LM boundary and status serialization:\n+   - embedding cache in `LlamaCVecDriver.peek_embedding`,\n+   - shared SVD projector / uniform cvec path in `KVCortex`,\n+   - optional `serialized_bytes` in organ `status()` methods.
3. `52f257f` — First-class `ReservedPosition` abstraction replaces literal\n+   `articulation_prefix`; LM perception parser hardened for short ambiguous tokens;\n+   ruff/pyright/pytest markers added to `pyproject.toml`.
4. `a5468b6` — Complete remaining review items:\n+   - bound linear-growth organs (`WorldModelCritic`, `IdentityHypernetwork`,\n+     `SurpriseGatedMemory`) with configurable caps and decay,\n+   - driver profiles + `OCZY_*` env-aware config for `CVecDriverConfig` and\n+     `LanguageAdapterConfig`,\n+   - versioned non-pickle `KVCortex` persistence via `manifest.json` + `arrays.npz`.
5. `7450067` — Tidy tooling config and fix ruff warnings.
6. `a9cca21` — Update `GOALS.md` to mark reserved-position API implemented.
7. `3985f04` — Wire ReservedPosition selection from the KnowledgeStore into
   `CortexAgent.articulate()`. Added `KnowledgeStore.get_reserved_position()` and
   tagged five facts with `reserved_token` metadata. `CortexAgent.articulate()` now
   applies the reserved prefix when recalling facts and skips cvec steering to avoid
   cvec+prefix interference. Benchmark: `code_qa_accuracy=1.0` (run #49).
8. `040ed56` — Add optional tensor-input MLP to `WorldModelCritic` and wire
   `CortexAgent.metabolize()` to pass `self._last_hidden` to predict/record calls.
   The string-only logistic path is preserved; the MLP path is gated by
   `use_hidden=True` and lazy-initializes on first hidden vector. Benchmark
   unchanged: `code_qa_accuracy=1.0` (run #50).
9. `b951011` — Add a signed SGD replay train step on `KVCortex.proj_hidden` and
   wire `CortexAgent.consolidate()` to call it per hippocampal summary.
   Correction summaries reinforce the response direction, neutral summaries
   suppress it.  Gated off by default via `KVCortexConfig.replay_sgd_step`.
   Fast tests pass; benchmark unchanged: `code_qa_accuracy=1.0` (run #51).
10. `2f7d116` — Wire IdentityHypernetwork state adapters into `KVCortex`
    articulation bias. `IdentityHypernetwork` now learns per-concept
    `state_adapters` via EMA during `update_identity` and emits a real
    `d_cortex`-dimensional adapter delta. `CortexAgent.metabolize()` applies
    this delta through `KVCortex.set_state_bias`, which is added to
    `warm_state` during cvec emission. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #54).
11. `877f329` — Convert `ExperienceAutoencoder` to optionally compress
    hidden-state deltas (`last_hidden - prev_hidden`) instead of raw text
    tokens. The new path is gated by `use_hidden_delta` (default off) and
    lazy-initializes a separate `_A_hidden` sensing matrix. `CortexAgent`
    now stores `_prev_hidden`, computes the delta, and trains the autoencoder
    on it. Legacy text-token path unchanged. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #55).
12. `24062b0` — Default `CortexAgent`'s `WorldModelCritic` to
    `use_hidden=True` so corrections are predicted from the LM hidden vector
    via a lazy-initialized MLP, rather than from string heuristics alone.
    Standalone `WorldModelCritic` keeps `use_hidden=False` for backward
    compatibility. Benchmark unchanged: `code_qa_accuracy=1.0` (run #56).
13. `18acbf8` — Wire the world-model critic's correction-likelihood
    probability into `DigestiveGate.ingest()` as a learned surprise signal,
    blended with the raw latent drift. `CortexAgent.metabolize()` passes
    `world_model_critic._last_correction_prob` to the gate. Default weights
    keep behavior near the old drift-only regime when the critic is
    unavailable. Benchmark unchanged: `code_qa_accuracy=1.0` (run #57).
14. `05cb3d2` — Add `CortexAgent.answer()` as a one-shot LM-driven answer
    surface. `OrganismAgent` gets `use_cortex_lm_answer` config flag (default
    False) that lets it delegate to the cortex agent instead of
    `PlasticCortex.answer()`. Codebase-QA recall path remains unchanged.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #58).

16. `8189823` — Wire the `WorldModelCritic` value head into
    `CortexAgent` metabolism. The critic is now constructed with
    `use_value_head=True`, and `metabolize()` passes the previous LM hidden
    as the TD state and the current hidden as the next state. The
    correction-prob MLP still trains on the current hidden. Benchmark
    unchanged: `code_qa_accuracy=1.0` (run #60).

17. `413a2b1` — Add a gated learned response-policy head to `CortexAgent`
    (`policy_score`, `policy_select`, `policy_update`). It scores candidate
    responses by combining the cortex warm state with per-candidate LM hidden
    vectors and supports a one-step REINFORCE gradient update. Gated by
    `use_policy_head` (default False); no other behavior changed.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #61).

18. `cd9991e` — Optionally wire the `CortexAgent` policy head into
    `OrganismAgent._rank_answer` via `use_cortex_policy` (default False) and
    `cortex_policy_weight`. When enabled, policy scores are added to the
    existing heuristic ranking. Default path unchanged.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #62).

19. `bf0cd1f` — Close the Phase 2 loop: `OrganismAgent._learn_from_correction`
    now calls `cortex_agent.policy_update(..., reward=-1.0, ...)` when
    `use_cortex_policy=True`, so the policy head is trained on real
    correction feedback. Benchmark unchanged: `code_qa_accuracy=1.0` (run #63).

20. `1ea95c9` — Make the Phase 2 policy signal symmetric: in addition to
    penalising the prior wrong answer with reward=-1.0, `OrganismAgent`
    now reinforces the corrected expected answer with reward=+1.0. Gated by
    `use_cortex_policy`. Benchmark unchanged: `code_qa_accuracy=1.0` (run #64).

21. `46fcdfb` — Connect the Phase 1 value head to the Phase 2 policy head:
    `OrganismAgent` now supports `use_value_baseline=True`, which passes
    the `WorldModelCritic` predicted return as the REINFORCE baseline for
    `policy_update`. Default baseline remains 0.0; benchmark unchanged:
    `code_qa_accuracy=1.0` (run #65).

22. `cc5cc2f` — Add an optional acceptance-reward policy update: when
    `use_acceptance_policy_reward=True`, `OrganismAgent` reinforces the
    chosen candidate with `reward=+1.0` whenever the critic predicts the
    emitted answer is acceptable. Refactors `_learn_from_correction` and
    the acceptance path through a shared `_policy_update_with_baseline`
    helper. Benchmark unchanged: `code_qa_accuracy=1.0` (run #66).

23. `d7ee473` — Add optional `--policy-log` instrumentation to the organism
    curriculum runner so the gated actor-critic loop (`use_cortex_policy`,
    `use_value_baseline`, `use_acceptance_policy_reward`) can be observed in a
    real word-association correction/uptake scenario. Default curriculum run
    unchanged; instrumentation only writes scores when `--policy-log` is used.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #67).

24. `ed63fe8` — Add a deterministic `CortexAgent` shim inside the organism
    curriculum runner. With `--use-cortex-shim`, `run_curriculum.py` now
    attaches a lightweight policy-head stand-in (no LM required), records
    non-null `policy_score_before`/`policy_score_after` per episode, and
    prints `Average corrected-answer policy score delta`. Stage 0 probe run
    produced a finite average delta of `-0.5576`. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #68).

25. `56abff2` — Improve the shim probe reporter to print both the absolute
    corrected-answer policy score delta and a corrected-vs-wrong *margin*
    delta. Stage 0 probe now produces absolute delta `-0.5075` and margin
    delta `+0.0926`, showing that relative preference shifts toward the
    corrected answer even when absolute score drift is negative. Benchmark
    unchanged: `code_qa_accuracy=1.0` (run #69).

26. `d3a5528` — Add a reproducible unit test
    (`src/oczy/experiments/organism_curriculum/tests/test_shim_policy_delta.py`)
    that runs the deterministic `CortexAgent` shim through the stage-0
    curriculum and asserts the corrected-vs-wrong policy margin delta is
    positive. Fast suite: `264 passed`. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #70).

27. `78e12ad` — Extend the curriculum runner with `--use-cortex-agent-mock`,
    which attaches a *real* `CortexAgent` driven by a deterministic mock LM
    driver (no real model required). Stage 0 probe run produced absolute
    delta `-0.0178` and margin delta `+0.0185`. Also fixed a numpy truth-value
    bug in `OrganismAgent._policy_update_with_baseline` that would break the
    value-head baseline path when `_prev_hidden` is a numpy array. Added
    regression test
    `src/oczy/experiments/organism_curriculum/tests/test_cortex_agent_policy_delta.py`.
    Fast suite: `265 passed`. Benchmark unchanged: `code_qa_accuracy=1.0`
    (run #71).

28. `1b1b067` — Add transfer-generalization test for the real `CortexAgent`
    policy head on organism curriculum stages 0+1. After stage 0 corrections,
    the policy head assigns a higher score to the corrected label than to the
    original wrong label on stage 1 transfer probes (different wording).
    Fast suite: `266 passed`. Benchmark unchanged: `code_qa_accuracy=1.0`
    (run #72).

29. `b9028b8` — Add `--use-real-driver` to the curriculum runner, which loads
    the local `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` model and attaches a real
    `CortexAgent` with `use_policy_head=True`. A probe of stages 0+1 completed
    in ~14s and produced a corrected-answer policy margin delta of `+1.4291`
    on real model hidden vectors, far stronger than the mock-driver probe.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #73).

30. `71baa3f` — Normalize policy-head scores to softmax probabilities in
    `OrganismAgent._rank_answer`. The raw-score contribution was unbounded
    and grew across episodes; softmax over the candidate set keeps the
    policy signal in `[0, 1]` so it can cleanly override the PlasticCortex
    +1.0 fast-answer bias. Fixed-margin override (run #74) was discarded.
    Real-driver curriculum stages 0+1 now achieve retention `0.88` and
    transfer `1.00` (up from `0.38`), with a corrected-answer margin delta of
    `+1.6694`. Fast suite: `266 passed`. Benchmark unchanged:
    `code_qa_accuracy=1.0` (run #75).

Test status: `pytest: 266 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer unit tests pass; full slow/model suite not rerun).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
  predicted-accepted answers. The policy head's ranking contribution is now normalized
  to softmax probabilities for stable, bounded influence. A curriculum instrumentation
  hook (`--policy-log`), a deterministic shim, a mock-driver `CortexAgent`, and a real
  LM driver (`--use-real-driver`) are available. The real LM-driven stages 0+1 achieve
  near-perfect retention and transfer in the probe configuration.

31. `a07fa77` — Add gated `use_policy_request_context` to `CortexAgent`.
    `perceive()` stores a request-context hidden vector and `_policy_features()`
    concatenates `[warm_state; request_context; candidate_hidden]`, giving the
    policy head a fixed context signal for context-dependent ranking. Default
    off so the benchmark is unaffected. Fast suite: `268 passed`.
    Real-driver stages 0+1 unchanged; stage 2 still shows `uptake=0/8`
    because the curriculum's default commonsense responses are label-wrong
    but never trigger a correction signal, so the policy head cannot learn
    the alternate sense. Benchmark unchanged: `code_qa_accuracy=1.0` (run #76).

Test status: `pytest: 268 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context unit tests pass).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.

32. `d9301db` — Move the `use_cortex_policy` policy-head update outside the
    critic-surprise gate in `OrganismAgent._learn_from_correction`. Previously
    the policy update lived inside `if prediction_error > _surprise_threshold:`;
    a well-calibrated critic predicts low `accepted_prob` on corrections, so a
    *better* critic *suppressed* policy learning. The policy head now updates
    on every correction; hippocampus/autoencoder/identity/immune remain gated.
    Added regression test `test_policy_update_fires_even_when_critic_not_surprised`.
    Mock/shim Stage 2 scope now reaches `uptake=1.00`. Real-driver Stage 2
    policy head shows learning (margin delta `+1.0160`), but uptake remains
    `0/8` because the ranking function's token-overlap and identity terms
    dominate the policy signal on real embeddings. Fast suite: `269 passed`.
    Benchmark unchanged: `code_qa_accuracy=1.0` (run #77).

Test status: `pytest: 269 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated unit tests pass).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
33. `c723660` — Add gated `policy_suppresses_fast_answer` flag to
    `OrganismAgent`. When enabled and a policy signal is present,
    `_rank_answer` suppresses the legacy `+1.0` fast_answer bias, allowing
    the learned policy head to drive final selection. The curriculum probe
    modes (`--use-cortex-shim`, `--use-cortex-agent-mock`, `--use-real-driver`)
    enable it by default. Added regression tests for weak-preference wins
    and default-off legacy behavior. Mock/shim Stage 2 remains at full
    uptake; real-driver Stage 2 now shows non-zero policy-driven learning
    (retention `0.25`, scope `0.50`). Fast suite: `271 passed`. Benchmark
    unchanged: `code_qa_accuracy=1.0` (run #78).

Test status: `pytest: 271 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer unit tests pass).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.

34. `ffbe2b9` — Add a configurable `IngestionPipeline` scaffold upstream of
    `CortexAgent.metabolize()`. The pipeline splits an utterance into
    `Chunk` objects, runs pluggable chunkers (fixed-window / sentence /
    paragraph / recursive), salience filters (pass-through /
    correction-marker / lexical-novelty / centroid-cosine), embedders
    (none / same-LM / identity placeholder), and observation modes
    (parallel / sequential). It emits `ChunkSignal` traces for direct
    hippocampal storage and a `TurnDigest` consumed by a new
    `DigestiveGate.ingest_digest()` method that maps within-turn
    statistics onto the existing scalar gate surface. Default off; the
    benchmark remains `code_qa_accuracy=1.0` (run #79). Fast suite:
    `290 passed`.

Test status: `pytest: 290 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline unit tests pass).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.

34. `bbd3474` — Add needle-per-turn stressor tests in
    `src/oczy/experiments/tests/test_ingestion_needle.py`. Three slow tests
    compare the status-quo single-embed metabolism against the new pipeline
    on a 512-token turn with the needle at position 0.8: the baseline stores
    ≤1 trace and misses the needle (recall 0), while the pipeline stores
    per-chunk traces and retrieves the needle (recall 1). A third test checks
    that embedding calls scale with the fixed-window chunk count. Fast suite
    unchanged at `290 passed`; slow needle tests pass in ~0.24s. Benchmark
    remains `code_qa_accuracy=1.0` (run #80).

Test status: `pytest: 290 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor unit tests pass; slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
35. `a1e9313` — Add `src/oczy/experiments/needle_sweep.py`, a runnable
    needle-position/length sweep benchmark. It emits `METRIC` and `ASI`
    lines that the autoresearch harness can parse. At length 512, the
    baseline single-embed metabolism achieves recall `0/5` across positions,
    while the pipeline with a 64-token fixed-window chunker and pass-through
    salience achieves recall `5/5` at a cost of ~10 embedding calls per
    position. Fast suite unchanged at `290 passed`; benchmark remains
    `code_qa_accuracy=1.0` (run #81).

Test status: `pytest: 290 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor + needle-sweep unit tests pass; slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
35. `42c4ef9` — Fix the default `salience_threshold` in `IngestionPipeline`
    so non-`pass-through` filters (correction-marker, lexical-novelty,
    centroid-cosine) drop chunks below salience 0.5 by default. Updated
    `test_correction_marker_filter_marks_only_correction_chunks` to expect
    only the correction chunk to survive. Verified with `needle_sweep.py`:
    at length 512, lexical-novelty recall stays 5/5 while embedding calls
    drop from 50 (pass-through) to 14; at length 4096, pass-through fails
    retrieval with 222 calls while lexical-novelty succeeds with only 8
    calls. Fast suite `290 passed`. Benchmark `code_qa_accuracy=1.0`
    (run #82).

Test status: `pytest: 290 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor + needle-sweep + salience-threshold unit tests pass; slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.

- CortexAgent now has a configurable `IngestionPipeline` upstream of metabolism with
  working salience filtering: `lexical-novelty` keeps 100% needle recall while cutting
  same-LM embedding calls by ~95% at 4k tokens vs pass-through.

36. `87ff263` — Add `mock-foreign` embedder option to `IngestionPipeline`.
    It builds a deterministic character-trigram histogram and projects it
    into `n_embd` via a lazy learned projection layer, exercising the
    foreign-embedding + projection architecture without adding dependencies.
    The pipeline now injects `n_embd` into `ctx_state` from the driver so
    all embedders can size correctly. Needle sweep at 512 tokens:
    same-LM `mean_recall=1.00` with 14 embedding calls; mock-foreign
    `mean_recall=1.00` with 5 calls (no driver forwards). Fast suite
    `293 passed`. Benchmark `code_qa_accuracy=1.0` (run #83).

Test status: `pytest: 293 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor + needle-sweep + salience-threshold + mock-foreign-embedder unit tests pass;
slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
37. `f78f43d` — Add `DigestiveGateConfig.use_hybrid_consolidation` and wire
    `CortexAgent.turn()` to scale consolidation strength by
    `TurnDigest.drift_max` when both the ingestion pipeline and hybrid mode
    are enabled (architecture H). Architecture S keeps the existing
    pressure-derived scalar behavior. Mock-driver curriculum stages 0+1 show
    identical S/H results (retention 0.88, transfer 1.00) because the
    episodes are single-sentence and `auto_consolidate` is disabled in the
    runner, so consolidation never fires. Fast suite `294 passed`.
    Benchmark `code_qa_accuracy=1.0` (run #84).

Test status: `pytest: 294 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor + needle-sweep + salience-threshold + mock-foreign-embedder +
hybrid-consolidation unit tests pass; slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
38. `0f566c4` — Add `src/oczy/experiments/multi_fact_stressor.py` and tests.
    The probe buries a novel fact and a correction-styled fact in filler
    (length 512), processes the turn with the chunked ingestion pipeline,
    forces consolidation, and measures `recall_a`, `recall_b`, and
    `co_recall` under scalar and hybrid modes. On the mock driver S and H
    show the expected mechanical difference in consolidation strength
    (1.0 vs ~6.4) and cold drift, but semantic recall remains 0/0 because
    the mock driver's hash embeddings are not retrievable. The stressor
    is ready for real-driver evaluation. Fast suite `298 passed`.
    Benchmark `code_qa_accuracy=1.0` (run #85).

Test status: `pytest: 298 passed` fast (reserve-position + tensor-critic + replay-SGD +
identity-adapter + hidden-delta + default-critic + critic-gate + cortex-answer-loop +
value-head + value-head-wiring + policy-head + organism-policy + policy-correction-loop +
policy-positive-reward + actor-critic-baseline + acceptance-reward +
curriculum-shim-margin + curriculum-cortex-agent-mock +
curriculum-cortex-agent-transfer + policy-request-context +
policy-update-ungated + policy-suppresses-fast-answer + ingestion-pipeline +
needle-stressor + needle-sweep + salience-threshold + mock-foreign-embedder +
hybrid-consolidation + multi-fact-stressor unit tests pass; slow needle tests 3 passed).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
39. `b171ec0` — Extend `multi_fact_stressor.py` with `--use-real-driver` and
    `--n-ctx` so the probe can run against the LFM2.5 GGUF. Added a
    slow/requires_model test. Real-driver runs at length 256 show
    `co_recall=0/0` for both scalar and hybrid modes: consolidation strength
    still scales (1.0 vs ~3.6), but cvec-only steering cannot force the exact
    target tokens. This confirms the prior finding that exact-token recall
    needs a reserved-position/prefix surface, not just residual cvecs.
    Fast suite `298 passed`; benchmark `code_qa_accuracy=1.0` (run #86).

Test status: `pytest: 298 passed` fast + 1 slow/real-driver construction test
(reserve-position + tensor-critic + replay-SGD + identity-adapter + hidden-delta +
default-critic + critic-gate + cortex-answer-loop + value-head + value-head-wiring +
policy-head + organism-policy + policy-correction-loop + policy-positive-reward +
actor-critic-baseline + acceptance-reward + curriculum-shim-margin +
curriculum-cortex-agent-mock + curriculum-cortex-agent-transfer +
policy-request-context + policy-update-ungated + policy-suppresses-fast-answer +
ingestion-pipeline + needle-stressor + needle-sweep + salience-threshold +
mock-foreign-embedder + hybrid-consolidation + multi-fact-stressor unit tests pass;
slow needle tests 3 passed, 1 slow real-driver construction test passes).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
40. `71854bd` — Add `--use-prefix` to `multi_fact_stressor.py` so the probe can
    set a reserved-position prefix with the two facts before retrieval.
    Instruction-formatted queries are required for the Instruct-tuned LFM2.5
    model to answer. Real-driver prefix mode reaches `co_recall=1/1` for both
    scalar and hybrid, while cvec-only mode stays `0/0`, confirming the
    prefix is the exact-recall surface. A hand-coded prefix trivializes the
    task, so it does **not** discriminate S vs H. Fast suite `299 passed`;
    benchmark `code_qa_accuracy=1.0` (run #87).

Test status: `pytest: 299 passed` fast + 1 slow/real-driver construction test
(reserve-position + tensor-critic + replay-SGD + identity-adapter + hidden-delta +
default-critic + critic-gate + cortex-answer-loop + value-head + value-head-wiring +
policy-head + organism-policy + policy-correction-loop + policy-positive-reward +
actor-critic-baseline + acceptance-reward + curriculum-shim-margin +
curriculum-cortex-agent-mock + curriculum-cortex-agent-transfer +
policy-request-context + policy-update-ungated + policy-suppresses-fast-answer +
ingestion-pipeline + needle-stressor + needle-sweep + salience-threshold +
mock-foreign-embedder + hybrid-consolidation + multi-fact-stressor unit tests pass;
slow needle tests 3 passed, 1 slow real-driver construction test passes).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
41. `849c1fb` — Integrate an optional `foreign-minilm` sentence embedder into
    the ingestion pipeline using `sentence-transformers` (lazy import) and a
    learned projection into `n_embd`. Added to the `lm` optional dependency
    group. Synthetic needle sweep (mock driver) with `foreign-minilm` achieved
    `mean_recall=1.00` using 5 embedder calls versus 14 for `same-lm`;
    however, the mock same-LM cost is artificial, so the decisive comparison
    must run on the real LFM2.5 driver. Fast suite `300 passed`; benchmark
    `code_qa_accuracy=1.0` (run #88).

Test status: `pytest: 300 passed` fast + 1 slow/real-driver construction test
(reserve-position + tensor-critic + replay-SGD + identity-adapter + hidden-delta +
default-critic + critic-gate + cortex-answer-loop + value-head + value-head-wiring +
policy-head + organism-policy + policy-correction-loop + policy-positive-reward +
actor-critic-baseline + acceptance-reward + curriculum-shim-margin +
curriculum-cortex-agent-mock + curriculum-cortex-agent-transfer +
policy-request-context + policy-update-ungated + policy-suppresses-fast-answer +
ingestion-pipeline + needle-stressor + needle-sweep + salience-threshold +
mock-foreign-embedder + hybrid-consolidation + multi-fact-stressor +
foreign-minilm-embedder unit tests pass; slow needle tests 3 passed, 1 slow
real-driver construction test passes).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
42. `fb954fe` — Extend `needle_sweep.py` with `--use-real-driver` and
    `--n-ctx`, plus per-position and total wall-clock timing. Also caches
    the MiniLM sentence-transformer model across embedder instances to avoid
    repeated model loads. Real-driver length-512 needle sweep:
    `same-lm` = 29.6s, `foreign-minilm` = 42.8s, both `mean_recall=1.00`.
    Foreign-MiniLM is not cheaper at this scale because `lexical-novelty`
    keeps only 1-2 chunks per position and `same-lm` embedding is cached.
    Fast suite `300 passed`; benchmark `code_qa_accuracy=1.0` (run #89).

Test status: `pytest: 300 passed` fast + 1 slow/real-driver construction test
(reserve-position + tensor-critic + replay-SGD + identity-adapter + hidden-delta +
default-critic + critic-gate + cortex-answer-loop + value-head + value-head-wiring +
policy-head + organism-policy + policy-correction-loop + policy-positive-reward +
actor-critic-baseline + acceptance-reward + curriculum-shim-margin +
curriculum-cortex-agent-mock + curriculum-cortex-agent-transfer +
policy-request-context + policy-update-ungated + policy-suppresses-fast-answer +
ingestion-pipeline + needle-stressor + needle-sweep + salience-threshold +
mock-foreign-embedder + hybrid-consolidation + multi-fact-stressor +
foreign-minilm-embedder + real-driver-needle-sweep unit tests pass; slow needle
tests 4 passed, 1 slow real-driver construction test passes).
`ruff check` clean on changed files.

Remaining blocks:
- Direct reserved KV-slot injection still blocked by `llama-cpp-python` C API surface.
- Exact-token uptake via cvec alone remains blocked; `ReservedPosition` prefix is the
  practical exact-recall surface, and it can now be selected automatically by the
  knowledge store.
- Hippocampal replay now has a differentiable SGD path on `proj_hidden`, gated by
  `replay_sgd_step` and defaulting to off.
- IdentityHypernetwork now emits real `d_cortex`-dimensional adapter deltas that are
  applied at articulation time, but the concept→latent mapping is still partially
  hand-seeded and the effect on downstream behavior has not yet been measured.
- WorldModelCritic now has tensor-input correction prediction (default in CortexAgent),
  a learned value head that is trained with TD on every `metabolize()`, and feeds the
  digestive gate, but none have been validated in a real correction/uptake loop.
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities, and policy updates fire on every correction. The
  `policy_suppresses_fast_answer` flag lets the head dominate final ranking in probe
  modes. Stages 0+1 reach near-perfect retention/transfer with the real LM driver,
  and Stage 2 scope control is starting to show policy-driven alternate-sense selection.
- CortexAgent now has a configurable `IngestionPipeline` upstream of metabolism with
  pluggable chunkers, salience filters, embedders (same-LM, mock-foreign, and optional
  foreign-MiniLM with learned projection), a scalar stats gate, an optional hybrid
  consolidation-strength boost, and stressors for needle recall and multi-fact turns.
  Real-driver length-512 needle sweep shows same-lm (29.6s) still beats foreign-minilm
  (42.8s) when lexical-novelty keeps only 1-2 chunks. Next frontier is length 4096 to
  stress the cache-miss regime where same-lm pays for many forward passes.
