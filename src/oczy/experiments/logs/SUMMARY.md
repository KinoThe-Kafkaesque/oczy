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
- CortexAgent's policy head can now optionally consume a request-context hidden vector
  in addition to warm_state and candidate hidden vectors, gated by
  `use_policy_request_context`. The ranking contribution remains normalized to
  softmax probabilities. Policy updates now fire on every correction regardless of
  critic surprise. Stages 0+1 reach near-perfect retention/transfer with the real LM
  driver; Stage 2 scope control is partial: mock/shim reaches full uptake, while the
  real-driver head learns alternate labels but final ranking is still dominated by
  token-overlap/identity scoring.
