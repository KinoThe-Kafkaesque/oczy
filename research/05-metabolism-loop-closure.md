# 05 — Closing the Metabolism Loop: Organs as Tensor Consumers

_Prove that a correction metabolized through `CortexAgent` produces compounding cold-state drift, and that the drift — not a stored label string — drives the answer._

**Status:** PROPOSED | **Thesis anchor:** experiments.txt §2 (fast-weight organ), §3 (neural hippocampus), §4 (learned plasticity), §11 (nested learning) | **Goal anchor:** GOALS.md Goal 3 (organ upgrades to tensor inputs) | **Depends on / relates to:** depends on `03-layer-l-hidden-extraction`; cross-link `01-correction-to-competence-benchmark`, `06-bounded-growth-consolidation`.

## Problem

GOALS.md Goal 3 states the metabolism loop is wired but the organs still consume strings, so "cortex metabolism without organ upgrades is a no-op." Its three done-when criteria are explicitly unmet/unmeasured:

1. *A correction through `CortexAgent` causes visible `cortex.cold_state` drift after `consolidate()`, not `corrected_answer` string retrieval.*
2. *Repeated corrections on the same concept produce compounding cold drift (not overwrite).*
3. *Stage 2 scope control becomes tractable because senses live in different cortex state regions, not the same label slot.*

The current code only partially supports this and has never validated it:

- `KVCortex.consolidate()` (`plastic-cortex/src/plastic_cortex/kv_cortex.py:354-398`) writes cold_state two ways: a **slow EMA nudge** `cold = (1-s)*cold + s*warm` (a convex combination, which mathematically *saturates* toward warm and cannot compound past warm's magnitude if warm is steady), and an **additive replay-absorption** term `cold += clip(replay_step*strength,0,1)*mean(tanh(proj_hidden @ replays))` that only fires when `len(replays) >= consolidate_replay_threshold=3`. So whether corrections *compound* vs *overwrite* hinges on the additive path firing — which today it usually does not. (Note: in the source the additive replay term runs *before* the slow nudge; ordering does not change the saturation argument.)
- `CortexAgent.consolidate()` (`cortex_agent.py:1004-1081`) already computes and returns `cold_drift = ‖cold_after − cold_before‖` (`cortex_agent.py:1067-1073`), but no experiment has plotted it across repeated corrections.
- The only relevant prior measurement is a **falsification** (`experiments_logs/2026-06-25_svd_init_proj_c_persistence.md`): 20 consolidate cycles drove cold_state norm 0.34 → 3.6 (≈10×, so drift *does* accumulate in norm) but the taught tokens never recalled at any scale. Cold state was "a posture bias, not retrievable content." That experiment proved norm grows but never tested whether the grown drift changes *behavior* in a discriminating, non-token way.
- `WorldModelCritic.record_outcome()` (`world-model-critic/src/world_model_critic/critic.py:319`) drives its TD/logistic update from a **string**-derived reward (`reward = -1.0 if actual_correction else 1.0`); Goal 3 wants it to consume `cortex.warm_cold_drift` as the prediction error. `warm_cold_drift` already exists as a status field (`kv_cortex.py:601`) but is never fed into the critic.
- `NeuralHippocampus` stores LM hiddens (`store(hidden=...)`, `neural-hippocampus/src/neural_hippocampus/hippocampus.py:61-89`) and surfaces a clustered `representative_hidden` in summaries (`neural-hippocampus/src/neural_hippocampus/core.py:222-231`), but retrieval/clustering keys on a **sha256-hash random embedding** of the query *text* (`core.py:264-274`, `dim=64`), not on the stored tensor — so the "tensor replay bank" is not actually tensor-addressed.

Net: the loop is shaped but the central claim — *experience → fast change → replay → compression → slow change* with drift (not a label) carrying the memory — is unproven, and the benchmark that would test it is saturated (`code_qa_accuracy=1.0` across runs #79–#101; `domain_co_recall` already pinned at 1/1 in run #95; `exact co_recall` floored at 0/0).

## Hypothesis

- **H1 (compounding, not overwrite):** K repeated corrections on one concept, each followed by `consolidate()`, produce a cold-state trajectory whose norm grows roughly monotonically (regression slope CI excludes 0) and whose *net* displacement ≈ the *sum* of per-step displacements (low cancellation). An overwrite architecture instead plateaus after the first correction (net displacement ≈ first-step displacement, slope ≈ 0). We predict the additive replay path is *necessary* for compounding past warm's magnitude: with the replay branch disabled the slow-EMA-only trajectory saturates toward (a possibly growing) warm. This is a prediction to be tested, not an established result.
- **H2 (drift drives the answer, not a label):** With every label/prefix/logit-bias surface disabled, an agent whose cold_state has absorbed K corrections shows a *continuous domain-shift* on a held-out probe that **increases with K** (Spearman ρ>0.5), while a zero-cold control shows ≈0 shift and a label-only (prefix) control produces shift *without* any accumulated drift. We do **not** claim exact-token recall — that path is already falsified (2026-06-25 log).

## Why now / what unblocks it

- The mechanical surfaces exist and persist: `consolidate()` is the only writer of cold_state, `cold_drift` is already returned, `warm_cold_drift` is already a status field, and `representative_hidden` tensors already flow into the replay list (`cortex_agent.py:1029-1045`). Nothing new in the C binding is required.
- A *continuous* steering read-out is now available from sibling work: the contrastive-cvec logs (`2026-06-27_contrastive_cvec_discovery.md`) established target-token **rank out of 65,536** and raw logit margins as a graded, non-saturating behavioral signal (cvec moved rank 47K→5K). We reuse that read-out machinery (per-token logits, not binary substring recall) as the basis for the domain-shift metric instead of the saturated binary recall metrics.
- This is the prerequisite validation for `06-bounded-growth-consolidation` (it needs a *measured* compounding curve before it can bound growth) and supplies `01-correction-to-competence-benchmark` with a discriminating metric to replace `code_qa_accuracy=1.0`.
- Honest dependency on `03-layer-l-hidden-extraction`: cortex perception currently reads the **final-layer mean-pooled** `peek_embedding` (`last_token_only=False`), not a mid-network residual (Goal 2 unimplemented). We run on the final-layer signal that exists today and flag layer-L as the richness upgrade; the experiment is valid on the current input but its ceiling may be input-limited.

## Approach

Tied to thesis §2/§3/§4/§11 (fast change → replay → slow change → nested learning):

- **Measure the trajectory, not a flag.** Extend `CortexAgent` instrumentation to log `cold_norm`, `warm_cold_drift`, and per-step `cold_drift` after each of K corrections on a single concept, then fit a slope and a compounding index. This directly tests Goal 3 criterion 2.
- **Dissociate drift from label.** Run held-out domain-shift probes under matched conditions: drift-only (compounded cold, no prefix/bias), label-only (zero cold + prefix), and zero-baseline. Score with the continuous domain-word logit margin (thesis §1: memory as changed dynamics).
- **Convert the critic to drift-as-error (thesis §4/§10).** Add a critic mode that takes `cortex.status()["warm_cold_drift"]` as the surprise/prediction-error target in `record_outcome`, matched-paired against today's string-reward critic, and compare correction-prediction AUC.
- **Make the hippocampus a tensor replay bank (thesis §3).** Add a retrieval/clustering path that keys on the stored LM `hidden` (cosine over d_embd) instead of the sha256 text hash, so that semantically-clustered corrections actually reach the `≥3 replays` additive-absorption threshold that compounding requires.
- **Single-variable matched pairs throughout** (repo standard): mock vs real driver, replay-on vs replay-off, drift vs label, string-critic vs drift-critic.

## Success criteria

Behavioral, continuous, and chosen to avoid the saturated metrics (`code_qa_accuracy`, binary `domain_co_recall=1/1`, `exact co_recall=0/0`):

- **C1 Compounding curve:** over K=20 corrections, `cold_norm` slope > 0 with bootstrap CI excluding 0, AND compounding index `‖Σ steps‖ / Σ‖step‖ ≥ 0.6` in the replay-on condition (the 0.6 bar is grounded, not arbitrary: a K-step random walk has expected index `1/√K 0.22` at K=20, so 0.6 is ~3× the random-walk null and corresponds to "step directions align, low cancellation"; the triangle inequality bounds the index at 1.0) while the replay-off (slow-EMA-only) condition gives slope ≈ 0 (≤ 25% of replay-on slope). Reference scale: the 2026-06-25 run grew norm 0.34→3.6 over 20 cycles.
- **C2 Drift-drives-answer:** domain-shift score (mean domain-word logit minus zero-cold baseline) rises monotonically with K (Spearman ρ>0.5, p<0.05) in the *drift-only, no-label* condition, with bootstrap CI of the K=20 value excluding 0; the zero-cold control stays within noise of 0.
- **C3 Critic conversion:** the drift-as-error critic's correction-prediction AUC on a held-out mixed correction/accept sequence is ≥ the string-feature critic's AUC by a margin whose CI excludes 0 (matched pair, same episodes).
- **C4 Tensor replay bank:** with tensor-keyed retrieval the additive replay branch fires (`len(replays) ≥ 3`) on clustered corrections on the real driver, and removing it (hash-keyed) drops C1's compounding index below 0.6 — proving tensor addressing is load-bearing.

**Kill criteria:** (a) if domain-shift saturates after k=1 (no additional behavioral change from corrections 2..K), the "metabolize, don't store" thesis fails for this organ set — report it. (b) If net cold displacement ≈ first-step displacement (compounding index < 0.3) under all settings, overwrite is confirmed and Goal 3 criterion 2 is unreachable with the current consolidate() math. (c) If drift-only domain-shift is indistinguishable from zero-cold (CI includes 0), then on this 1.2B model the final-layer signal is too weak and the result is gated on `03-layer-l-hidden-extraction`. We pre-register that **exact-token recall is NOT a success criterion** (already falsified).

## Risks & open questions

- **Slow-EMA saturation:** the convex slow nudge cannot compound on its own past a steady warm; compounding may *require* the additive replay term and/or a growing warm. If norm grows but is off-manifold (the 2026-06-25 trap: 10× norm, zero recall), C1 could pass while C2 fails — which is exactly why C2 measures behavior, not norm.
- **Scale band:** drift only shows behaviorally inside the usable `articulate_scale` band (raw_hidden ≈0.001 clean, SVD-init proj_c ≈0.03 clean, proj_random washes out 0.5–16 / garbage at 30). Wrong scale silently nulls C2. Reuse a known-good scale and sweep as a control. Note the proj_random random-init band only steers at 0.5–16; the 0.03 band is specific to an SVD-initialised proj_c, so probing must occur *after* `init_proj_c_from_svd` has been called.
- **Mock driver:** `_MockDriver` hash embeddings (n_embd=16) are semantically empty (run #85 mock co_recall 0/0), so mock domain-shift should be ≈0 — a deliberate null, not a failure.
- **Open:** Does compounding require differentiable plasticity on `proj_c` (none today; only `proj_hidden` trains, `replay_sgd_step=0.0` default)? Is `warm_cold_drift` the right critic target vs `cold_drift`? Does bounding (project 06) preserve C2 or flatten it?

## Prior evidence

- `experiments_logs/2026-06-25_svd_init_proj_c_persistence.md` — 20 consolidate cycles, cold norm 0.34→3.6 (≈10×) but taught tokens never recalled; "posture bias, not retrievable content." (norm-compounding seen; behavior untested)
- `plastic-cortex/src/plastic_cortex/kv_cortex.py:354-398` — `consolidate()` slow-EMA + gated additive replay (`consolidate_replay_threshold=3`); `:601` `warm_cold_drift` status field.
- `src/oczy/experiments/cortex_agent.py:1004-1081` — `consolidate()` builds replays from `representative_hidden` (`:1029-1045`), returns `cold_drift` (`:1073`).
- `world-model-critic/src/world_model_critic/critic.py:305-329` — TD/value head with string-derived reward `±1.0` (`:319`); Goal 3 wants `warm_cold_drift`.
- `neural-hippocampus/src/neural_hippocampus/core.py:264-274` — sha256 text-hash retrieval (`dim=64`); `:222-231` clustered `representative_hidden`.
- `experiments_logs/2026-06-27_domain_recall_metric.md` (run #95) — `domain_co_recall=1/1` (saturated), `exact co_recall=0/0`; `2026-06-27_contrastive_cvec_discovery.md` — rank-of-65,536 / logit-margin continuous read-out (47K→5K).
- `src/oczy/experiments/smoke_consolidation_uptake_compare.py` — existing correction-loop (8 corrections) + SVD-init + hard-consolidate + domain-word scoring scaffold to extend.
