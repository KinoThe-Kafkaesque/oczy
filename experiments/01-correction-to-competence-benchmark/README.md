# Experiment: Correction-to-Competence Benchmark v2 — de-saturating the eval
Research proposal: ../../research/01-correction-to-competence-benchmark.md

## Status

- **Implementation:** `src/oczy/experiments/correction_competence_v2.py` — implemented and tested.
- **Campaign 0d48130 (2026-07-11):** **NULL** (behavior-delta transfer) — `v2_behavior_delta_mock=0.0`, `v2_discrimination=0.0`; domain_recall=1.0 but exact_recall=0.0; 5 de-saturation events, no behavior delta. Evidence: [`campaign log`](../../experiments_logs/2026-07-11_campaign_0d48130.md)

## Objective
Does a behavior-only v2 scorecard (signed interference-forgetting + separated exact/domain recall + an un-inverted `behavior_delta_per_byte`) produce non-overlapping 95% CIs between variants the current eval scores identically — steering (prefix/logit-bias vs cvec) and consolidation (S vs H)?

## Setup
- **Drivers: both.** Mock `_MockDriver` (`multi_fact_stressor.py`: `n_embd=16`, `idx=sum(ord(c))%16`, deterministic, semantically empty) for plumbing, and the real `LiquidAI/LFM2.5-1.2B-Instruct-Q4_K_M.gguf` via `LlamaCVecDriver` (`CVecDriverConfig`, `embedding=True`) for semantics. Mock and real are reported separately.
- **Reuse, do not reinvent:**
  - `src/oczy/experiments/eval_suite.py` — extend `EvalSuite` snapshots (`pre_test`/`post_test`) into a v2 `score()`; do not modify v1 in place (keep it for the A/B in C8/C9).
  - `src/oczy/experiments/baselines.py` — `ZeroMemoryAgent`, `ContextOnlyAgent`, `FastOnlyAgent`, `HippocampusOnlyAgent`, `IdentityOnlyAgent`, plus `OrganismAgent` from `organism.py`.
  - `src/oczy/experiments/multi_fact_stressor.py` — S/H consolidation harness, `co_recall`/`domain_co_recall`, `memory_bytes`, `--use-agent-prefix`, `--hybrid-cap`, FACTS `skylark`/`rook`/`marmalade`.
  - `src/oczy/experiments/codebase_qa/benchmark.py` — `_score(expected, answer)` (semantic vs domain), the marmalade/vertical exact-token probes, and METRIC-line printing for `autoresearch.sh`.
  - `src/oczy/experiments/cortex_agent.py` — `use_hybrid_consolidation`, `articulate_scale`, `use_hippocampus_prefix`, `use_logit_bias` (`logit_bias_strength=20.0`).
- **Config anchors:** `KVCortexConfig(d_cortex=8)` for steering probes (matches `benchmark.py`); `d_cortex=4` for the curriculum path (`run_curriculum.py:38-43`). `articulate_scale=0.001` for cvec; cvec disabled when a prefix is active (interference rule, `2026-06-25_prefix_steering_poc.md`).

## Conditions / ablation matrix (matched single-variable pairs)
| # | Driver | Axis | Setting | Held fixed | Single variable |
|---|---|---|---|---|---|
| C1 | real | steering | prefix-only (`use_hippocampus_prefix=True`, cvec off) | probes, seed | steering surface |
| C2 | real | steering | logit-bias (`use_logit_bias=True`, bias=20) | probes, seed | steering surface |
| C3 | real | steering | cvec-only (`articulate_scale=0.001`, no prefix) | probes, seed | steering surface |
| C4 | real | consolidation | S (`use_hybrid_consolidation=False`) | facts, seed | S vs H |
| C5 | real | consolidation | H (`use_hybrid_consolidation=True`, cap 10) | facts, seed | S vs H |
| C6 | mock | consolidation | S | facts, seed | semantics off (control) |
| C7 | mock | consolidation | H | facts, seed | semantics off (control) |
| C8 | real | metric version | legacy `EvalSuite.score` (v1) | agent=Organism, trajectory | metric version |
| C9 | real | metric version | v2 scorecard | agent=Organism, trajectory | metric version |

C1–C3 isolate the steering axis (known ground truth from runs #136–#139). C4/C5 the S/H axis. C6/C7 are the mock mirror (must show 0 semantic signal). C8/C9 run on the *same* Organism trajectory to isolate "old metric saturates / new metric separates" from any agent change.

## Procedure
1. Implement `correction_competence_v2.py`: a v2 scorer that consumes `EvalSuite` snapshots and adds the interference protocol + bootstrap CIs.
2. Interference protocol per fact pair (A, B): probe A (`acc_A_before`) → teach A → teach interfering B → re-probe A (`acc_A_after`) → probe B (`uptake`).
3. Compute the seven v2 sub-metrics + `behavior_delta_per_byte` (definitions below) from `respond()` outputs only.
4. Run C1–C3 on the marmalade/vertical exact-token probes, ≥5 seeds (curriculum seed + SVD-init shuffle).
5. Run C4–C7 on the `multi_fact_stressor` fact set, S vs H, mock and real.
6. Run C8/C9 on one shared Organism trajectory: emit both legacy and v2 cards.
7. Bootstrap 95% CIs over seeds; compute pairwise CI overlap for (C1∪C2) vs C3 and C4 vs C5.
8. Print `METRIC ...` lines so `autoresearch.sh` captures them.

## Metrics (exact names, computation, and what each replaces)
- `exact_recall` ∈{0,1}/probe = `benchmark.py _score(semantic_expected, answer)`. **Replaces** token-overlap `sense_match` (eval_suite.py:179) — cannot count a domain shift as exact.
- `domain_recall` = `_score(domain_expected, answer)`. Kept SEPARATE from exact (new; v1 conflated them).
- `uptake_gain` = `acc_after_correction − acc_before`. **Extends** the 0/1 latency (`eval_suite.py:329`) to a signed accuracy delta.
- `forgetting_delta` = `acc_A_before − acc_A_after_interference` ∈[−1,1]. **Replaces** `forgetting_score=min(1.0, post/pre)` (line 374). Signed, uncapped → no saturation; a no-op agent scores 0, not 1.
- `identity_drift` = `|id_acc_after − id_acc_before|`. **Replaces** `min(1.0, post/pre)` (line 375); a stable agent scores 0, not 1.
- `compression_ratio` = `raw_trace_size / consolidated_size` (both already in `EvalResult`, lines 433–435; newly surfaced).
- `delta_persistent_bytes` = serialized cold/persistent state AFTER − BEFORE the lesson set (changed bytes). Fallback `‖Δcold_state‖` if byte-count is constant.
- `behavior_delta_per_byte` = `(uptake_gain + transfer_gain − max(0, forgetting_delta) − identity_drift) / max(1, delta_persistent_bytes)`. **Replaces** `memory_bytes_per_behavior_delta` (lines 396/422); un-inverted (higher=better), uses CHANGED bytes not absolute pickle.
- `discrimination` = 1 if a pair's 95% CIs are disjoint else 0. NEW headline; v1 has no analog.

## Acceptance & kill criteria
- **ACCEPT** if `discrimination=1` for (C1∪C2 prefix/logit-bias) vs (C3 cvec) on `exact_recall` across ≥5 seeds (CIs disjoint), AND ≥3 of 7 sub-metrics show >0.2 spread across the baseline agent set.
- **ACCEPT-null** for S/H: `behavior_delta_per_byte` reported with CIs — disjoint = architecture detected; overlapping (n≥5) = pre-registered null. Either passes de-saturation.
- **KILL** if v2 still saturates: ≥5/7 sub-metrics within 0.05 across all agents → revert.
- **KILL** if the steering pair (known ground truth) CIs overlap → the exact axis is not measuring what runs #136–#139 proved → reject.
- **Hygiene gate:** unit test mutates `consolidation_strength` and `cold_drift`, asserts the v2 card is unchanged (no internal-mechanic leakage).

## Controls
- **Matched single-variable pairs:** C4 vs C5 flips only `use_hybrid_consolidation`; C1/C2 vs C3 flips only the steering surface; C6/C7 mirror C4/C5 on the mock driver.
- **Mock vs real:** C6/C7 must show `exact_recall=0` (hash embeddings carry no semantics — runs #85), confirming the metric reads behavior, not plumbing.
- **Metric-version control:** C8 vs C9 on the identical trajectory isolates "old saturates / new separates" from any agent change.
- **Negative control:** `ZeroMemoryAgent` must score `uptake_gain≈0` and `behavior_delta_per_byte≈0` — guards against a metric that rewards inaction (the v1 forgetting/identity flaw).

## Expected failure modes
- S/H genuinely tie on behavior (gradients don't flow — rl_pipeline_design). Outcome = measured null; hand off to 05-metabolism-loop-closure / 06-bounded-growth-consolidation.
- `delta_persistent_bytes ≈ 0` because `cold_state` is fixed-shape (norm grows, byte-count constant — `2026-06-25_svd_init_proj_c_persistence.md`). Fallback: `‖Δcold_state‖`, pre-registered.
- Real-driver CIs degenerate under deterministic greedy decode → variance only from curriculum seed; add probe items rather than seeds.
- Exact-token probe at its sensitivity limit (commit 4ef82e4) → single target may not separate; fall back to multi-fact `co_recall`.

## Artifacts to add
- `src/oczy/experiments/correction_competence_v2.py` — v2 scorer: interference protocol, seven sub-metrics, `behavior_delta_per_byte`, bootstrap CIs, discrimination test; imports `EvalSuite` snapshots + `baselines.py` + the `multi_fact_stressor` S/H harness.
- `src/oczy/experiments/tests/test_correction_competence_v2.py` — saturation regression (v1 1.0 vs v2 spread), inversion regression (monotone in transfer), internal-mechanic hygiene test, negative-control (ZeroMemory → 0).
- `experiments_logs/2026-06-28_correction_competence_v2.md` — run notes + the discrimination table.
- Reproduce:
  - `uv run python -m oczy.experiments.correction_competence_v2 --driver real --seeds 5 --conditions C1,C2,C3,C4,C5`
  - `uv run python -m oczy.experiments.correction_competence_v2 --driver mock --conditions C6,C7`
