# 01 — Correction-to-Competence Benchmark v2: de-saturating the eval
*The eval can no longer tell two architectures apart; rebuild it so a behavior difference is the only thing it can register.*

Status: TESTED-NULL (2026-07-11) | Thesis anchor: experiments.txt §14 (Correction-to-Competence Benchmark), §10 (mistake immune system), §9/§13 (compression / how to train), §1 (correction-gated cortex) | Goal anchor: GOALS.md Goal 1 (LM-side steering) + Goal 3 (organ tensor upgrades / metabolism loop) | Depends on / relates to: 02-kv-slot-fact-injection, 03-layer-l-hidden-extraction, 04-context-scoped-attractors, 05-metabolism-loop-closure, 06-bounded-growth-consolidation, 07-conversation-world-model-rl

> **Outcome (2026-07-11):** TESTED-NULL. Campaign 0d48130 (colab, commit `537260c`) ran the v2 scorecard: `v2_behavior_delta_mock=0.0`, `v2_discrimination=0.0`. The benchmark produced 5 de-saturation events (separated exact-vs-domain recall, signed interference forgetting both spread=1.0) but detected **zero behavior delta** — no architecture pair separated on the new scorecard. `domain_recall=1.0` but `exact_recall=0.0`. The de-saturation itself is a result (the eval can now distinguish exact from domain), but the S/H behavioral null (H2) is the measured verdict. Evidence: `../experiments_logs/2026-07-11_campaign_0d48130.md`.

## Problem
The headline eval is saturated and cannot discriminate the architecture variants it exists to compare.

- **Saturated headline.** `autoresearch.sh` runs `oczy.experiments.codebase_qa.benchmark`; every recent run prints `code_qa_accuracy=1.0` and `cortex_agent_recall_accuracy=1.0` (`src/oczy/experiments/logs/SUMMARY.md`, runs #79–#105).
- **Capped ratio metrics reward non-regression, not learning.** In `src/oczy/experiments/eval_suite.py` three of the seven scorecard metrics are ratios clamped at 1.0: `forgetting_score = min(1.0, post/pre)` (line 374), `identity_drift_score = min(1.0, post/pre)` (line 375), `consolidation_score = min(1.0, cons_overall/post_overall)` (line 390). In `SUMMARY.md` both `FastOnlyAgent` and `OrganismAgent` score `forget=consol=identity=1.0000`; they differ only on transfer (0.1667 vs 0.25) and scope ties (0.1667 vs 0.1667). Any agent that does not *regress* on the old battery banks a free 1.0 on three of seven metrics.
- **The memory metric is inverted and pickle-based.** `memory_bytes_per_delta = consolidated_size / max(1, successful_lessons)` (eval_suite.py:396), surfaced as card key `memory_bytes_per_behavior_delta` (line 422). Lower is "better" — the literal inverse of the north star `behavior_delta_per_byte_of_persistent_memory` (rl_pipeline_design.md:342). Consequence in `SUMMARY.md`: the agent with the *best* transfer (`OrganismAgent`, 0.25) gets the *worst* Mem/Δ (68636.5 vs `FastOnlyAgent` 12.0). The composite punishes the best learner.
- **The match function counts a domain shift as exact recall.** `_matches` with `sense_match` returns True on ≥2 shared non-stopword tokens (eval_suite.py:179). A cvec that only shifts DOMAIN therefore scores as if it recalled the exact fact — even though briefs show cvec exact-token recall is provably 0 (`2026-06-25_prefix_steering_poc.md`; runs #136–#139).
- **The "measures internal mechanics not behavior" trap.** `experiments_logs/2026-06-19_extended_learning_evaluation.md` scored `NeuralHippocampus` aggregate `1.000`, ranking it ABOVE the Oracle upper bound (`0.844`). That score is surprise-gating precision + replay accuracy + compression ratio — internal organ mechanics, not `respond()`-level behavior.
- **Non-discrimination, concretely.** S vs H consolidation are mechanically distinct — `consolidation_strength` 10.0 vs 36.0 and `cold_drift` 0.10 vs 0.64 — yet tie on EVERY behavioral metric: identical `co_recall` and `domain_co_recall`, and near-identical `memory_bytes` (29,211 vs 29,214 uncapped; 10,443 vs 10,444 capped) (runs #84, #85, #94, #95; `experiments_logs/2026-06-27_memory_per_byte_sh.md` and `2026-06-27_domain_recall_metric.md` run #95). The eval cannot separate the two architectures it is meant to compare.

## Hypothesis
- **H1 (de-saturation).** A behavior-only scorecard built from (a) signed interference-based forgetting, (b) separated exact-vs-domain recall, and (c) an un-inverted `behavior_delta_per_byte` over *changed* persistent bytes will produce non-overlapping 95% CIs between steering variants known to differ behaviorally (prefix / logit-bias `exact_recall≈1` vs cvec-only `exact_recall≈0`) — pairs the current scorecard scores identically.
- **H2 (null-capable S/H verdict).** For S vs H consolidation the same scorecard will EITHER separate them (disjoint CIs on ≥1 behavioral metric) OR establish a measured null (CIs overlap, n≥5) — replacing today's saturated 1.0 tie with a quantified, falsifiable statement.

## Why now / what unblocks it
- Every ingredient already exists; the blocker was metric design, not tooling. `eval_suite.py` owns the snapshot/score scaffold, `baselines.py` the matched comparators (`ZeroMemoryAgent`, `ContextOnlyAgent`, `FastOnlyAgent`, `HippocampusOnlyAgent`, `IdentityOnlyAgent`) plus `OrganismAgent`, `multi_fact_stressor.py` the S/H consolidation harness with mock + real LFM2.5 drivers and `memory_bytes`, and `codebase_qa/benchmark.py` the two-axis `_score` (semantic vs domain) and the known-ground-truth steering separation from runs #136–#139.
- The needed compression inputs are *already captured but discarded*: `EvalResult` stores `raw_trace_size` and `consolidated_size` (eval_suite.py:433–435) but `final_card` never surfaces a compression ratio.
- Recent logs explicitly ask for "a metric that moves" (`memory_per_byte_sh` open Q1; SUMMARY remaining-blocks: "further progress on behavior_delta_per_byte requires moving beyond exact-token recall probes").
- This project is upstream of all siblings (02–07): none of them can claim a win until the eval can register one.

## Approach (tied to thesis §14)
- **Behavior-only rule.** Every metric is a pure function of `agent.respond()`/`articulate()` on held-out probes; no metric may read internal organ state (`consolidation_strength`, `cold_drift`, surprise-gating precision). This structurally kills the NeuralHippocampus-1.000 trap (§14, §10).
- **Interference forgetting (§1, §3).** Teach A, probe A; teach interfering B; re-probe A. `forgetting_delta = acc_A_before − acc_A_after ∈ [−1,1]`, signed and uncapped. A stable no-op agent scores 0 here AND 0 on uptake — it cannot farm a free 1.0.
- **Two-axis recall (§14; relates 02, 03).** Keep exact-token and domain as separate scores (reuse `benchmark.py _score`) so domain-only steering can never count as exact recall.
- **Un-invert the north star (§9, §13; relates 06).** `behavior_delta_per_byte = net_behavior_delta / max(1, Δpersistent_bytes)`, where `Δpersistent_bytes` is the serialized cold/persistent state AFTER − BEFORE learning (changed bytes, not absolute pickle). Surface `compression_ratio = raw_trace_size / consolidated_size` as its own sub-metric.
- **Discrimination is the headline acceptance test.** Across N seeds the benchmark must separate ≥1 known-different pair with disjoint 95% CIs. If it cannot even separate the steering pair (whose ground truth is known from runs #136–#139), the benchmark is rejected.

## Success criteria (discriminating, with kill criteria)
Primary (discrimination power):
- Across ≥5 seeds, the v2 scorecard yields non-overlapping 95% CIs between prefix/logit-bias (`exact_recall≈1`) and cvec-only (`exact_recall≈0`) on the exact_recall axis. **KILL:** if those CIs overlap, the new exact axis is no better than the old token-overlap match — reject and revert.
- ≥3 of the 7 sub-metrics show a spread >0.2 across the `baselines.py` agent set + `OrganismAgent`. **KILL:** if ≥5 of 7 still saturate (all agents within 0.05), de-saturation failed.

Secondary (the S/H verdict):
- Report `behavior_delta_per_byte` for S and H with 95% CIs. SUCCESS = disjoint CIs (difference detected) OR a pre-registered null (CIs overlap, n≥5). Today's 1.0 tie is neither.

Hygiene:
- No scorecard metric reads internal organ state — verified by a unit test that mutates `consolidation_strength`/`cold_drift` and asserts the v2 card is byte-identical.
- `behavior_delta_per_byte` is monotone increasing in transfer at fixed bytes (a regression test the legacy inverted metric fails).

## Risks & open questions
- **S vs H may be a genuine behavioral null** because gradients do not flow through the toy organs (rl_pipeline_design.md; SUMMARY). Then the honest deliverable is a measured null + handoff to 05-metabolism-loop-closure. That is still a de-saturation win (a moving, falsifiable number) but not an architecture win.
- **Δpersistent_bytes may be ≈0** because `cold_state` is a fixed-shape array — its norm grows 0.34→3.6 over 20 cycles but its byte count is constant (falsification in `2026-06-25_svd_init_proj_c_persistence.md`). Fallback: use `‖Δcold_state‖` (information delta) as the denominator; pre-registered.
- **Real-driver CIs may be degenerate** because greedy LFM2.5 decoding is deterministic per prompt; seed variance comes only from curriculum sampling + SVD-init. May need more probe items, not more seeds. The mock `_MockDriver` (hash embeddings) carries 0 semantics, so exact recall is structurally 0 there (runs #85) — report mock and real separately.
- **Exact-token probes are at their sensitivity limit** (commit 4ef82e4 "probe is at its sensitivity limit"); a single-target probe may not separate. Fall back to multi-fact `co_recall` (`multi_fact_stressor.py`) as the exact axis.
- **The coherence heuristic also collapsed** (commit `1a6f298`: "coherence sensitivity=0 for both parallel and sequential modes, all coh=1"). With order-shuffle CIs degenerate and the new tiling-detector (2–6 char) fixing only the false-positive side, the heuristic joins the ranks of saturated metrics — direct evidence the eval went blind, not just one metric.

## Prior evidence
- `src/oczy/experiments/eval_suite.py`: capped ratios (lines 374, 375, 390); inverted pickle memory metric (lines 396, 422); token-overlap sense match (line 179); `raw_trace_size` captured but absent from `final_card` (lines 433–435 vs 416–426).
- `src/oczy/experiments/logs/SUMMARY.md` scorecard: `FastOnlyAgent`/`OrganismAgent` tie at forget=consol=identity=1.0; Mem/Δ 12.0 vs 68636.5 inverts against transfer 0.1667 vs 0.25.
- `experiments_logs/2026-06-19_extended_learning_evaluation.md`: `NeuralHippocampus` 1.000 > Oracle 0.844 (internal-mechanics score).
- S vs H ties: runs #84, #85, #94, #95; `experiments_logs/2026-06-27_memory_per_byte_sh.md` (strength 10 vs 36, identical `co_recall`, near-identical `memory_bytes` 10,443 vs 10,444); `2026-06-27_domain_recall_metric.md` run #95 (identical `domain_co_recall` 1/1, memory_bytes 29,211 vs 29,214).
- Steering ground truth to separate: prefix `exact=1/domain=1` vs cvec `exact=0/domain=1` (`2026-06-25_prefix_steering_poc.md`); logit-bias `exact=1` (runs #136–#137); composition run #139.
- North star defined but never scored: `behavior_delta_per_byte_of_persistent_memory` (rl_pipeline_design.md:342); `autoresearch.sh` → `codebase_qa/benchmark.py` is the entry point.
