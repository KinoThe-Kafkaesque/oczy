# S3.M2a — Additive retrieval ablation on the minimal organism

**Date:** 2026-07-02 (run completed 2026-07-03, 03:04)
**Pre-registered spec:** `research/14-s3-organ-ablation-matrix.md` (M2, retrieval half)
**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (HFDriver, CPU float32, greedy)
**Eval:** frozen eval **v2** (pre-v2.1 expansion; 17 holdout probes total),
holdout split `split_probes(fraction=0.3, salt="v2")`, frozen scorer
**Base:** `MinimalOrganism` (prefix channel, cvec posture off) per research/11
**Seeds:** 3 (pre-approved fallback; projected >15 min/seed at 5)
**Wall clock:** 7450 s. `verify_manifest()` OK before and after.
**Code:** `src/oczy/experiments/organ_additive_retrieval.py`
(`uv run python -m oczy.experiments.run_s3m2a`), results JSON alongside.
**Provenance:** harness + template by the s3-m2-retrieval fanout agent
(attempt 2); the agent's omp process hit its time cap with the real run
detached and still executing; the orchestrating session verified completion
and wrote this report from the results JSON. Verdicts are NOT applied here —
adjudication combines M1+M2 (see `2026-07-03_s3_organ_triage_adjudication.md`).

## Condition × stage holdout accuracy (mean ± CI95, n=3)

| Condition | S0 ground | S1 transfer | S2 scope | S3 dialog | S4 consol | S5 cross |
|---|---|---|---|---|---|---|
| vanilla | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| BASE (minimal) | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.2500 ±0.00 | 0.5556 ±0.48 |
| + hippocampus-at-answer | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.2500 ±0.00 | 0.5556 ±0.48 |
| + DSI fact index | 0.0000 ±0.00 | 0.6667 ±1.43 | 0.0000 ±0.00 | 0.0000 ±0.00 | 0.3333 ±0.72 | 0.5556 ±0.48 |
| + scope-slot reranker | **0.6667 ±0.00** | 0.6667 ±1.43 | 0.1111 ±0.48 | 0.0000 ±0.00 | **0.5000 ±0.00** | 0.5556 ±0.48 |

## Δ vs BASE (the M2 additive deltas)

| Component | S0 | S1 | S2 | S3 | S4 | S5 | all-stage mean |
|---|---|---|---|---|---|---|---|
| hippocampus-at-answer | 0 | 0 | 0 | 0 | 0 | 0 | **0.0000** (exactly) |
| DSI fact index | 0 | +0.667 (±1.43) | 0 | 0 | +0.083 (±0.72) | 0 | +0.125, no stage CI excl. 0 |
| scope-slot reranker | **+0.667 (±0.00)** | +0.667 (±1.43) | +0.111 (±0.48) | 0 | **+0.250 (±0.00)** | 0 | +0.283, two stages CI excl. 0 |

## Findings

1. **The scope-slot reranker is the only component with zero-variance
   positive deltas**: stage 0 +0.667 and stage 4 +0.250, identical across all
   3 seeds — decisive additive evidence, consistent with M1's subtractive
   +0.205 on stage 2. Its stage-1/2 gains are seed-noisy at n=3 on v2's thin
   holdout.
2. **Answer-time hippocampal retrieval adds exactly nothing** — bit-identical
   to BASE on every stage and seed. Storing correction exemplars and
   consulting them per-request (reinforce k=3) never changed a scored answer.
3. **DSI shows a suggestive but unsupported stage-1 signal**: 2 of 3 seeds
   perfect on v2's single stage-1 holdout probe (CI ±1.43 — meaningless at
   n=3×1 probe). The v2.1 expansion (40 stage-1 probes) is the pre-registered
   place to re-test this before drawing anything. M1 found DSI net-harmful in
   the full stack (−0.060 all-stage).
4. **The BASE minimal organism is not uniformly zero**: stage 4 holdout 0.250
   with zero seed variance (vanilla 0.000) and stage 5 0.556 vs vanilla
   0.333. The prefix content channel does lift multi-episode consolidation
   probes — a small honest positive that S2.1's stage-0-only verdict could
   not see (recorded as exploratory; S2.1's REFUTE on its pre-registered
   primaries stands).

## Deviations

- 3 seeds (pre-approved fallback), eval v2 (worktree branched before the
  v2.1 expansion — deliberate, to keep the M2 matrix like-for-like).
- The agent's omp process died at its time cap; run completion and this
  report were finalized by the orchestrating session (data untouched from
  the results JSON).
