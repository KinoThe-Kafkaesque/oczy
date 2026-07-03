# S3.M2a — Additive retrieval ablation results

**Date:** 2026-07-02
**Task:** S3.M2a from research/14-s3-organ-ablation-matrix.md
**Status:** Run in progress (background PID 2625911)

## Spec (quoted from research/14-s3-organ-ablation-matrix.md)

> **M2 (additive, the decisive one):** research/11's `MinimalOrganism` plus one
> component at a time. A component only *earns a place* through M2; M1 alone
> cannot save it (full-stack interactions can mask dead weight).
>
> | Component | Prediction | Reason |
> |---|---|---|
> | Hippocampus (answer-time) | positive | it is retrieval — the baseline to beat |
> | DSI fact index | positive | retrieval |
> | Scope-slot reranker | positive on Stage 2/5 | retrieval; source of honest 0.92 |

## Verdict rule

For each component, with Δ = (with − without) holdout accuracy, mean over
seeds, on the stage(s) the component claims to serve and on the all-stage
mean:

- **RETRIEVAL-BASELINE:** Δ > 0 with 95% CI excluding 0 on at least one
  stage AND all-stage Δ ≥ 0. Kept but labeled as retrieval baseline.
- **ARCHIVE:** CI includes 0 on every stage, or all-stage Δ < 0 with CI
  excluding 0. Archived under `attic/` with post-mortem.

## Configuration

| Parameter | Value |
|---|---|
| Model | Qwen2.5-0.5B-Instruct (CPU float32) |
| Seeds | 3 (pre-registered fallback; dry-run projected ~8h for 5 seeds) |
| Stages | All 6 (grounding, transfer, scope, dialog, stress, cross-domain) |
| Conditions | BASE, HIPPOCAMPUS_AT_ANSWER, DSI_FACT_INDEX, SCOPE_SLOT_RERANKER |
| Substrate | HFDriver via HF Transformers |
| cvec posture | OFF (default per S2.1 spec) |
| Prefix budget | 48 tokens |
| Split | `split_probes(stage, fraction=0.3, salt="v2")` |

## Stage overview

| Stage | Episodes | Holdout probes |
|---|---|---|
| Stage 0: Sense grounding | 8 | 3 |
| Stage 1: Transfer within domain | 8 | 1 |
| Stage 2: Scope control | 8 | 3 |
| Stage 3: Dialog | 4 | 3 |
| Stage 4: Consolidation stress | 10 | 4 |
| Stage 5: Cross-domain disambiguation | 6 | 3 |
| **Total** | **44** | **17** |

## Adaptation notes

### Hippocampus-at-answer
Lifts the `_HippocampusGuard` ban at answer time. Calls
`_hippo_raw.reinforce(request, k=3)` and injects the best correction as
`[Recall: <correction_utterance>]` prepended to the request. This is a direct
translation of the full organism's answer-time replay path (organism.py
lines 262-273), where `replay_hint` boosts the matching label in ranking.
In the LM path, prompt injection is the closest honest analogue.

### DSI fact index
Populated at teach time: stores the correction utterance's last-token hidden
state with the extracted expected answer as label. At answer time: mean-pools
the probe request's hidden state and retrieves the top fact via inner product
(F + LoRA adapter). Top label injected as `[Fact: <label>]`. Mirrors
organism.py's `use_diff_fact_index` path (lines 462-481).

### Scope-slot reranker
Adapted from organism.py's `_scope_key` / `_scope_label_for` / slot store.
**Differences from organism.py original:**
- No `CortexAgent` → no warm_state manipulation (can't restore per-slot
  warm_state into cortex)
- No `multi_label`, `sense_split`, or `ALLOC_THRESHOLD` config knobs
- Always allocates a new slot per episode (no slot reuse)
- Uses label injection as the closest honest analogue to the original's
  ranking boost: retrieved matching labels are prepended as
  `[Scope: <label1> | <label2> | <label3>]`
- Same `_cosine` and `_RETRIEVE_THRESHOLD` (0.3) from scope_selectivity_stressor

All three components compose: multiple flags on → all retrieval hints prepended.

## Dry-run projection

```
Stage 0: Sense grounding (8 episodes, 3 holdout probes, Qwen2.5-0.5B):
  BASE:                 ~397s
  HIPPOCAMPUS_AT_ANSWER: similar
  DSI_FACT_INDEX:        similar
  SCOPE_SLOT_RERANKER:   similar

Projected: 4 conditions × 6 stages × 3 seeds ≈ 8 hours
5-seed:   4 conditions × 6 stages × 5 seeds ≈ 13 hours → fallback to 3 seeds
```

## Commands

```bash
# Unit tests
uv run pytest src/oczy/experiments/tests/test_organ_additive_retrieval.py -x

# Full run (background)
nohup uv run python -m oczy.experiments.run_s3m2a > experiments_logs/s3m2a_stdout.log 2>&1 &
```

## Results

*Results pending — run in progress. The table below will be filled from
`experiments_logs/2026-07-02_s3_m2_retrieval_results.json` when complete.*

### Condition × Stage accuracy matrix (mean ± ci95_half)

| Condition | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | All-stage |
|---|---|---|---|---|---|---|---|
| BASE | — | — | — | — | — | — | — |
| HIPPOCAMPUS_AT_ANSWER | — | — | — | — | — | — | — |
| DSI_FACT_INDEX | — | — | — | — | — | — | — |
| SCOPE_SLOT_RERANKER | — | — | — | — | — | — | — |

### Δ vs BASE per component

| Component | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | All-stage | Verdict |
|---|---|---|---|---|---|---|---|---|
| HIPPOCAMPUS_AT_ANSWER | — | — | — | — | — | — | — | — |
| DSI_FACT_INDEX | — | — | — | — | — | — | — | — |
| SCOPE_SLOT_RERANKER | — | — | — | — | — | — | — | — |

## Wall clock

Projected: ~8 hours. Run started at 2026-07-02 ~01:00 UTC.

## Files changed

| File | Status |
|---|---|
| `src/oczy/experiments/organ_additive_retrieval.py` | New — module (331 lines) |
| `src/oczy/experiments/tests/test_organ_additive_retrieval.py` | New — tests (10 tests, all pass) |
| `src/oczy/experiments/run_s3m2a.py` | New — runner script |
