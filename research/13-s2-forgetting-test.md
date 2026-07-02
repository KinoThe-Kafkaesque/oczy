# 13 — The forgetting test (Sprint 2 / S2.5)

**Pre-registered 2026-07-02** (human-approved sprint setup, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations. Depends on research/11; uses research/12's content channel if
H-KVCONTENT was accepted, otherwise the S2.1 prefix channel (decided by those
verdicts, not by this experiment's results).

## Problem

`behavior_delta_per_byte` is only meaningful if the bytes counted are the ones
that *remain*. Every prior "memory" result in this repo is compatible with
"retrieval with extra steps": raw correction traces persist and answer-time
paths could lean on them. This is the thesis's signature move — *memory
becomes changed dynamics, not retrieved content* — and the first experiment
that can actually distinguish the two.

## Hypothesis

**H-FORGET:** after consolidation, deleting ALL raw hippocampal traces
(episode texts, stored hidden vectors, replay bank) leaves the minimal
organism's held-out behavior change intact, because the behavior is carried by
the consolidated artifact (cold state + compiled content channel), not by the
traces.

## Design — 2×2 deletion at K=N (same stage/split/seeds as research/11)

Train one organism per seed to K=N, consolidate, then measure holdout
accuracy under four states derived from saved copies of the SAME trained
organism (no retraining between arms):

| Arm | Raw traces | Consolidated artifact | Thesis prediction |
|---|---|---|---|
| `A_full` | kept | kept | high (= S2.1/S2.2 result) |
| `A_forget` | **deleted** | kept | ≈ `A_full` |
| `A_retrieval` | kept | **deleted** | ≈ `A_none` |
| `A_none` | deleted | deleted | ≈ vanilla |

- **Trace deletion** = hippocampus fully cleared (episode count 0) AND the
  cortex's replay bank / cached episodes cleared. Verified in-run by
  asserting the trace count is 0 and reporting `memory_bytes` before/after.
- **Artifact deletion** = consolidated content channel removed (prefix
  cleared / KV entries dropped) AND cold-state reset to boot value.
- No answer-time code path may read traces in ANY arm (this is already S2.1
  law; `A_retrieval` exists to catch violations empirically).

## Primary metric & acceptance

`forgetting_survival_ratio` =
mean over seeds of (`A_forget` − `A_none`) / (`A_full` − `A_none`).

- **Accept H-FORGET:** ratio ≥ 0.8, with the validity gate
  `A_full − A_none ≥ 0.10` (the loop must have measurably closed; otherwise
  BLOCKED, not a verdict).
- **Refute:** ratio < 0.5.
- **0.5 ≤ ratio < 0.8:** PARTIAL — reported as such; no acceptance.

## Pre-registered secondary analyses (exploratory only)

1. `retrieval_dependence` = (`A_retrieval` − `A_none`) / (`A_full` − `A_none`).
   Prediction ≤ 0.2. A high value means answer-time behavior tracks the traces
   — architectural leak; report file/line of the leaking path if found.
2. `behavior_delta_per_byte` recomputed with POST-deletion `memory_bytes`
   (the honest version of the GOALS.md metric), alongside the pre-deletion
   value for contrast.
3. Per-arm per-seed accuracy table.

## Reporting

2×2 table (mean ± CI per arm), survival + retrieval-dependence ratios,
memory_bytes before/after deletion, model id, exact commands; log to
`experiments_logs/2026-07-02_s2_5_forgetting_test.md` quoting this spec.

---

## Amendment 2026-07-02 (before any primary verdict was drawn)

The pre-registered split call `split_probes(stage, fraction=0.3, salt="v2")`
was discovered to be degenerate on stage 0: an unlucky hash assigned all 8
probes to dev, 0 to holdout — a state `validate_split` itself defines as an
ERROR. The first S2.2 run executed against this empty holdout and is recorded
as **INVALID (instrument failure)**; its 0/0 "REFUTE" is void and carries no
evidential weight for or against H-KVCONTENT.

**Repair (instrument-level, not spec-level):** `split_probes` now guarantees a
non-empty holdout for every stage by promoting the lowest-force-hash probes to
`ceil(fraction × total)` when thresholding yields none (stage 0: 3 holdout
probes). No previously non-empty split is altered (locked by regression test
`test_split_guarantee_never_alters_nonempty_holdouts`), so all previously
logged numbers remain comparable. The spec's split call, salt, and fraction
are unchanged. Amendment applies identically to research/11, 12, and 13.
