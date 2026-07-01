# Honest post-leakage-removal baseline (Sprint 0 / S0.2 DoD)

**Date:** 2026-07-01
**Context:** Sprint 0 (SPRINT.md) removed two classes of test-set leakage from
the answer path (branch `fix/s0-2-leakage`, 11 sites across `lanes/lane_04.py`,
`scope_selectivity_stressor.py`, `run_curriculum.py`):

1. `_SCOPE_TEACHING` dictionaries keyed to specific episode IDs (including
   entries authored for exactly the episodes that were failing), consumed by a
   pre-stage teaching pass.
2. `prefix_targets=[probe.expected]` — the probe's expected answer fed into
   the generation-time logit-bias path for scope probes.

This log records the curriculum scores **after** removal, per S0.2's
definition of done. These are the numbers all future work must beat.

## Configurations

- Frozen eval data: `src/oczy/experiments/organism_curriculum/stages/` (44
  episodes), semantic scoring, `--split all` (matches the historical runs the
  old claims came from; the dev/holdout split lands with S0.6 for future work).
- Vanilla column: `VanillaAgent` (no-op learn) over the same probes — new in
  S0.7, so historical runs have no equivalent.
- Reports: `organism_curriculum/reports/honest_baseline_{mock,mock_seed0,real}_2026-07-01.json`.

## Results — real driver (LFM2.5-1.2B GGUF, `--use-real-driver`, 1 seed)

| Stage | Uptake | Pre | Post | Vanilla | Mem Δ |
|---|---|---|---|---|---|
| Stage 0: Sense grounding | 8/8 | 0.00 | **0.88** | 0.00 | +2,161,672B |
| Stage 1: Transfer | 6/8 | 0.50 | **0.75** | 0.00 | +3,240B |
| Stage 2: Scope control | 8/8 | 0.62 | **0.69** | 0.00 | +13,472B |
| Stage 3: Dialog | 4/4 | 0.38 | **0.38** | 0.00 | +2,751B |
| Stage 4: Consolidation | 10/10 | 0.80 | **0.90** | 0.00 | +14,778B |
| Stage 5: Cross-domain | 6/6 | 0.58 | **0.92** | 0.00 | +10,504B |

## Results — mock CortexAgent driver (`--use-cortex-agent-mock`, 5 seeds)

| Stage | Post accuracy (mean ± std, n=5) |
|---|---|
| Stage 0 | 0.6250 ± 0.0000 |
| Stage 1 | 0.6250 ± 0.0000 |
| Stage 2 | 0.4375 ± 0.0000 |
| Stage 3 | 0.3750 ± 0.0000 |
| Stage 4 | 0.6000 ± 0.0000 |
| Stage 5 | 0.3333 ± 0.0000 |

Vanilla: 0.00 on every stage (verified answering, e.g. `'user profile'`
against corrected-sense probes — a genuine measured zero, not a harness gap).

## Comparison against the pre-removal claims

The 2026-06-30/07-01 logs claimed (real driver, semantic): Stage 2 scope
**1.00**, Stage 5 scope **1.00** (after two-slot `_SCOPE_TEACHING`),
Stage 5 retention 1.00. Post-removal, the same configuration measures
Stage 2 **0.69** and Stage 5 **0.92** (aggregate per-stage accuracy; the old
logs reported per-category scope/retention numbers, so the comparison is
directional rather than exact — the aggregate mixes both categories).

Reading:

- **Stage 2 lost ~0.3.** A substantial share of the claimed scope-control
  ability was the harness teaching to the test and biasing probes toward
  their expected answers. 0.69 is the honest capability.
- **Stage 5 held up well (0.92).** The scope-slot reranker + residual wiring
  earn most of their cross-domain claim legitimately; the leakage was
  responsible for roughly the last tenth.
- **Stages 0/1/4 are essentially unchanged** (0.88/0.75/0.90) — consistent
  with the leakage being scope-specific.
- **Vanilla = 0.00 everywhere** confirms all measured accuracy is
  attributable to learning, not LM priors leaking through the probe format.

## Methodology notes

1. **Mock-path seed variance is exactly zero.** Five seeds produce identical
   scores: the deterministic mock/shim answer ranking fully dominates; the
   seeded policy-head randomness never reaches the final answer. Multi-seed
   statistics only become informative on paths with real stochasticity
   (real-driver multi-seed runs are the follow-up — single real-driver seed
   here is a known limitation, flagged by the harness's own warning).
2. Mock and real driver are not comparable to each other (mock has zero
   semantics by design); each is a baseline for its own track.
3. Historical stage tables before 2026-06-30 are additionally invalidated by
   the scope-slot reranker bugs (see
   `2026-06-30_scope_slot_reranker_fix.md`); this log supersedes them as the
   reference point.
