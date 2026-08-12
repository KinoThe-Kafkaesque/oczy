# R18 Consolidation-as-Distillation — Implementation & First Runs

**Date:** 2026-07-09
**Session:** `2026-07-09T19-48-11-983Z` ("Reschedule queue to resume int8")

## Goal

Implement the Research/18 consolidation-as-distillation experiment, move
autoresearch from the experiment orchestrator to the new module, and get
initial baseline runs on Qwen2.5-0.5B with LoRA.

## Method

- Created `src/oczy/experiments/consolidation_distillation.py` (stub → 17KB
  real module implementing teacher-student hidden-state distillation with
  LoRA adapters).
- Wired `autoresearch.sh` to invoke `consolidation_distillation` instead of
  `experiment_orchestrator`.
- Bumped autoresearch to segment 10 (codebase-knowledge-recall), metric
  `distill_delta_holdout`.
- Used Qwen2.5-0.5B-Instruct, LoRA rank 2, 1 seed, 1 max_step.

## Results

| Run | Result |
|-----|--------|
| #200 | FAILED — eval_guard blocked commit (protected eval/ assets changed) |
| #201 | Stub baseline logged (`distill_delta_holdout=0.0`) |
| #202 | Real implementation ran (~220s). Holdout acc 0.0; `teacher_dev_delta ~0.176` below validity gate of 0.2 |

## Conclusion

The teacher validity gate failed at ~0.176 against the 0.2 threshold. This
was later confirmed by dedicated mechanism diagnostics on 2026-07-11
(`2026-07-11_r18_mechanism_diagnostics.json`) and a 5-seed diagnostic
(`2026-07-11_r18_five_seed_diagnostic.json`). The LoRA implementation was
functional but the teacher model could not clear the gate, blocking any
H-DISTILL verdict.

## Concurrent infrastructure work

- **Numba CPU kernel acceleration** (`62ab18e`): Wired Numba JIT kernels into
  `LMPlasticCortex` character-level RNN training loops.
- **Kaggle research compute workflow** (`6dee16b`): Added guarded Kaggle
  infrastructure with CPU/T4/P100/L4 smoke test configs.
- **INT8 rescheduling**: The session title "Reschedule queue to resume int8"
  indicates the R20 INT8 campaign was being planned a full week before its
  July 15-16 execution. The July 9 subagents produced the complete design
  for the staged orchestrator, runtime manifest v2, offline wheel/archive
  bootstraps, and Kaggle submission pacing.
