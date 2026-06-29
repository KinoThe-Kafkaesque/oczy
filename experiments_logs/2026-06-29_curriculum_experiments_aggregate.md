# Curriculum Experiments — Aggregate Orchestrator Report

Date: 2026-06-29  
Branch: `autoresearch/session-20260625`  
Orchestrator: `src/oczy/experiments/experiment_orchestrator.py`  
Benchmark entrypoint: `bash autoresearch.sh`

## Summary

All seven curriculum experiment modules described in `experiments/` are now
implemented, tested, and runnable from a single orchestrator. The current
aggregate status is:

- **Accepted: 5 / 7**
- **Primary metric: `experiments_accepted_count=5` (higher is better)**
- **Full experiments test suite: 220 passed, 6 warnings**

| # | Module | Status | Primary metric |
|---|---|---|---|
| 01 | `correction_competence_v2.py` | accepted | `v2_desaturation_count=5`, `v2_discrimination=1` |
| 02 | `kv_slot_injection.py` | accepted | `kv_slot_rank1_count=3/3` |
| 03 | `layer_l_probe.py` | refuted / honest null | `layer_l_silhouette_gap=-0.057` |
| 04 | `scope_selectivity_stressor.py` | accepted | `scope_selectivity_index=0.625` |
| 05 | `metabolism_loop.py` | null on first surface | `metabolism_drift_delta=0`, `compounding_index=1.0` |
| 06 | `bounded_growth/bounded_growth_eval.py` | accepted | `bounded_growth_m1_ratio=0.072706` |
| 07 | `conversation_world_model.py` | accepted | `marker_free_uptake_gap=1.0` |

## New files added

- `src/oczy/experiments/scope_selectivity_stressor.py` + tests
- `src/oczy/experiments/experiment_orchestrator.py` + tests

## Test-suite fixes

- Primary-metric emission in `conversation_world_model.py` now consistently
  prints `METRIC marker_free_uptake_gap=...` even on real-driver failure paths.
- Heavy-import assertions in multiple test files now use subprocess isolation
  so they do not flake when llama_cpp is already loaded by earlier tests.

## Honest nulls and next directions

- **Experiment 03 (Layer-L hidden extraction)** is refuted: final-layer mean-pool
  outperforms all mid-layer last-token representations for this cortex/config.
- **Experiment 05 (Metabolism loop closure)** is currently an honest null. The
  cvec-surface chosen does not drive domain-word drift, even though repeated
  corrections accumulate additively (`compounding_index=1.0`). Remaining
  options: try diverse correction phrasings, a stronger observation budget, or
  a different probe target.

To push `experiments_accepted_count` to 6, the most promising remaining lever is
revisiting Experiment 05; if that remains null, the next step is to document the
refuted/negative results and reassess the research graph.
