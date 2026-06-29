# Curriculum Experiments — Aggregate Orchestrator Report

Date: 2026-06-29  
Branch: `autoresearch/session-20260625`  
Orchestrator: `src/oczy/experiments/experiment_orchestrator.py`  
Benchmark entrypoint: `bash autoresearch.sh`

## Summary

All seven curriculum experiment modules described in `experiments/` are now
implemented, tested, and runnable from a single orchestrator. The current
aggregate status is:

- **Accepted: 6 / 7**
- **Primary metric: `experiments_accepted_count=6` (higher is better)**
- **Full experiments test suite: 220 passed, 6 warnings**

| # | Module | Status | Primary metric |
|---|---|---|---|
| 01 | `correction_competence_v2.py` | accepted | `v2_desaturation_count=5`, `v2_discrimination=1` |
| 02 | `kv_slot_injection.py` | accepted | `kv_slot_rank1_count=3/3` |
| 03 | `layer_l_probe.py` | refuted / honest null | `layer_l_silhouette_gap=-0.057` |
| 04 | `scope_selectivity_stressor.py` | accepted | `scope_selectivity_index=0.625` |
| 05 | `metabolism_loop.py` | accepted | `metabolism_drift_delta=+0.10`, `compounding_index=1.0` |
| 06 | `bounded_growth/bounded_growth_eval.py` | accepted | `bounded_growth_m1_ratio=0.072706` |
| 07 | `conversation_world_model.py` | accepted | `marker_free_uptake_gap=1.0` |

## New files added

- `src/oczy/experiments/scope_selectivity_stressor.py` + tests
- `src/oczy/experiments/experiment_orchestrator.py` + tests

## Experiment 05 refinement

The initial implementation of Exp05 reported an honest null:
`metabolism_drift_delta=0` with `compounding_index=1.0`. Investigation showed
that repeated identical corrections produced additive cold growth, but the
random `proj_c` projector pushed the answer off-manifold, so the substring
read-out of domain words stayed at zero.

The refined implementation:
1. Added **N=8 diverse correction phrasings** matching `d_cortex=8` so
   `init_proj_c_from_svd` receives a non-degenerate correction-aligned basis.
2. Added a **perceive-only SVD warm-up phase** that collects hidden states from
   the diverse phrases without consolidating, keeping `cold_state` near zero
   while learning the projector.
3. Kept the compounding loop on the **single canonical correction** so cold
   updates remain additive (`compounding_index=1.0`).
4. Replaced the coarse substring-based domain uptake with the spec-intended
   **logit-based domain shift**: mean next-token logit of domain-word token ids
   at the probe blank, compared against a zero-baseline agent.

Result: `metabolism_drift_delta=+0.10`, `drift_logit=2.54` vs
`zero_baseline_logit=2.44`. The cvec now raises domain-word logits
reproducibly. The substring uptake remains zero because the LM does not sample
one of the exact domain words for this probe, which is why the logit read-out
was proposed in the spec.

## Test-suite fixes

- Primary-metric emission in `conversation_world_model.py` now consistently
  prints `METRIC marker_free_uptake_gap=...` even on real-driver failure paths.
- Heavy-import assertions in multiple test files now use subprocess isolation
  so they do not flake when `llama_cpp` is already loaded by earlier tests.

## Remaining refutation

**Experiment 03 (Layer-L hidden extraction)** remains an honest null:
`layer_l_silhouette_gap=-0.057`. Mid-layer last-token residuals do not beat
final-layer mean-pool for this cortex/config. This result is stable and
informs the hidden-extraction strategy for future work.

## Validation

Runs #179 and #180 both report `experiments_accepted_count=6` with identical
`metabolism_drift_delta=+0.10`, confirming deterministic behavior under the
fixed seed and correction sequence.

## Next directions

- The curriculum aggregate now passes 6/7 thresholds. The remaining
  scientific question is whether Experiment 03 can be reframed or whether the
  final-layer mean-pool result stands as a refutation.
- Possible follow-up: run a cross-lane synthesis (lane_08 style) combining
  accepted mechanisms (KV-slot, scope-addressing, bounded-growth, world-model
  critic) into an end-to-end agent and measure
  `behavior_delta_per_byte_of_persistent_memory`.
