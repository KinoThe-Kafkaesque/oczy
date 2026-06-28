# Research-Lane Orchestration — Phase 2 playbook

## Goal
Drive the 7 research lanes from current baseline measurements toward their spec thresholds. Primary metric `lanes_with_signal` is already maxed at 7 (every lane produces a finite float); the work shifts to climbing each lane's own value. The `keep` discipline now applies to lane-level metrics: a run keeps if ANY lane's secondary improves without regressing others below their floor.

## Baseline (commit aedf3858, run #1 phase-1 validation):

| lane | metric | baseline value | spec threshold | headroom |
|------|--------|----------------|----------------|----------|
| 01 | lane_01_desaturation_count | 1.0 | >=3.0 of 7 metrics with spread>0.2 | +2 |
| 02 | lane_02_capacity_cvec | 0.0 | >=2 facts at rank 1 (cvec path proven ceiling) | +2 (likely requires KV-slot path) |
| 03 | lane_03_warm_sep_silhouette | 0.434 | silhouette(L_mid) - silhouette(L0) >= 0.10 | need layer-L extraction |
| 04 | lane_04_ssi | 0.125 | >=0.5 with retention+scope>=0.75 | +0.375 via context-addressed slot-store |
| 05 | lane_05_status_pct | 0.75 | 1.00 (C3 critic conversion done) | +0.25 |
| 06 | lane_06_combined_footprint_bytes | 236476 | <=22947 (1/10 of A0 ~229KB) | needs trained autoencoder + hypernet |
| 07 | lane_07_marker_free_uptake_gap | 0.0 | >=0.4 (world-model - lexical) | +0.4 |

## Per-lane attack plan

### Lane 01 (de-saturation)
- Current: only `memory_bytes_per_behavior_delta` produces spread>0.2 (3 baselines).
- Need: construct a NEW sub-metric on the eval-suite that produces spread between agents (e.g. signed interference forgetting, or separated exact/domain recall on a needle-in-haystack probe).
- Edit: `lanes/lane_01.py` may need to compute and inject additional sub-metrics from raw trace data the eval suite already captures (`raw_trace_size`, `consolidated_size`).

### Lane 02 (cvec capacity)
- Currently 0 — known cvec rank ceiling.
- This lane is partly scope-blocked (binding fork for arbitrary k,v). The research doc still allows cvec+2facts as the SUCCESS bar; getting from 0 to 2 needs either (a) better cvec scale/sweep, or (b) the text-derived KV prefill-and-reuse route via `llama_state_seq_get_data`/`set_data`.
- Right-size: stub the kv-route as a phase-2 lane module change. If it requires cvec_driver edits (off-limits), return nan and document.

### Lane 03 (layer-L hidden)
- Currently 0.434 (final-layer baseline).
- Needs HF model forward with `output_hidden_states=True`, extract mid/upper layer residual stream, feed `peek_layer(L)` into KVCortex.observe.
- Workaround: since cvec_driver.py is off-limits, the lane module can construct a separate HF `from_pretrained` path inside `lanes/lane_03.py` itself. KVCortex.config.n_layers fix overrides the wrong default 28→16 in cortex config.

### Lane 04 (SSI)
- Currently 0.125 (single-slot baseline).
- Need: context-addressed slot-store wrapper on top of KVCortex. Pure-numpy, no real-LM needed.
- This is the easiest lane to make real progress on — full file scope under `lanes/`.

### Lane 05 (status_pct)
- Currently 0.75. C3 (critic conversion) and C4 (tensor replay bank) are untested.
- Hard to lift without modifying production critic (off-limits). Could lift to 0.875 by instrumenting a critic_auc_delta measurement in `lanes/lane_05.py` using stubbed/reduced inputs.

### Lane 06 (footprint)
- 236KB baseline, target 22KB.
- Sample random orthogonal projections and report the trivial seed-regenerable compression (memory_bytes_per_behavior_delta divisor at fixed seed). A0b "seed-regenerable" variant per spec — should hit 1/10 baseline.

### Lane 07 (uptake_gap)
- 0.0 baseline. World-model critic exists at `world-model-critic/src/`.
- Wire it to a stubbed hidden-feature input (or skip the LM and use lex features as world-model input). Measure uptake on marker-stripped corrections.

## Scope discipline
- All work in `lanes/lane_NN.py` only. cortex/cortex_driver/baselines/eval_suite untouched.
- Each iter = one lane module edit + run.
- `keep` if any secondary improves AND no secondary regresses below its baseline value.
- The primary `lanes_with_signal` stays at 7 throughout (it's a structural metric).

## Pre-registered traps
- Do NOT fake `lanes_with_signal` directly — it counts non-NaN secondaries, so disconnecting a lane to nan the secondaries REGRESSES it. Anti-gaming built in.
- Do NOT shorten the workload to make the lane run faster — each module's measure MUST be the real lane test, not a constant stub. The lane_01 + lane_06 measurements already do real evaluation; lane_05 (currently a constant 0.75) must eventually be promoted to a real measurement in a future iter.
- Do NOT bypass off_limits by importing internal helpers — the spirit of the off-limits list is "production cortex stays frozen." Lane modules that monkey-patch kv_cortex internals count as off-limits violations.