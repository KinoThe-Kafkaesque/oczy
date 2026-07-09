# Research 18 Implementation Context

## Goal
Implement `src/oczy/experiments/consolidation_distillation.py` as the full, runnable research-18 experiment (consolidation as context distillation). The active autoresearch segment has primary metric `distill_delta_holdout` and benchmark `bash autoresearch.sh`.

## Source of truth
Read `research/18-consolidation-as-distillation.md` verbatim before editing. Do not deviate from the pre-registered design.

## Summary of research 18
- Substrate: `HFDriver` / `Qwen/Qwen2.5-0.5B-Instruct`, CPU float32, greedy eval.
- Base organism: `MinimalOrganism` (prefix channel, posture off).
- Teaching: research/11 protocol, stage 0, all N episodes, seed-shuffled.
- Consolidation: for each stored correction c, distill prefix-conditioned teacher behavior into a LoRA adapter (rank ≤8, attention projections) on the student = base model + LoRA, no prefix.
- Loss: token-level KL(student || teacher) on distillation prompts.
- Distillation prompts: episode's `initial_request` plus a fixed generic template list (e.g., "Q: {request}\nA:", imperative/question paraphrase frames). Eval `expected` strings and holdout probe texts must NOT appear in training/loss.
- After consolidation: prefix cleared, hippocampus traces deleted, organism = driver + LoRA only.
- Score: stage-0 HOLDOUT probes (repaired split, frozen scorer).
- Seeds: ≥5 (LoRA init + teaching order); fallback 3 if >15 min/seed.
- Validity gate: `teacher_dev_delta` = base+per-fact-prefix dev accuracy - vanilla dev accuracy ≥ 0.2.
- Primary metric: `distill_delta_holdout` = [holdout accuracy, LoRA-only, traces deleted] - [vanilla holdout accuracy], averaged over seeds.
- `distill_specificity_delta` = change on other stages' holdout probes + S2.3 control-word logit shift.
- Accept: `distill_delta_holdout > 0` with 95% CI excluding 0, `distill_specificity_delta ≥ -0.05`, trace deletion verified.

## Existing code to reuse
- `src/oczy/lm/hf_driver.py`: `HFDriver` with `load()`, `generate()`, `peek_embedding()`, `set_reserved_position()`, `clear_reserved_position()`, `encode_kv()`, `generate_with_kv()`.
- `src/oczy/experiments/organism_curriculum/dataset.py`: `Stage`, `Episode`, `Probe`, `build_curriculum()`, `load_stage()`, `split_probes()`.
- `src/oczy/eval_v2/scoring.py`: `probe_matches(answer, probe, episode)`.
- `src/oczy/experiments/minimal_loop.py`: reference for `_run_one_seed`, `_run_experiment`, using `HFDriver` and stage loading.
- `src/oczy/experiments/hf_kv_slot_experiment.py`: reference for HFDriver KV-slot usage.
- `src/oczy/experiments/eval_suite.py`: reference for snapshot/score scaffold.

## New CPU/Numba kernels
Use the Numba kernels in `plastic-cortex/src/plastic_cortex/_numba_kernels.py` and `lm_cortex.py` to accelerate any CPU-native RNN/surrogate components if applicable. The primary HFDriver/Qwen forward passes stay PyTorch CPU, but any trainable NumPy-only surrogate (e.g., a tiny LMPlasticCortex for distillation scaffolding) should use `_numba_kernels`.

## Implementation requirements
- Module: `src/oczy/experiments/consolidation_distillation.py`
- CLI: `python -m oczy.experiments.consolidation_distillation [--seeds N] [--dev] [--max-steps-per-fact N] [--lora-rank R]`
- Must emit `METRIC distill_delta_holdout=<mean>` (primary) and `METRIC distill_specificity_delta=<mean>`.
- Must emit ASI lines: `teacher_dev_delta`, `vanilla_holdout_acc`, `lora_holdout_acc`, `specificity_delta`, `persistent_bytes`, `seeds`, `lora_rank`, `distillation_steps`.
- Manually implement LoRA (no `peft` dependency). Add LoRA A/B to chosen attention projection weights (e.g., `q_proj`, `v_proj` or `o_proj`) and train only A/B.
- Distillation: for each correction, compute teacher logits on the distillation prompts with the per-fact prefix, then update student LoRA to match teacher via KL.
- Use dev split for hyperparameter tuning and holdout for final metric. `split_probes(stage, fraction=0.3, salt="v2")` returns dev_ids, holdout_ids.
- Delete traces after consolidation: prefix cleared, transient correction texts deleted, memory bytes measured before/after.
- Include vanilla baseline and retrieval/parametric baseline as comparators if feasible, but the primary metric is LoRA holdout delta.

## Output
Replace the stub `src/oczy/experiments/consolidation_distillation.py` with a full, runnable implementation. Do not touch `autoresearch.sh`, `eval/`, `research/`, `lanes/`, or other protected paths. The implementation should be safe to run under `bash autoresearch.sh` (600s timeout default).
