# R18 amended run — frontier-teacher 5-seed, stage_0_grounding

**Date:** 2026-08-06 (local run, background, ~27.7 min wall)
**Condition:** REGISTERED AMENDMENT A1 (`research/18-consolidation-as-distillation.md`)
— dev-gate teacher = OpenRouter `deepseek/deepseek-v4-flash-0731` (provider
pinned to DeepSeek, no fallback), `--teacher openrouter`. Student, LoRA
distillation, metric, `salt="v2"` split, and the `>= 0.2` gate unchanged.
**Command:** `python -m oczy.experiments.consolidation_distillation
--teacher openrouter --stage stage_0_grounding --seeds 5 --max-steps 10`
(offline local organ: HF_HUB_OFFLINE=1; teacher seed=0, temperature 0,
reasoning off, unbound max_tokens).
**Log:** `2026-08-06_r18_openrouter_5seed_stage0.log` (this dir).
**Status:** **HUMAN-ADJUDICATED — TESTED-PARTIAL with gate resolved (2026-08-06).**

## Results (per seed)

| seed | teacher_dev_delta | vanilla_holdout | lora_holdout | distill_delta_holdout | specificity_delta | wall_s |
|---|---|---|---|---|---|---|
| 0 | 0.4706 | 0.0 | 0.3333 | **0.3333** | 0.0 | 321.7 |
| 1 | 0.5294 | 0.0 | 0.3333 | **0.3333** | 0.0 | 336.2 |
| 2 | 0.4706 | 0.0 | 0.0 | **0.0** | 0.0435 | 344.3 |
| 3 | 0.4706 | 0.0 | 0.3333 | **0.3333** | 0.0 | 315.2 |
| 4 | 0.5294 | 0.0 | 0.3333 | **0.3333** | 0.0870 | 346.7 |

Headline: mean `distill_delta_holdout` = **0.2667**, 95% CI **[0.136, 0.397]**
(4/5 positive, seed 2 null); `distill_specificity_delta` mean 0.0261, CI
[-0.008, 0.060]. Persistent footprint constant across seeds (17.699 MB, span
46 B) — deterministic bounded state.

## Gate: the blocker is removed

- **teacher_dev_delta: {0.4706, 0.5294, 0.4706, 0.4706, 0.5294}, mean 0.4941 —
  cleared on all 5 seeds** (registered gate `>= 0.2000`).
- Prior 0.5B-prefix teacher: 0.1765 on every seed (gate failed; all original
  reruns retired). The expressivity-ceiling blocker documented in
  `CURRENT_STATE.md` / mechanism diagnosis `33169cc` no longer holds under
  Amendment A1.
- Non-determinism note: temperature 0 + seed 0 still gives 8/17 vs 9/17
  across calls (provider-level); both clear the gate comfortably.

## Scientific adjudication

**Human verdict (2026-08-06): TESTED-PARTIAL with gate resolved.** The amended
condition makes the H-DISTILL evidence admissible because the required teacher
admission gate now passes. The student effect reproduces the pre-amendment
diagnostic numbers exactly (mean 0.2667, bimodal
{0.3333,0.3333,0.0,0.3333,0.3333}, 4/5 positive, CI95 lower bound 0.136 > 0),
so there is an admissible distillation signal, but it remains unreliable in
1/5 seeds and does not receive full H-DISTILL acceptance. The *distillation
signal itself is unchanged* (still the local 0.5B prefix logits); only the gate
that blocked adjudication was substituted. The original local-teacher
condition remains BLOCKED and is not relabeled.

## Cost

~85 teacher gate calls (~17/seed) ≈ 4-5k tokens ≈ **~$0.0005** (DeepSeek-V4
Flash listed $0.09/M in, $0.18/M out; measured ~5.2e-6 USD/call). Well under
the approved $5 ceiling. Dominant cost was local CPU: 27.7 min total.

## Constraints honored

eval/v2 untouched; no holdout access by the teacher (vanilla/holdout scoring
all local, offline); no pre-registered text rewritten (Amendment A1 appended,
human-authorized); retrieval stays the mandatory baseline; teacher results
never labeled as metabolism. No episode-ID-conditioned code.
