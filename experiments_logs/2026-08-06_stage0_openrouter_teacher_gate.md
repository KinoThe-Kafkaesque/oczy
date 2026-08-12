# Stage-0 OpenRouter teacher gate check (R18 de-block, §4a first check)

**Date:** 2026-08-06 (local diagnostic)
**Status:** DIAGNOSTIC — not a scientific verdict, not an experiment run.
**Related:** `notes/2026-07-26_r18-r19_deblock_proposal.md` §4a; R18 teacher gate.
**Wiring added this day:** `src/oczy/lm/openrouter_teacher.py`,
`scripts/teacher_validity_check.py`, `--teacher {local,openrouter}` in
`src/oczy/experiments/consolidation_distillation.py`.

## Purpose

Cheap first check from the de-block proposal: does the **OpenRouter frontier
teacher** (`deepseek/deepseek-v4-flash-0731`, provider pinned to DeepSeek,
no cross-provider fallback) clear R18's registered admission criterion
`teacher_dev_delta >= 0.2` on the same stage-0 dev facts, in the same role
that the 0.5B-prefix teacher failed (recorded 0.1765 on every seed, gate
failed, all reruns retired)?

## Method

- Instrument: `scripts/teacher_validity_check.py` (reads eval/v2 dev split
  only; no holdout access; eval/v2 untouched).
- Stage: `stage_0_grounding`, dev split via `split_probes(salt="v2")` — the
  exact split R18 used.
- Teacher: `deepseek/deepseek-v4-flash-0731` @ OpenRouter, `provider.only=["DeepSeek"]`,
  temperature 0.0, seed 0, deterministic; **reasoning pass suppressed**
  (`reasoning.enabled=false` — required because the V4-Flash reasoning pass
  consumed the whole token budget and returned empty content otherwise).
  **Output length unbound** (`max_tokens` omitted from the request -> provider
  default max output): an artificial cap truncates answers and would corrupt
  later co-learning relabeling. A 32-token cap was the originally-wired bug.
- Prompt: context-first system — "Consider the definition given in the user
  message, then respond to the user's request using that definition in a
  single short sentence." with the correction follow by the probe (chat
  analog of R18's reserved-position prefix). Fact-agnostic template; not
  keyed to any episode.
- Scorer: `oczy.eval_v2.scoring.probe_matches` (unchanged). Empty answers
  are counted as misses (guarded in the check script).
- Baseline mode: `--vanilla teacher` — vanilla = teacher answering with NO
  correction. `teacher_dev_delta = (with_correction - no_correction)/N`.

## Result (stage_0_grounding, full dev split)

| quantity | value |
|---|---|
| dev probes | 17 |
| teacher correct (with correction) | 9/17 = 0.5294 |
| teacher correct (no correction) | 0/17 = 0.0000 |
| **teacher_dev_delta** | **0.5294** |
| registered gate | >= 0.2000 |
| **GATE CLEARED** | **YES** |
| R18 recorded baseline (0.5B teacher) | 0.1765 (failed on all seeds) |
| API tokens (38 requests) | 1,672 (cost approx < $0.01) |
| provider actually used | DeepSeek (confirmed in response metadata) |
| max_tokens | unbound (omitted; provider default) — re-ran 2026-08-06 after removing the 32-token cap; result identical |

## Interpretation

- The teacher expressivity ceiling that blocked R18 is **not present** for
  the frontier teacher: it clears the gate at 2.6x the threshold. This is the
  first, cheapest hypothesis from the de-block proposal and it passes.
- Design notes surfaced by the check (recorded for the amendment):
  1. DeepSeek V4-Flash runs a proprietary reasoning pass by default; without
     suppression the answer is empty (budget eaten by `reasoning_tokens`) and
     the eval's substring fallback would silently mis-score. The wiring
     suppresses reasoning and guards empty answers as misses.
  2. With-correction still misses 8/17 by phrasing (e.g. "The jam is
     cleared."), not by fact knowledge. The gate is comfortably cleared
     regardless; the R18 amendment may optionally tighten the teacher prompt
     for higher `teacher_abs`, but the gate is the recorded admission bar and
     is met.
- Capability-floor caveat stands: the *student* (0.5B + LoRA) floor is a
  separate question (see proposal §4 item 4 and S4.3). Approving the frontier
  teacher does not clear the student.

## Provenance / reproducibility

- Key: Prime Agent's OpenRouter credential (`~/.prime/agent/auth.json`,
  `openrouter.key`); never committed.
- Model + pinning versioned in `OpenRouterTeacherConfig` defaults; every ASI
  line carries `teacher_model`, `teacher_provider_only`,
  `teacher_vanilla_source`, `teacher_dev_probes`, `teacher_abs`.
- Command: `python scripts/teacher_validity_check.py --stage stage_0_grounding --vanilla teacher`

## Constraints honored

eval/v2 unchanged; no holdout access; no pre-registered spec edited; no
student training; retrieval stays the mandatory baseline; gate and metric
definitions untouched.
