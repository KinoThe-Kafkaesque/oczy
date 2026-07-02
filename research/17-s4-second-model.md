# 17 — Second-model generalization (Sprint 4 / S4.3)

**Pre-registered 2026-07-02** (human-approved sprint setup).
Agents MUST NOT edit this spec. Depends on research/11 (and research/12 if
H-KVCONTENT was accepted — the content channel under test is whichever the
Sprint-2 verdicts selected).

## Problem

Every architecture result so far comes from one substrate at a time (LFM2.5
on the legacy path, Qwen2.5-0.5B on HF). Single-model results say nothing
about the *architecture* generalizing versus exploiting one model's quirks —
S1.4 already showed the two models disagree about layer geometry.

## Hypothesis

**H-GEN:** the minimal metabolism loop's held-out effect (research/11's
`loop_delta_holdout > 0`, compounding ρ ≥ 0.6) reproduces on a second,
architecturally distinct small model with NO per-model tuning beyond the
substrate adapter.

## Second model (selection rule fixed now)

The already-benchmarked fallback `Qwen/Qwen2.5-1.5B-Instruct` is EXCLUDED as
the second model (same family, same tokenizer — weak generalization
evidence). Choose the fastest cached-or-downloadable model satisfying ALL of:
plain decoder-only transformer with per-layer (k,v) cache (verified by
`check_kv_cache.py`), different model family from Qwen, instruct-tuned,
≤ 2B params, ≤ 300 ms/tok CPU float32 on this host
(`bench_hf_cpu.py`). Candidates to try in order: TinyLlama-1.1B-Chat
(known EOS fragility — re-test with the chat template; if unusable, record
why), `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `google/gemma-2-2b-it`. The
choice and benchmark numbers are recorded before any loop run.

## Protocol

Exactly research/11's protocol (stage 0, `split_probes(fraction=0.3,
salt="v2")`, K ∈ {0,1,2,4,N}, ≥5 seeds — fallback 3, reported), with the
Sprint-2-selected content channel, on the second model. NO hyperparameter
changes from the Qwen run: same prefix budget, same alpha, same gate. If the
second model needs a chat template for coherent output, apply it identically
in the vanilla column.

## Primary metrics & acceptance

- `gen_delta_holdout` (= research/11's `loop_delta_holdout` on model 2)
- `gen_compounding_rho` (= `loop_compounding_rho` on model 2)

- **Accept H-GEN:** `gen_delta_holdout > 0` with 95% CI excluding 0 AND
  `gen_compounding_rho ≥ 0.6`.
- **Refute:** either fails — recorded as "architecture does not yet
  generalize"; per-model tuning that rescues it belongs in a NEW
  pre-registered spec, not this one.
- **Validity gates:** model-2 vanilla holdout < 0.5 (else INVALID); research/11
  must have ACCEPTED H-LOOP on Qwen (else this experiment is BLOCKED — there
  is nothing to generalize).

## Pre-registered secondaries (exploratory only)

1. Effect-size comparison model 1 vs model 2 (ratio of deltas).
2. Drift triple on model 2.
3. ms/tok and memory cost table for the adapter on model 2.

## Reporting

Model-2 selection record (candidates tried, benchmark numbers, exclusions),
full research/11-style tables for model 2, side-by-side with the Qwen
numbers; log to `experiments_logs/<date>_s4_3_second_model.md` quoting this
spec.
