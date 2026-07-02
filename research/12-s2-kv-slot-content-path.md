# 12 — KV slots replace the articulation prefix (Sprint 2 / S2.2)

**Pre-registered 2026-07-02** (human-approved sprint setup, before implementation).
Agents running this experiment MUST NOT edit this spec; deviations are reported
as deviations. Depends on research/11 (S2.1 minimal organism) being merged.

## Problem

S2.1's content channel is a token-burning articulation prefix: consolidated
facts occupy visible prompt tokens on every forward pass. S1.3
(`research/09`, log `2026-07-01_s1_3_hf_kv_slot_injection.md`) established
that KV entries spliced **immediately pre-blank** are rank-for-rank equivalent
to a text prefix at zero visible-token cost (and that 2× K/V norm scaling is
catastrophic — scaling is out of scope here). This experiment retires the
prefix from the minimal organism.

## Hypothesis

**H-KVCONTENT:** replacing the S2.1 consolidated articulation prefix with the
same content encoded once at consolidation time via `HFDriver.encode_kv` and
spliced pre-blank via `generate_with_kv` at answer time preserves the loop's
held-out behavior change with ZERO fact tokens in the visible prompt.

## Conditions (identical protocol to research/11: same stage, split, seeds, K=N)

- **C0** vanilla: bare `HFDriver`, no injection.
- **C1** prefix organism: the S2.1 configuration exactly (upper anchor).
- **C2** KV organism: identical organism, but `consolidate()` compiles the
  same content into KV entries (encoded once, cached on the organism) and
  `answer()` uses `generate_with_kv` with pre-blank splice. No
  `set_articulation_prefix` call anywhere in the C2 path. K/V norms unscaled
  (1×).

**Prompt-token audit (mandatory):** for every C2 probe, the visible prompt
token count must equal C0's for the same probe. Any mismatch invalidates C2.

## Primary metrics & acceptance

1. `kv_effect_delta` = mean over seeds of [C2 holdout accuracy − C0 holdout
   accuracy] at K=N.
2. `kv_parity_delta` = mean over seeds of [C2 − C1 holdout accuracy] at K=N.

- **Accept H-KVCONTENT:** `kv_effect_delta > 0` with 95% CI excluding 0, AND
  `kv_parity_delta >= −0.05` (non-inferiority vs the prefix).
- **Refute:** either fails.

**Validity gates:** research/11's vanilla gate applies; if research/11
REFUTED H-LOOP (C1 itself produces no effect), this experiment is reported as
BLOCKED, not run to a fake verdict.

## Pre-registered secondary analyses (exploratory only)

1. Per-answer latency: C2 vs C1 vs C0 (ms, mean over holdout probes).
2. KV bytes vs prefix bytes in `memory_bytes` accounting.
3. Rank tables (target-token rank at blank) on the 3 lane_02 facts under the
   organism's consolidated state, C1 vs C2 — continuity check against S1.3.

## Reporting

Per-seed table for C0/C1/C2; prompt-token audit result; model id; exact
commands; log to `experiments_logs/2026-07-02_s2_2_kv_content_path.md`
quoting this spec.

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
