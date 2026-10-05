# R20 S3 sign-off condition — decode-change comparability (2026-10-05)

**Gate:** precondition on sign-off item S3 (not a gate in the G1–G6 sequence).
**Classification:** READ-ONLY COMPARABILITY AUDIT.
**Authorization:** user authorization relayed by the manager on 2026-10-04,
kanban card `t_5a7b48de`.

S3 (the uncommitted `organ.py` change decoding generated ids with
`skip_special_tokens=True`, scalar and batch paths) was approved **adopt with a
version bump**, conditional on first demonstrating exactly which prior recorded
scores the decode change alters, and on recording any comparability break as a
limitation with the change quarantined to v2-lineage runs.

This entry discharges that condition.

## Scope and method

The change under audit (`src/oczy/experiments/meta_cortex/organ.py`):

- line 951, scalar `generate()`: `decode(generated_ids)` →
  `decode(generated_ids, skip_special_tokens=True)`
- lines 1081 and 1089, batch `generate_batch()`: the same substitution on both
  trimming branches.

Source of the affected prior scores:
[`2026-09-11_r20_dev_output_path.json`](2026-09-11_r20_dev_output_path.json),
whole-file SHA-256
`e6d3c0ee85c769d185ab6affaff6b0c4f1d8096720f0e64f226ee1bee7841fdf`.

That artifact is the **only** recorded R20 artifact that retains raw generated
answer text, and it is the only one that stores both decodes per row
(`generated` = old decode, `content_only_counterfactual` = special-token-stripped
text). Every other recorded R20 artifact stores scores and counts only, with no
answer text and no decode-mode field, so no decode-mode attribution is possible
from them in either direction.

Method: for each recorded probe row, compare `generated` against
`content_only_counterfactual` for (a) text equality, (b) whether the old text is
exactly the new text with special tokens removed, and (c) the recorded
correctness under each decode. No model was run, no score was recomputed, no
file was modified.

## Commands and results

```bash
.venv/bin/python -c "<per-row old-vs-new decode comparison over the 68 recorded rows>"
```

Exit code 0. Per run, all rows are the same three conditions (7 same-rule probes
with no context, 7 with the full teaching transcript, 3 oracle probes):

| Run | Rows | Text changed by decode | Rows containing special tokens | Correct under old decode | Correct under new decode | Score delta |
|---|---:|---:|---:|---:|---:|---:|
| `articulation-before` | 17 | 11 | 11 | 0 | 0 | **0** |
| `articulation-after` | 17 | 0 | 0 | 0 | 0 | **0** |
| `articulation-v3` | 17 | 0 | 0 | 1 | 1 | **0** |
| `articulation-v4` | 17 | 0 | 0 | 2 | 2 | **0** |
| **total** | **68** | **11** | **11** | **3** | **3** | **0** |

In all 68 rows, `generated` with special tokens removed equals
`content_only_counterfactual` exactly (68/68), so the new decode is precisely the
old decode with special tokens dropped — nothing else changes.

## Which prior recorded scores the decode change alters

**None. The score delta is 0/68 rows, 0 across all four recorded runs.**

The change alters the answer *text* of 11 of 68 rows (all 11 in
`articulation-before`) by removing one special token per row (10–13 characters
per row). It moves no recorded score, for two independent reasons visible in the
rows:

1. All 11 affected rows were already incorrect under the old decode, and the
   stripped text is still not the expected token — the special token was trailing
   noise in an otherwise wrong answer, never the whole answer.
2. The 3 correct rows across all four runs (`articulation-v3` 1, `articulation-v4`
   2) contain no special tokens, so their text is byte-identical under both
   decodes.

Concretely, per condition, all recorded scores are unchanged:

| Condition | old decode | new decode |
|---|---|---|
| no context | 0/7 | 0/7 |
| teaching transcript (retrieval comparator) | 0/7 | 0/7 |
| oracle context | 0/3 | 0/3 |

## Verdict and quarantine

**S3 condition satisfied. The decode change is adopted with a version bump, and
is quarantined to v2-lineage runs.**

Comparability breaks to record as limitations:

1. **Text-level, not score-level.** The change is not score-neutral in general —
   it is score-neutral on every recorded score because no recorded score was
   decided by a special token. A future probe whose expected answer *is* a
   special token spelling, or a corpus where extra prose is stripped, would move.
   The quarantine exists for that reason.
2. **Quarantine.** The change applies only inside the v2 lineage
   (`meta_cortex/v3`). No v1-lineage score may be re-read, restated or compared
   through the new decoder. v1 recorded scores stay as recorded.
3. **The `articulation-before` run is the only decode-sensitive record.** Its 11
   text changes are recorded here so that a later reader comparing raw strings
   between the two decoders is not surprised by them.
4. **No score was recomputed or re-derived.** This audit replays text that was
   recorded on 2026-09-11 under the historical organ identity
   `a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`; it
   produced no model call and no optimizer step.

## Limitations (restated, not resolved)

These are carried forward from earlier records and are **not** addressed by this
audit:

- **R20 local reproduction: four nonzero state hashes still differ.** The
  2026-09-11 local reproduction matched all scores and score-vector hashes but
  not the untrained, trained, feedback-shuffled and state-swapped state hashes.
  Exact numerical-state reproduction remains unresolved; this audit does not
  repair it and inherits the limitation.
- **eval-v2 A2 and C repairs remain unsigned.** The sense/alias contract (A2)
  and tool-output scoring (C) are still unapplied. Nothing here changes
  `eval/`, `src/oczy/eval_v2/` or the tool curriculum.
- **758 flip-target mutation checks are still vacuous by construction.** They
  are split from the 688 discriminating delete-a-fact checks and reported
  separately, but a flipped target cannot fail once its certificate verifies.
  The real guarantee remains the independent derivation disagree-check.
- **No capability claim.** Correcting a transport defect is not evidence that
  the organ can express a task, and the sampled oracle remains 2/3 on three
  tuning tasks.