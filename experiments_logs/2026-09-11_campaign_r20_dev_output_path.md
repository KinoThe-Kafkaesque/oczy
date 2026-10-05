# R20 output-path diagnosis and approved DEV prompt amendments

**Date:** 2026-09-11. **Classification:** DEV DIAGNOSTIC ONLY.
No H-META-CORTEX verdict, candidate sign-off, remote submission, or meta-test
access. These local diagnostics do not replace historical v2 calibration.

Raw outputs, commands, identities, failed attempts and comparisons are retained
in [the JSON report](2026-09-11_r20_dev_output_path.json). The user approved
Amendments A then B in [APPROVAL.md](../experiments/r23.5-serialization-dev/APPROVAL.md).

## Identity blocker: isolated and bypass-free reproduction path implemented

All ten model artifact files and the exact historical runtime manifest match.
The organ's legacy parameter hash includes Hugging Face `_name_or_path`.
Changing only that configuration value from the local snapshot path to the
recorded Kaggle mount changes `2621e258…` to the required `a342431c…`;
restoring it restores the original hash. This isolates a path-sensitive
identity defect, not a weight mismatch.

The reproduction wrapper binds the verified model files to their historical
path in an offline, private mount namespace. It leaves the original source,
model, checkpoint and hash checks intact. See
[LOCAL_REPRODUCTION.md](../infrastructure/kaggle/LOCAL_REPRODUCTION.md).

A complete six-condition cell, development seed 0/evaluation seed 0/task 0,
then ran under clean source `a8c98d638209a8425b14a0f853e9fc46ae7da581`.
All scores, score-vector hashes and non-state-hash audit fields match the
historical cell. Model and checkpoint hashes are unchanged, optimizer steps
are zero, and traces are deleted. Four nonzero state hashes differ:
untrained, trained, feedback-shuffled and state-swapped. The exact comparison
therefore **fails**. CPU numerical differences are a possible explanation,
not a demonstrated cause. This reproduces one cell's zero scores, not the
full campaign or its exact internal numerical state.

The preceding full-shard attempt was deliberately stopped after 1,076.7 s
while replaying the required 20 no-update episodes; it produced no shard.
Its exit `-15` and log are preserved. The subsequent focused cell took
583.1 s and does not include those repeats. Neither attempt replaces a
calibration shard.

## Decoder defect: fixed, with a null score effect in the paired check

Scalar and batched generation returned the printable EOS spelling, such as
`left<|im_end|>`, to an exact text scorer. Decoding now omits special tokens.
Ordinary punctuation and extra prose are retained and still fail exact scoring.

The old and corrected decoders ran identical prompts for the first tuning
task in each frozen family: seven same-rule probes under no context, the same
seven with the full teaching transcript, and three existing oracle probes.
All 17 prompt hashes match; all 17 corrected strings equal the original
strings after special-token removal. Eleven originals contained special
tokens. Both runs score **0/7 no context, 0/7 teaching, 0/3 oracle**.
This fixes a real transport defect but does not explain the sampled zeros.

## Amendment A — response format only, meta_cortex/v3

An identical bare-answer system instruction is prepended to every public DEV
probe. The instruction remains first when a teaching transcript is supplied.
Original user messages, events, answers, splits, seeds, rules and scoring remain
unchanged. Oracle descriptions remain the original shorthand.

| Condition | Corrected decoder, v2 | Amendment A, v3 |
|---|---:|---:|
| No context | 0/7 | 0/7 |
| Teaching transcript / retrieval comparator | 0/7 | 0/7 |
| Oracle context | 0/3 | 1/3 |

The finite-state oracle changes from incorrect to exact `q2`. The other two
oracle families remain wrong. Teaching outputs still include prose, such as
`The transformed output for input nu is nurpnu.` instead of exact `nurpnu`.
No substring credit or answer extraction is applied.

V3 manifest: `5c944abc72fac57b29890da46d9a3b3807c551c5c2b0370c24d771c5616162b5`.
Execution: 35.9 s, exit 0.

## Amendment B — complete oracle semantics only, meta_cortex/v4

V4 inherits v3 and changes only the transformation oracle's rule description
to the approved full algorithm. Worked examples, operands and expected answers
remain unchanged. Hidden rule semantics appear only in oracle probes.

| Condition | Amendment A, v3 | Amendment B, v4 |
|---|---:|---:|
| No context | 0/7 | 0/7 |
| Teaching transcript / retrieval comparator | 0/7 | 0/7 |
| Oracle context | 1/3 | 2/3 |

The conditional-rule oracle now outputs exact `nurpgamma`. The contextual
mapping oracle remains wrong (`rise`, expected `left`), while the finite-state
oracle remains correct. Exactly one of 17 prompt hashes changes between v3
and v4; all 16 unchanged prompts produce identical outputs.

V4 manifest: `d58adb749f4112f2920f08e0dbe0ce5c6a1ee7711f562faee4e41072eccc8764`.
Execution: 41.4 s, exit 0, after v3 completed.

## Integrity and limits

- The materialized v3/v4 instruments contain only the original 90 training
  and 15 tuning tasks. Original public task bytes match the original definition
  manifest. The generated public catalog matches all 105 records exactly.
- All 916 probes receive the v3 instruction. V4 additionally changes only
  35 transformation-oracle descriptions: 9 substitution, 8 permutation,
  10 conditional, 8 composition. Tests compare the descriptions' operations
  with the actual generator function on all 18 public operands plus five
  boundary/branch examples per rule. No threshold changed.
- Every model run retains organ hash `a342431c…` before and after, greedy
  decoding, maximum 32 new tokens, empty soft bank and the same frozen exact
  scorer. No model or cortex optimization occurs in the articulation checks.
- The prompt checks use local uncommitted code on `e71268b`, with source hashes
  recorded. They are not a clean-source remote research campaign. The original
  v2 source/checkpoint/instrument and `eval/v2` manifest remain intact.
- Validation: 181 targeted tests passed, 5 real-model unit tests skipped;
  actual model behavior was separately exercised in the four recorded local
  articulation runs. Ruff and whitespace checks pass. The frozen eval manifest
  verifies without an override.

## Decision and next unblocker

The two approved prompt repairs are complete. They improve the sampled oracle
from 0/3 to 2/3 in separate comparisons; that is interface evidence only.
The sample is three tuning tasks, not an estimate over the complete catalog.

**R23.5 serialization remains blocked on teaching-context articulation.**
Teaching and no-context scores are still equal at 0/7, so the recovery-ratio
denominator is zero and the ratio is undefined. The ordinary context condition
has both response-format failures and incorrect mapping/state responses.
The next useful experiment is a separately frozen DEV comparison of teaching
transcript presentation using the same examples and exact scorer, with the
no-context and raw-transcript retrieval baselines retained. A proposed repair
must not expose the hidden oracle rule or held-out answer to that condition.
The v3/v4 approval does not cover another prompt version.

R20 retains its separate zero-effect/power no-go and unsigned meta-test gate.
Exact state reproduction remains an additional unresolved audit issue.
Do not spend a new full calibration fanout or claim serialization recovery
from these oracle improvements.
