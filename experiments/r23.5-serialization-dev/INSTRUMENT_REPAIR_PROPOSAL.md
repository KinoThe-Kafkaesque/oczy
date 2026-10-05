# DEV instrument repair proposal — approved September 11, 2026

Prepared September 11, 2026, after the R20 identity and output-path diagnostics.
The user approved both amendments, in order; see [APPROVAL.md](APPROVAL.md).
The proposal records their frozen scope and is not a meta-test authorization.

## Evidence requiring a decision

The historical model identity is reproducible when loaded at the recorded
Kaggle path. The decoder's EOS spelling leak has been fixed independently.
On the same three public tuning tasks, both old and corrected decoders score
0/7 without context, 0/7 with teaching context, and 0/3 with oracle context.
All 17 prompt hashes agree across the paired runs, and every output differs
only by special-token removal. The model parameters remain unchanged.

Two separate prompt-contract problems remain:

1. The exact scorer expects a bare token/string, while prompts ask ordinary
   questions without consistently requiring that output format. For example,
   the teaching-context output `The transformed output for input nu is nurpnu.`
   is wrong under the existing scorer's contract, whose expected text is
   `nurpnu`. Substring matching would conceal this mismatch and is not proposed.
2. The transformation oracle does not state its complete algorithm. Its
   conditional prompt says `Rule: conditional with parameters 'pex' and 'nurp'`
   plus three consonant-start examples. The generator actually branches on
   whether the first character is a lowercase vowel. That predicate is absent
   from the oracle text. An oracle failure here cannot establish a clean
   language-model expressivity ceiling.

## Amendment A: response format only — proposed meta_cortex/v3

Prepend this exact system message to every probe, including vanilla,
retrieval, learned-state and oracle conditions:

> Return only the requested answer token or string. Do not add an explanation,
> label, or surrounding quotation marks.

Keep the original user-message sequence, examples, rule semantics, expected
answers, task assignments, splits, seeds, scoring code, metrics, thresholds,
and model/quantization artifacts unchanged. Oracle rule wording is unchanged
in this amendment. The single changed variable is the response-format request.

Create a new versioned instrument and recompute its content hashes; never
edit an existing materialized v2 view or checkpoint. Repeat the identical
three-task DEV diagnostic and record all per-family outputs. This is a prompt
contract check, not a cortex-learning claim or authority to promote R20.

## Amendment B: complete oracle semantics only — proposed meta_cortex/v4

Starting from the fixed response-format version, replace the oracle's
template-name/parameter shorthand with these exact algorithm descriptions:

| Template | Oracle description |
|---|---|
| permutation | Reverse the input string character by character. |
| substitution | Replace every lowercase vowel (a, e, i, o, u) with `{param1}`. Leave all other characters unchanged. |
| conditional | If the input begins with a lowercase vowel (a, e, i, o, u), append `{param1}` to the input. Otherwise, prepend `{param2}` to the input. |
| composition | Reverse the input string character by character. Then replace every lowercase vowel (a, e, i, o, u) with `{param2}`. Leave all other characters unchanged. |

Retain the existing worked examples and probe operands. Apply these
descriptions only to the oracle condition. Ordinary teaching and learned-state
conditions must not receive a hidden rule or held-out expected answer.

Keep every other field from Amendment A unchanged. Generate a separately
versioned, hash-checked instrument and repeat the same DEV check. Report the
oracle change separately from the response-format change; do not combine them
into a claim of one causal improvement.

## Frozen boundaries and subsequent work

- Leave `eval/v2` and the original `meta_cortex/v2` evidence untouched.
- Keep normalized exact scoring; no substring matching, answer extraction,
  threshold lowering, or removal of failed conditions.
- Test each oracle description against the existing rule function over the
  public DEV operands and both conditional branches before it is used.
- Use a clean commit-addressed source and the verified CPU/runtime/model path
  for any eventual remote work. This approval would not itself launch a batch.
- Do not access meta-test, sign a candidate, or claim an R20/R23.5 verdict.
- Only after a positive, interpretable contextual effect exists should the
  separately frozen R23.5 serialization pilot proceed. A zero recovery
  denominator remains undefined. Recalibration, power feasibility and any
  later meta-test sign-off remain separate gates.

Requested human decision: authorize these two DEV-only instrument amendments
in the stated order, with separate version bumps and separate result tables.
