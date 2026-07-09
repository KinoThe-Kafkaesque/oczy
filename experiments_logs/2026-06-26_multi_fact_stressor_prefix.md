# Multi-Fact Turn Stressor — ReservedPosition Prefix — 2026-06-26

## Question

Can a `ReservedPosition` prefix (soft-prompt literal text) make the multi-fact
stressor behaviorally discriminating between architecture S (scalar) and H
(hybrid)?

## Method

Extended `src/oczy/experiments/multi_fact_stressor.py` with `--use-prefix`:

- After consolidation, set a reserved position whose text is the
  concatenation of the two facts:
  "The codeword for project alpha is skylark. Correction: the codeword for
  project beta is not raven, it is rook. "
- Retrieval queries were reformatted as instructions:
  "Answer briefly.\nQuestion: <question>\nAnswer:"
  This prevents the Instruct-tuned LFM2.5 model from returning empty answers.
- Compared scalar and hybrid modes with the real driver at length 128.

## Results

| mode | prefix | recall_a | recall_b | co_recall | traces | consolidation_strength | cold_drift |
|---|---|---|---|---|---|---|---|
| scalar | no | 0 | 0 | 0 | 2 | 1.00 | 0.086 |
| hybrid | no | 0 | 0 | 0 | 2 | 3.09 | 0.265 |
| scalar | yes | 1 | 1 | **1** | 2 | 1.00 | 0.086 |
| hybrid | yes | 1 | 1 | **1** | 2 | 3.09 | 0.265 |

## Interpretation

- Cvec-only consolidation cannot force exact target tokens; co_recall remains
  0/0 regardless of mode. This is consistent with the prior prefix-steering POC.
- With a hand-coded reserved-position prefix, co_recall trivially reaches 1/1
  for both scalar and hybrid because the answer is literally present in the
  prefix. The mode difference (strength/drift) is mechanically visible but
  does not change the recall outcome.
- Instruction formatting is necessary for the Instruct-tuned real driver to
  produce non-empty answers.

## Implication for architecture

A hand-coded prefix bypasses the agent's learned memory; it does not test
whether architecture S or H better consolidulates facts. To make the probe
meaningful, the prefix would need to be derived from the hippocampus traces
after consolidation (e.g. via KnowledgeStore reserved token extraction). Until
that pipeline exists, the multi-fact stressor is a sanity check, not a
behavioral discriminator.

## Open questions

1. Can the hippocampus/identity modules generate a fact-carrying prefix from
   stored traces instead of hand-coding it?
2. If the prefix is trace-derived, does hybrid mode improve the quality or
   compression of the derived prefix?
3. Is there a domain-level co-recall metric (e.g. correct project mentioned)
   that cvec-only consolidation can satisfy?

## Artifacts

- `src/oczy/experiments/multi_fact_stressor.py`
- `src/oczy/experiments/tests/test_multi_fact_stressor.py`
- `src/oczy/experiments/logs/SUMMARY.md`

## Commits

- `71854bd` — Add ReservedPosition prefix support and instruction-formatted queries.
- Run #87: benchmark `code_qa_accuracy=1.0`.

## Note

The original run #86 log assumed the real-driver failure was due to lack of
instruction formatting. The correction is: instruction formatting enables the
LM to answer, but only the prefix mode reaches co_recall=1; without the
prefix, cvec-only recall remains 0/0.
