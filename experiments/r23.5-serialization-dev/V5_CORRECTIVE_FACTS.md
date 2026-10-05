# V5 DEV teaching-format comparison — September 12, 2026

## Authorization and frozen scope

After being told that the next step was a DEV teaching-format comparison, the
user instructed: **"get on the next blocker"**. This authorizes the proposed
next comparison. It does not authorize new task answers, threshold changes,
meta-test access, a full serialization study, or a remote research campaign.

The precise implementation is frozen in
[V5_CORRECTIVE_FACTS_PROPOSAL.json](V5_CORRECTIVE_FACTS_PROPOSAL.json), SHA-256
`f87fee485939371596921498b9f6d0c227280c3a3d660d4092a5df67ea3b9cc6`.
That file preserves its original proposal status; the separate
[V5_APPROVAL.json](V5_APPROVAL.json) records authorization for those exact bytes.

The parent is the unchanged v4 manifest
`d58adb749f4112f2920f08e0dbe0ce5c6a1ee7711f562faee4e41072eccc8764`.

## One changed variable: teaching presentation

Retain the original no-context, full teaching-transcript and oracle conditions.
Add `corrective_facts_context`, constructed as:

```text
[unchanged v4 system instruction]

Corrected examples:
- {first event.correction, verbatim}
- {second event.correction, verbatim}
...

[unchanged original query]
```

The renderer takes only public events and query messages. It copies every
correction in original order, omits the failed assistant attempts and their
observation turns, and never reads an oracle rule, task identifier or expected
probe answer. The omitted observations do not contain positive facts absent
from the corresponding corrections in these generated tasks.

For example, the finite-state condition retains these exact facts:

```text
Corrected examples:
- From q0, input a transitions to q2.
- From q1, input a transitions to q2.
- From q1, input b transitions to q2.
- From q2, input b transitions to q1.
- From q2, input a transitions to q2.
```

The query remains `Given current state q0 and signal a, which state follows?`.
Its answer is already a public teaching fact. It is not extracted from the
held-out probe target to construct the prompt.

## Fixed execution and interpretation

- Same first tuning task in each family; all original probes and targets.
- Seven no-context, seven original-teaching, seven corrective-facts and three
  oracle generations: 24 total. The old 17 prompt hashes must match v4 exactly.
- Same historical Qwen INT8 model/runtime, empty soft bank, greedy decoding,
  maximum 32 new tokens, exact scorer and zero optimizer steps.
- No task selection from observed outputs, filtering of failed probes,
  substring credit, threshold selection or answer postprocessing.
- Report raw outputs and full/per-family counts. A gain is a teaching-format
  diagnostic, not a learned-cortex or serialization result.
- Three contextual questions lack taught mappings; retain them in the table
  and explicitly label that limitation. Four questions are directly taught.
- Any repair of probe coverage, finite-state action teaching, or contextual
  composition semantics is a separate instrument change and is not applied.

The runner is `scripts/probe_dev_corrective_facts.py`. It rejects missing or
mismatched approval, altered plans/source/parent artifacts, and reused output
directories. It records every output before proceeding to the next generation.
