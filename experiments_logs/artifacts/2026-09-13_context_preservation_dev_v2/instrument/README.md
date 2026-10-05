# Context and preservation DEV v2

Authorized September 13, 2026 by the user's request to proceed on learning in
the correct context, preserving knowledge, and repairing the language interface.
Manifest: `c68b31b35fb9c5f12dd9e68174ee26313145a4c2a3e262255bc2e23bb2454390`.
All data, execution sources, model/runtime/scorer hashes and choices were frozen
before the first model call. No old eval, score or instrument source is changed.

## What this can establish

This is a bounded DEV sequence: language-interface comparisons, joint scoped
representation, then a conditional preservation comparison. Joint training is
a representation diagnostic and does not demonstrate continual learning. The
gradient-based updater remains a DEV serializer; it does not satisfy R20's
primary evaluation requirement of learned state updates without backpropagation.

## Language interface: five registered conditions

| Condition | System | Query | Complete-rule presentation |
|---|---|---|---|
| v1_full | Original | Original sentence | Original prose |
| query_full | Original | Separate Client/Input/Output lines | Original prose |
| concise_full | Short instruction | Separate lines | Original prose |
| concise_table | Short instruction | Separate lines | Explicit input-plus-suffix table |
| resolved_oracle | Short instruction | Separate lines | Oracle selects just the applicable instruction |

Only adjacent comparisons through `concise_table` isolate one interface change.
The resolved oracle additionally removes context selection: it diagnoses the
ability to perform a specified string operation and cannot validate learned
selection. Neither oracle supplies the final concatenated answer.

No-context and crossed-example prefix retrieval remain in the reference table;
logit-bias and rerank are not implemented and have no result claim here.

The old words pear/plum/kiwi/fig are DEV calibration only. New confirmation words
are lime/melon/grape/peach. All seven conditions run on calibration and stage-2
confirmation, with the table and resolved oracles also run on stage-3 confirmation.
The learner uses the concise system/query chosen before any results; it does
not adopt whichever prompt happens to score best.

## Crossed curriculum and joint representation

In v1, amber and cobalt initially had different teaching inputs. A learner could
fit those examples using word identity without learning the context distinction.
The new nine-example curriculum crosses the same three words (oak/pine/elm) with
all three teaching contexts: amber appends vek, cobalt appends mip, silver copies
unchanged. Each word has three different required outputs, forcing context to
matter for a perfect training fit. Quartz is never taught and tests the default
identity behavior. Silver measures a taught neutral context, not an untaught one.

The same frozen Qwen 0.5B organ receives one shared 8 x 896 FP32 bank, without
client-dependent routing. Seeds 0/1/2, 48 full-batch Adam updates, learning rate
.03, initial standard deviation .02, gradient clip 1, mean per-example CE
including EOS. No held-out optimization, early stopping or checkpoint selection.
The final bank is the only learned state released to a fresh process.

Capacity validation has four fresh words under each of four contexts: 16 probes
per seed. Initial states and a zeroed bank are controls. Required admission to
the preservation comparison is all 9 teaching fits and all 16 fresh responses
correct for every seed. This is the predeclared prerequisite of already acquired
knowledge, not a new statistical promotion threshold. Failure blocks correction
training and is not evidence of mathematical impossibility or a global optimum.

V2 changes both curriculum and interface relative to v1. Its joint-memory result
cannot attribute improvement to either factor alone. The paired reference ladder
is the isolated interface comparison.

## Conditional preservation comparison

If capacity passes, clone each joint bank into two arms. Teach only the three
new amber corrections (append zul) for 24 Adam updates, using identical settings:

- Unprotected: mean new-example CE including EOS.
- Proximal: the same CE plus `10 * mean((bank - pre_update_bank)^2)`.

The coefficient is fixed before execution, with no search or adaptive tuning.
The arms differ only by this penalty. It discourages changing the existing bank
but does not guarantee preservation of unrelated functions. No old examples or
oracle/probe outputs are available to the correction worker. Its anchor is a
transient copy of the incoming numeric bank and is discarded; the serialized
state size remains 28,672 payload bytes / 28,800 .npy file bytes per arm.

Fresh-process scoring measures new amber behavior, paired retention of cobalt,
neutral silver, and untaught quartz. No action or composition claim is tested
in this version. Successful joint initialization would still limit any later
preservation result to correcting an already established collection of rules.

## Isolation, validation and run

Each phase runs in a separate four-thread CPU process with networking disabled
and source read-only. Registered forbidden input/output directories, test files,
data-builder source, bytecode and this README are masked as applicable. Model
inputs contain only the registered prompt and bank, never probe targets. These
are audited phase boundaries, not a guarantee against arbitrary malicious host
code. The runner checks the exact model/runtime manifest and organ/scorer hashes.

Before execution: 10 new checks plus 21 eval-guard checks pass. They cover crossed
teaching, new confirmation inputs, prompt equivalence, one-field prompt changes,
target exclusion, forbidden inputs and fail-closed capacity admission. The guard
accepts the human-authorized new instrument and protects it against future edits.

```bash
/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/runtime/bin/python \
  scripts/run_context_preservation.py \
  --root experiments/context-preservation-dev-v2 \
  --output /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-13-context-preservation-dev-v2 \
  --model /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/model \
  --provenance experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json
```

Use a new output directory for reproduction. No remote jobs, production actions,
old autonomous loop restarts or meta-test access are part of this authorization.
