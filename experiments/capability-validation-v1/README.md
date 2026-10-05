# Six-capability local DEV validation v1

Authorized by the user's September 12 request to validate selective application,
accumulation, correction, composition, useful compression and reliable action.
This is a new, separately versioned DEV instrument. It does not modify or approve
the pending legacy-eval/R20 repair proposal and does not access meta-test.

Frozen manifest: `74d6614fc332c832761c208e63e753dd85f437b850ebfaeeac196faeccdf53ce`.
All inputs, execution sources, model, runtime, scorer and training settings are
bound before the first model run. `MANIFEST.json` is authoritative.

## Teaching and probes

One shared numeric bank receives three lessons in sequence:

1. Client amber appends `vek`.
2. Client cobalt appends `mip`; amber should retain `vek`.
3. Client amber changes to `zul`; cobalt should retain `mip`.

Each lesson has three corrected examples. Unassigned clients explicitly copy
the input. Four disjoint words test each of amber, cobalt and unassigned silver
after each lesson. Two of those words also test both client orders. Thus there
are 16 probes per stage; stage 2/3 compositions require two learned rules.
`pearvekmip` and `pearmipvek` differ, so reversing composition order cannot pass.

The rules are identifiable within an enumerated 11-member candidate family;
append operations are defined on every string, including intermediate outputs.
This is evidence about one explicit-context formatting family, not arbitrary
human preferences, implicit context, or broad generalization. The four base
words repeat across stages and seeds and are not independent new tasks.

## Fixed learner and comparisons

The frozen Qwen organ uses one 8 x 896 FP32 input bank, with no client-dependent
routing. Seeds 0/1/2, 24 Adam steps per lesson, learning rate .03, initial standard
deviation .02, gradient clipping 1, mean example CE including EOS. Optimizer
state resets between lessons; the learned bank persists. No old-example replay,
held-out optimization, early stopping, checkpoint selection, or result-driven
retry is allowed. This extends the prior pilot interface and cannot establish
the independent causal effects of its richer prompts and sequential training.

Reference comparisons: no context, chronological example history, latest-example
retrieval, complete rule oracle, and losslessly compressed/reloaded latest
examples. Numeric controls: initial, zeroed and fresh-process learned banks.
Identical deterministic control prompts may reuse a generated output; every
such reuse is recorded. Zlib replay is independently generated after decoding.
Prefix retrieval is the implemented retrieval comparator; logit-bias and rerank
are not implemented in this bounded battery and make no result claim.

Acquisition prerequisites must be reported: failure to acquire a rule is not
evidence of forgetting it. Accumulation measures retained previously correct
amber responses after learning cobalt. Correction measures new amber responses
and retained previously correct cobalt responses. Composition reports component
accuracy alongside combined-rule accuracy. Full battery success requires all
required cases on all three seeds; partial counts and counterexamples remain
results. No research promotion threshold or power claim is introduced.

## Storage

Charge the actual standalone `.npy` or `.zlib` file, including format headers,
against the exact UTF-8 active example text; also disclose chronological text
size and neural payload size. Shared model, decoder and runtime are excluded
equally. Audit/provenance files are reported separately rather than charged to
one method alone. Useful storage reduction also requires preserved behavior.
Lossless text compression remains retrieval, not learned neural compression.

## Actions

Four real temporary-filesystem tasks: read, case-sensitive write, read then edit
one substring, and copy unknown contents obtained from a read. The model sees
tool schemas, task instructions and actual tool responses; gold tool plans and
initial file contents are never inserted into its prompt.

Require a strict JSON object, exact tool sequence/arguments, exact final answer,
exact complete final filesystem snapshot and no execution error. Limit to four
tool calls, 128 generated tokens per turn and 1,024 bytes per string argument.
No arbitrary shell, permissive JSON extraction, inferred execution, or substring
matching. All work happens in a newly created temporary directory per case.
The no-bank actor, each final learned bank, and latest-example prefix retrieval
run the same tasks. This is a new bounded executor, not a claim that the legacy
tool curriculum or production agent integration is repaired.

## Isolation and execution

Each phase runs in a separate CPU process with network disabled and source
read-only. Training cannot open probes, oracle, action fixtures or reference
outputs; restored numeric evaluation cannot open teaching, optimizer histories
or retrieval text. The data builder and cached Python bytecode are masked.
Actual forbidden-file visibility checks and model hashes are preserved.

```bash
/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/runtime/bin/python \
  scripts/run_capability_validation.py \
  --root experiments/capability-validation-v1 \
  --output /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-12-capability-validation-v1 \
  --model /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/model \
  --provenance /home/nyanpasu/.local/state/oczy/remote-queue/campaigns/r20-int8-dev-calibration-v6/calibration-a8c98d6/results/cal3-d0-t00-03/remote_run_provenance.json
```

The output path must not already exist. Reproduction uses the exact historical
model path inside an offline namespace to preserve model identity. No remote
runner, background optimizing loop or sealed-data job is started.
