# Language interface DEV v3 — September 13, 2026

**COMPLETE: minimal direct operations improve to 15/16 on both word sets;
reliable context selection remains unproved. A live decoder discrepancy is
identified and isolated for a separate versioned repair.**

[Frozen protocol](../experiments/language-interface-dev-v3/README.md),
[all prompts and outputs](2026-09-13_language_interface_dev_v3.json).
Manifest `460e8eb5a53cf17b44a58011bd80efcb09ef3dce9ce6b25cb0f239a5a2327893`.
Local source base `e71268bc9cb01839995e5b3db69d0ad43a9ef04d`, plus hash-frozen
uncommitted sources. No existing instrument, scorer or model weights changed.

## Fixed interface comparison

Each cell is exact responses out of 16: four words under amber, cobalt, silver
and quartz. Amber appends vek; cobalt appends mip; silver and quartz copy.
Calibration uses lime/melon/grape/peach, already examined in DEV v2. Confirmation
uses apple/mango/lemon/guava, frozen before this run. All seven conditions run
on every case; the optimizer in the concurrent learning experiment keeps its
previously fixed interface and receives no feedback from these results.

| Condition | Calibration | New confirmation |
|---|---:|---:|
| Previous oracle-selected operation | 9/16 | 8/16 |
| Same operation/query merged into one user turn | 8/16 | 8/16 |
| Minimal direct append/copy command | **15/16** | **15/16** |
| Plain-text crossed-example retrieval | 2/16 | 0/16 |
| Same examples as user/assistant turns | 10/16 | 6/16 |
| Complete rule table in user message | 3/16 | 3/16 |
| Same table moved to system message | 4/16 | 4/16 |
| Logit-bias / rerank | Not run | Not run |

Merging user turns alone does not improve the operation prompt. Moving the rule
table to the system message gives one additional correct response in each set,
with substantial remaining errors. Conversation-form example retrieval improves
on the plain-text baseline in both sets, but generalizes poorly: confirmation
amber is 0/4, cobalt 3/4, silver 1/4 and quartz 2/4. Its success is retrieval
behavior, not persistent neural learning or a storage-efficiency result.

The direct-operation condition uses a minimal system instruction and a query
such as `Append the suffix "vek" to the string "apple".` It supplies the operand
and rule, but not the final concatenated target. Confirmation amber is 4/4,
cobalt 3/4, and both identity scopes 4/4. The remaining error is
`mango.mip` instead of `mangomip`; calibration similarly gives `grape.mip`.
No punctuation stripping or other scoring repair is applied.

This is a composite prompt change and an oracle-selected primitive operation:
it removes the client-resolution problem. It establishes that this pinned organ
can often execute the operation when expressed directly. It does not establish
learned context selection, complete-rule mastery, preservation, or a universally
reliable language interface. The new-word check repeats only four words and
two affixes; trial counts are not independent tasks or a powered acceptance test.

## Decoder finding

The old scalar API, native Hugging Face input-ID API and existing batch API
produce the same decoded string on only **8/16** of the registered parity cases.
All three use deterministic greedy choices, but their effective settings differ:
the scalar API uses raw argmax, whereas native/batch APIs inherit the model's
repetition penalty of 1.1. Input-ID and embedding-based generation also expose
different token histories to that penalty. The runtime manifest declares 1.0;
merely recording that declaration does not enforce it inside the batch call.

Source inspection additionally identifies supplied position IDs that remain
fixed during continuation under the pinned Transformers 5.0.0 implementation.
A separate decoder-parity DEV v1 tests the two factors independently and a new
candidate with explicit neutral generation settings and advancing positions.
This report does not preempt that diagnostic's result.

The language table above uses the scalar path throughout. Its improvements and
failures are not caused by switching decoders between prompt conditions. All 48
paired scalar baseline strings reproduce DEV v2 exactly, despite this run using
two CPU threads versus four there. The live parity check compares decoded text;
it does not establish token or logit equality.

## Verification and next gate

All 224 scores are independently recomputed using the unchanged normalized exact
scorer. Trial coverage, complete prompts, registered source/input hashes, runtime
identity and unchanged model hash pass audit. Zero optimizer updates, no network,
no meta-test. Execution takes 595.98 seconds alongside independent training.
Four prompt/coverage tests passed before the manifest was frozen.

The next language task is to preserve the simple operation interface while
learning which operation belongs to the current context. Direct-operation
oracle results must remain a separate upper control, and example retrieval
must remain in the comparison. The decoder candidate must qualify before it
can be adopted by a newly versioned batched experiment; old scores remain intact.
