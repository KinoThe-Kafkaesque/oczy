# Language-interface DEV v3

Frozen September 13, 2026 before model calls, under the user's request to
continue language-interface work. Manifest:
`460e8eb5a53cf17b44a58011bd80efcb09ef3dce9ce6b25cb0f239a5a2327893`.
No earlier instrument or scorer is changed. This run has zero optimizer updates
and uses the same pinned INT8 Qwen, tokenizer and normalized exact scorer.

All seven conditions run on the same 32 cases: four clients crossed with four
calibration words (lime/melon/grape/peach, already used in DEV v2), and four new
confirmation words (apple/mango/lemon/guava). Amber appends vek, cobalt appends
mip, silver and never-taught quartz copy unchanged. Total: 224 scored responses.
All conditions, words and prompts were serialized and hashed before execution.
The data file contains scoring targets, but model calls receive only its
`messages` field. No transformed confirmation answer is supplied in a prompt.

| Condition | Meaning and permitted comparison |
|---|---|
| resolved_v2 | Exactly the previous oracle-selected operation, separate user turns |
| merged_resolved | Same system, operation and query text, merged into one user turn |
| direct_operation | Minimal direct append/copy command with explicit operand; composite primitive diagnostic |
| text_examples_v2 | Same nine crossed examples in the previous plain-text format |
| chat_examples | Same examples rendered as nine user/assistant demonstrations; same system and final query |
| table_v2 | The previous rule table in a user message |
| table_in_system | The identical rule table placed after the system instruction |

The resolved and direct-operation oracles select the rule externally and cannot
demonstrate learned context selection. Chat examples are an explicit retrieval
baseline, not persistent neural learning. Logit-bias/rerank are not run. There
is no best-prompt selection, no adaptive follow-on variant within this run, and
no promotion threshold or broad language-competence claim.

Decoder parity is checked on all 16 resolved_v2 calibration cases: the existing
scalar generator, Hugging Face greedy generation from input IDs, and the organ's
batch API with one item. Each sees the same rendered prompt and zero-width bank.
The audit compares decoded strings exactly; it does not claim token-level parity,
learned-bank parity or equivalence for multi-item padded batches.

Execution is one separate local CPU process with two threads, no network, a
read-only source tree and writable dedicated output directory. It runs alongside
the independent DEV-v2 training process. Model/runtime hashes are verified before
execution, and model identity is rechecked afterward. Filesystem isolation is an
execution boundary, not protection against arbitrary malicious local worker code;
this reference worker intentionally has access to all registered prompt cases.
Two threads differ from DEV v2's four: identical baseline strings are checked in
the final report, and this is not claimed as bitwise cross-thread reproduction.

Four new prompt/coverage checks pass before freezing; the eval guard protects the
new version and source. No remote jobs, meta-test, production or old loop restart.

```bash
/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/runtime/bin/python \
  scripts/probe_language_interface.py run \
  --root experiments/language-interface-dev-v3 \
  --output /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-13-language-interface-dev-v3 \
  --model /home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/model \
  --provenance experiments_logs/artifacts/2026-09-13_capability_validation_v1/selected_runtime_provenance.json
```

Use a new output directory for reproduction.
