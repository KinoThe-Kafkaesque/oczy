# Capability regression DEV v1

Completed September 14, 2026, Africa/Casablanca. Manifest: `6c4e04aa6f6aef11465d361dae9e1e95c18adb6243be2e597b67c586ac3e3414`.

**All historical comparisons reproduce exactly.** The 804 historical comparisons cover the six-capability battery,
serialized-state/text persistence, the direct-operation language improvement,
and the qualified scalar-compatible decoder. Reproducing an old failure is an
integrity pass, not a capability pass. Historical optimizers were not rerun.

| Replay | Rows | Exact reproduction |
|---|---|---|
| capability_actions | 20 | True |
| capability_reference | 240 | True |
| capability_restore | 336 | True |
| decoder_batch4 | 32 | True |
| decoder_single | 32 | True |
| direct_operation | 32 | True |
| pilot_restore_soft | 96 | True |
| pilot_restore_text | 16 | True |

## Candidate regressions on the unchanged earlier prompts

All three final diversity states receive the original v1 stage-2 probes, with
matched initial/control states and a zeroed state. These 160 probes include
individual rules, neutral copying and composition. The prompt remains the old
v1 prompt, so this is a robustness check beyond the new concise training interface.

| Seed | Context | Earlier control | Varied teaching | Lost | Gained |
|---|---|---|---|---|---|
| 0 | amber | 0/4 | 0/4 | 0 | 0 |
| 0 | cobalt | 0/4 | 3/4 | 0 | 3 |
| 0 | composition | 0/4 | 0/4 | 0 | 0 |
| 0 | silver | 0/4 | 2/4 | 0 | 2 |
| 1 | amber | 0/4 | 0/4 | 0 | 0 |
| 1 | cobalt | 0/4 | 1/4 | 0 | 1 |
| 1 | composition | 0/4 | 0/4 | 0 | 0 |
| 1 | silver | 0/4 | 0/4 | 0 | 0 |
| 2 | amber | 0/4 | 0/4 | 0 | 0 |
| 2 | cobalt | 1/4 | 0/4 | 1 | 0 |
| 2 | composition | 0/4 | 0/4 | 0 | 0 |
| 2 | silver | 0/4 | 3/4 | 0 | 3 |

Joint learning is not sequential accumulation or correction. Failed composition
does not isolate a composition mechanism when individual rules also fail.
Per-input changes and every raw response remain inspectable in the archived run.

## Actual tools and files

Forty-eight new action trials exercise the same four strict-JSON temporary-file
tasks. Initial/zeroed action controls fill the previous study's missing control
coverage. Tool core requires the exact call sequence, all actual files correct,
and no execution error. Strict success also requires the exact final answer.
The frozen action prompt's known final-answer placeholder is preserved.

| Condition | Seed | Tool core / 4 | Strict / 4 |
|---|---|---|---|
| initial | 0 | 2/4 | 0/4 |
| control | 0 | 0/4 | 0/4 |
| diversity | 0 | 0/4 | 0/4 |
| initial | 1 | 2/4 | 0/4 |
| control | 1 | 0/4 | 0/4 |
| diversity | 1 | 0/4 | 0/4 |
| initial | 2 | 1/4 | 0/4 |
| control | 2 | 0/4 | 0/4 |
| diversity | 2 | 0/4 | 0/4 |
| zeroed | 0 | 0/4 | 0/4 |
| text_control | None | 0/4 | 0/4 |
| text_diversity | None | 1/4 | 0/4 |

## Verification and packaged experience

All 1,012 rows have exact coverage and independently checked scores. Historical
comparisons include complete row objects, prompt hashes, raw strings and tool
outcomes. Candidate actions independently check expected calls, actual files,
final answers and execution errors. The model/scorer/runtime/source identities
remain pinned; zero optimizer steps and no meta-test access. Replay runs in one
fresh offline process, which is not the original phase-isolated training design.
Elapsed replay time: 2161.67 seconds.

The [local workbench](../workbench/README.md) combines this evidence with saved
trials and real exploratory inference. All seeds and errors are visible. Its
seven-condition live comparison does not alter frozen scores or learned state.
Model weights/runtime remain external to the source-and-evidence archive.
