# Six-capability DEV validation v1 — September 12–13, 2026

**COMPLETE: valid bounded DEV evidence of context leakage and loss of earlier
knowledge; none of the six full learned-state capability requirements passes.**
Ordinary lossless text compression passes byte/behavior preservation, with a
weak retrieval baseline. The uncoupled model executes three of four tool tasks
correctly; reliable action with the learned bank is not established.

[All scores, outputs, tool transcripts and trajectories](2026-09-13_capability_validation_v1.json)
and [reproducibility archive](artifacts/2026-09-13_capability_validation_v1/README.md).

| Requested capability | Verdict in this battery |
|---|---|
| Selective application | **Fails.** Unassigned silver is correct on only 1/12, 2/12 and 1/12 trial responses after the three lessons; every initial-state seed is 4/4 per stage. |
| Accumulating knowledge | **Fails.** Adding cobalt loses all previously correct amber responses: retention 0/1, 0/1, 0/2 across seeds. |
| Correcting knowledge | **Fails the preservation requirement.** New amber accuracy is 4/4, 1/4, 0/4; unrelated cobalt retention is 0/3, 0/2, 0/2. |
| Composition | **Not established.** Two-rule composition is 0/24. None of these trials has both corresponding single-rule prerequisites correct, so a distinct composition deficit is not isolated. |
| Useful compression | **Fails for learned neural state.** The 28,800-byte file exceeds 143–295 bytes of active examples. Zlib uses 91–129 bytes and preserves all raw retrieval outputs, whose accuracy remains weak. |
| Reliable action | **Partial uncoupled execution, unreliable coupled execution.** Correct full tool sequences/arguments plus verified outcomes: no bank 3/4; learned banks 0/4, 0/4, 1/4. Strict end-to-end success including final response is 0 in every condition. |

The user explicitly requested validation of selective application, accumulation,
correction, composition, useful compression, and reliable action. A new local DEV
instrument was frozen before execution. Historical eval-v2, R20 and R23.5 scores,
targets and thresholds were not changed, and the pending legacy-eval repair was
not implicitly approved.

Instrument: [protocol](../experiments/capability-validation-v1/README.md),
[manifest](../experiments/capability-validation-v1/MANIFEST.json).
Manifest SHA-256: `74d6614fc332c832761c208e63e753dd85f437b850ebfaeeac196faeccdf53ce`.
Source base: main at `e71268bc9cb01839995e5b3db69d0ad43a9ef04d`, with individually
hash-frozen local uncommitted execution sources. This is not a remote campaign.

## Scope and method

One shared 8 x 896 FP32 input bank receives three lessons: amber appends `vek`,
cobalt appends `mip`, then amber changes to `zul`. Unassigned silver copies the
input. Each lesson supplies three corrected examples; there is no rehearsal of
older examples. Three seeds each receive 24 fixed Adam updates per lesson,
learning rate .03, gradient clip 1, mean example CE including EOS. The bank
persists while the Adam state resets between lessons. No result-driven tuning,
early stopping, or checkpoint selection occurs.

Each stage tests four unseen words per client and two words under both client
orders: 16 probes per stage. Composition is defined on every intermediate
string and its order changes the answer. Training identifies each rule within
the declared 11-member candidate family. This does not establish arbitrary
function identifiability or broad natural-language preference competence.

Fresh-process learned banks are compared with their initial states and a zeroed
bank. References include chronological example history, latest-example prefix
retrieval, complete oracle rules and losslessly compressed/reloaded examples.
No-context controls are also retained. Prefix retrieval is implemented here;
logit-bias and rerank have no result in this bounded suite.

Retention counts pair the same probes across stages and count only previously
correct answers as eligible for loss. Acquisition failures are reported
separately. Composition also reports whether both component rules succeed on
the same input. The four base words repeat across stages and seeds; pooled rows
are not independent tasks or a powered research acceptance test.

## Reference results

Each cell is correct responses out of 16, including three scopes and composition.
The JSON result retains separate category counts and every generated answer.

| Condition | First rule | Add second rule | Correct first rule |
|---|---:|---:|---:|
| No context | 8/16 | 4/16 | 4/16 |
| Chronological example history | 0/16 | 0/16 | 1/16 |
| Latest-example prefix retrieval | 0/16 | 0/16 | 3/16 |
| Complete oracle rules | 6/16 | 2/16 | 3/16 |
| Zlib example retrieval | 0/16 | 0/16 | 3/16 |
| Logit-bias / rerank | Not run | Not run | Not run |

The full-rule reference already has low exact accuracy. For example, after
the second lesson, oracle cobalt/pear returns `mippear.` rather than `pearmip`,
and oracle silver/pear returns `pearmip` rather than copying `pear`. This is an
interface/articulation limitation under this frozen prompt, not evidence of
unsupported gold targets and not a general impossibility result about the model.

## Training result

All 216 fixed updates completed. Every seed reproduced all three current-lesson
teaching examples after each lesson: 27/27 exact teaching checks across nine
seed/lesson combinations. All state gradients were finite and nonzero, and no
organ parameter received gradients. The organ hash after training matches its
pre-training hash. Teacher-forced first losses ranged from 6.60 to 10.44 and
final losses from .000924 to .004921. These are current-lesson fit checks, not
evidence of retaining old rules or succeeding on held-out inputs.

## Fresh-process learned-state results

Every cell is correct responses out of four. Stage 1's two-client probes involve
one known rule and an identity operation; their occasional successes do not count
as evidence of combining two learned rules.

| Seed | Stage | Amber | Cobalt | Unassigned silver | Composition |
|---|---|---:|---:|---:|---:|
| 0 | Teach amber | 1/4 | 1/4 | 1/4 | 0/4 |
| 0 | Add cobalt | 0/4 | 3/4 | 0/4 | 0/4 |
| 0 | Correct amber | 4/4 | 0/4 | 0/4 | 0/4 |
| 1 | Teach amber | 1/4 | 0/4 | 0/4 | 3/4 |
| 1 | Add cobalt | 0/4 | 2/4 | 0/4 | 0/4 |
| 1 | Correct amber | 1/4 | 0/4 | 0/4 | 0/4 |
| 2 | Teach amber | 2/4 | 0/4 | 0/4 | 2/4 |
| 2 | Add cobalt | 0/4 | 2/4 | 2/4 | 0/4 |
| 2 | Correct amber | 0/4 | 0/4 | 1/4 | 0/4 |

All initial and zeroed states score zero on taught-rule and composition targets.
Initial states preserve unassigned silver on 4/4 inputs per seed at every stage;
the zeroed control does too. At stage 1, initial cobalt copies 3/4, 4/4 and 4/4;
zeroed cobalt copies 4/4. These controls are retained in the full JSON tables.

Scope witness: after lesson 1, restored seed 0 returns ` pearvek` for unassigned
silver/pear, whose target is `pear`. After adding cobalt, amber loses its
previously correct `fig` response in seed 0, `pear` in seed 1, and `plum`/`fig`
in seed 2. After correcting amber, cobalt loses previously correct pear/plum/fig
in seed 0, pear/plum in seed 1, and pear/fig in seed 2. These are losses of
previously correct responses, not a claim that either complete rule had first
been mastered by every seed.

The same bank updates can fit new examples and sometimes generalize to unseen
words, but do not maintain a scoped collection of preferences. These results
do not distinguish inadequate training constraints from inadequate memory
architecture. There is no preservation objective or replay in this updater.

## Storage accounting

| Stage | Active example UTF-8 bytes | Chronological bytes | Neural payload | Neural file | Zlib file |
|---|---:|---:|---:|---:|---:|
| First rule | 143 | 143 | 28,672 | 28,800 | 91 |
| Add second | 295 | 295 | 28,672 | 28,800 | 128 |
| Correct first | 295 | 440 | 28,672 | 28,800 | 129 |

Per-memory file counts include their format headers; shared model/runtime and
audit/provenance files are excluded equally. Codec and UTF-8 decoding conventions
are fixed in the shared contract. Lossless text round trips are byte-exact and
their replay generations are independently checked. Text compression remains
retrieval, and its accuracy is limited by the weak raw-text baseline. The neural
state does not meet the user's less-storage-than-examples requirement.

## Reliable-action contract

The model receives strict JSON tool schemas, task instructions and actual tool
responses. Gold plans and initial file contents do not enter its initial prompt.
Four actual temporary-filesystem tasks test reading, case-sensitive writing,
reading then replacing one substring, and copying unknown contents obtained
through a read. The no-bank actor, all three final learned banks and the
latest-example prefix baseline run the same tasks.

Success requires the exact entire tool sequence and arguments, exact final
answer, exact final filesystem snapshot and no execution error. A correct final
answer alone cannot pass. Extra calls, wrong path case, extra arguments and
false completion are rejected by the evaluator. The new bounded executor does
not imply that the old tool curriculum or a production agent has been repaired.

| Condition | Exact execution and verified outcomes | Full contract including final response |
|---|---:|---:|
| No bank | 3/4 | 0/4 |
| Learned bank, seed 0 | 0/4 | 0/4 |
| Learned bank, seed 1 | 0/4 | 0/4 |
| Learned bank, seed 2 | 1/4 | 0/4 |
| Latest-example prefix retrieval | 1/4 | 0/4 |
| Logit-bias / rerank | Not run | Not run |

The no-bank model correctly reads, writes and edits real files, then echoes
the system prompt's `REQUESTED FINAL STRING` placeholder instead of the required
final answer. Grounded copying emits duplicate JSON keys and performs no action.
These are separate failure modes; the strict zero score must not conceal three
correctly executed tasks. A future tool-prompt repair should address placeholder
echo without relaxing execution, argument or filesystem checks.

Learned seed 0 claims `Report.txt created and saved as done.` without calling
any tool; the requested file does not exist in its final snapshot. Learned seed
2 successfully reads and copies the unknown `CODE-731` contents, then gives the
wrong final response. On the read-only task, that same seed makes unrequested
edits to the source file and exhausts the tool budget. All actions occur only
inside disposable test directories; the complete actual snapshots and tool
responses are retained in the JSON.

The coupled learned-bank condition is unreliable in this battery. Action tests
do not include initial or zeroed banks, so they do not separately identify the
effects of inserting a bank and changing its learned values. This limitation
does not affect the direct context/retention comparisons, which do include both
controls. The action instrument measures UTF-8 text fixtures, not arbitrary
binary files, networks, or production tool integration.

## Verification and limits

Before execution, 22 adversarial/positive evaluator checks passed. Subsequent
checks also verify that gold targets do not enter model prompts and that failure
to acquire a rule cannot be counted as forgetting; the focused suite totals
47 passing checks including the eval guard; the three retention-accounting checks
were rerun after adding final coverage audits and pass. The new instrument paths are guarded
against future autonomous changes. The previous serialization pilot's manifest
and bound execution sources still verify unchanged.

The pinned Qwen 0.5B organ, INT8 weight-only recipe, four-thread CPU runtime and
normalized-exact scorer match the prior persistence pilot. Four separate
processes use read-only source and disabled networking. Registered training,
probe, oracle and output directories are masked according to phase, with actual
file-visibility checks. These masks supplement audited model inputs; they are
not a general adversarial isolation guarantee over all documentation on the host.

All four phases exited successfully, with distinct process IDs and unchanged
organ/scorer hashes. Their total execution time was 1,875.14 seconds (31.25
minutes). The final audit independently recomputed 576 probe scores and 20
strict action verdicts, verified exact case/update coverage without omissions
or duplicates, verified all 216 finite nonzero gradients and every released
numeric-state hash, and confirmed exact zlib/raw replay output equality.
The authorized guard invocation `EVAL_CHANGE_APPROVED=1 python
scripts/eval_guard.py HEAD...HEAD --allow` passed. Focused Ruff checks passed.

No meta-test, remote job, production action, background-loop restart, or
meta-trained cortex-writer validation is part of this run. This tests the current
soft-bank learner and frozen organ on one narrow rule family and four tool tasks.

## Next blocker

Prioritize scoped acquisition and preservation of unrelated behavior. Full
teaching fit already succeeds, while most unseen first-rule inputs fail and
subsequent updates erase every previously correct old-rule response. A follow-up
should compare one preservation mechanism at a time against these fixed controls;
the present run does not identify whether replay, training contrasts or a
different memory architecture is the best repair. Any added teaching examples
need their own versioned curriculum.

The language interface also needs an interpretable complete-rule reference, and
the tool prompt needs a separate placeholder repair plus initial/zeroed action
controls. Establish two retained component rules before claiming a distinct
composition capability. Storage reduction remains a separate requirement.
