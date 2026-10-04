# Curriculum and evaluation repair plan — September 12, 2026

**Status: REVIEW CANDIDATE. Instrument changes are not applied.**

The user authorized an audit and improvements before fresh-pattern confirmation.
The audit and the eval-guard engineering repair are complete. This document
defines the specific scoring/task changes for human sign-off under
[AGENTS.md](../../AGENTS.md). No change below authorizes sealed/meta-test access,
remote submission, threshold selection from test results or automatic promotion.

## A. Repair the legacy scorer in separate versions

### A1 — eval-v2.3, concrete patch ready

Apply [eval_v2_3.patch](eval_v2_3.patch), then regenerate/verify the manifest
using the approved eval-change workflow. The patch:

1. Rejects an empty or whitespace-only answer/expected target before any
   exact, substring or semantic matching.
2. Advances the declared version from v2.2 to v2.3.
3. Adds runtime scorer, validator, dataset/split loader and curriculum driver
   source files to the manifest, so those source bytes are checked on load.

No episode, probe, target, match mode, split allocation, threshold or nonempty
matching behavior changes in A1. The patch applies cleanly but has not been
applied. [Shadow validation](SHADOW_VALIDATION.json) against all 120 shipped
probes changes empty-answer acceptance from 120/120 to 0/120, rejects
whitespace-only answers on all 120, retains all 120 expected-answer passes,
and changes none of 960 nonempty comparisons. An isolated source-tamper check
confirms that changing the candidate scorer invalidates its manifest.

This is one scoring-variable repair plus provenance enforcement. It is not
evidence of model improvement and does not repair nonempty wrong-sense matches.

### A2 — separately frozen sense contract

Replace union-of-senses matching with explicit per-sense accepted aliases and
negative foils. Each probe's domain/sense must be unambiguous from the actual
agent-visible request and preceding teaching. A fashion answer must not pass
an ML-model probe; a Git answer must not pass a tree-branch probe. Empty,
negated-only, wrong-sense and all-labels-at-once responses must fail. Correct
paraphrases need an explicit reviewed alias, rather than shared incidental
tokens or a newly introduced model judge.

Prepare and review the accepted/negative response corpus before activating
this second version. Re-score available raw outputs in a separate comparison
table, preserving original scores and versions. Do not reinterpret logs that
lack raw probe outputs. Rebaseline only after the scorer contract passes its
positive and adversarial response distribution checks.

## B. Rebuild R20's public DEV task support

Create a separate `oczy/meta-cortex/taskgen/v2-dev` instrument lineage. Preserve
the historical v1 generator, materialized v2 instrument and v3/v4/v5 prompt
diagnostics. No sealed generator or catalog is opened or regenerated here.

The measurement scorer remains `normalized-exact/v1`. Retain no-update,
retrieval/context, oracle, random/untrained, feedback-shuffled, zeroed and
swapped controls required by the corresponding experiment. Keep task families,
event budgets and endpoint definitions explicit in the new manifest; do not
compare absolute v1/v2 task scores as a causal learner improvement.

Materialize support repairs and specificity repairs as separately labeled
versions/steps so the source of changed measurements is inspectable.

| Defect | Required constructor behavior | Pre-model admission check |
|---|---|---|
| Untaught random contextual mappings | Same-rule and paraphrase probes select from actually taught pairs. | Every required lookup appears in teaching; no answer inferred from an independent unseen random assignment. |
| Undefined contextual composition | Construct typed chains: first taught mapping outputs a symbol accepted by the second taught mapping. | Both transitions are defined and taught; an independent evaluator follows the chain without a fallback. |
| Duplicate events after exhausting pairs | Sample distinct teaching facts without replacement; never silently duplicate after a retry limit. | Declared event count equals distinct teaching records; insufficient support is a construction error. |
| Unseen random FSM edges | Transfer means a new path over taught transitions, rather than guessing independently sampled edges. | Every traversed edge is taught; every intermediate state is defined. |
| Untaught actions and absent goals | Include the action mapping and initial goal required by the probe in the allowed teaching transcript. Ask a goal-dependent decision if claiming goal retention. | Removing/changing the goal changes the correct decision in registered paired examples; no hidden action lookup. |
| Conditional rule branch absent | Select teaching examples that cover every branch required by transfer/composition, including branches reached at the second application. | Enumerate the complete declared rule grammar; all teaching-consistent rules agree on every scored target. |
| Contradictory specificity targets | Give unrelated questions an explicit unchanged scope/rule. Same full observable input/state may not carry two incompatible targets. | Cross-check teaching, primary and specificity targets; retain a no-update comparator on the same battery. |

Use exactly the public DEV split roles already available: training and tuning.
Freeze complete rule/assignment/composition/paraphrase fingerprints before
surface rendering. If the finite rule population cannot populate every split
with the required diversity, fail construction rather than silently dropping
subtypes. Report template, branch, event-count, answer-length and support
distributions by split. Do not select tasks by subsequent model success.

Every scored probe must have an offline support certificate identifying the
teaching facts and pure derivation needed to determine its answer. The learner
never receives that certificate, hidden rule parameters or target annotations.
Constructors and a separately implemented checker must agree on targets.
Deleting a required teaching fact or flipping a target must make admission fail.

After structural checks pass, run the contextual/oracle capability screen on
the new DEV version. A failed oracle is an articulation block, not a cortex
refutation. New task semantics invalidate reuse of the old calibration for
new decisions: distributions, margin, power/sample-size and candidate sign-off
must be rebuilt before any later meta-test request.

## C. Repair tool-output scoring before tool competence claims

Version the recorded-output instrument independently. Compare the entire
observed call sequence to the required sequence, including count and order;
reject extra calls. Attach expected parameters to each call, rather than only
the first. File paths are case-sensitive exact fields; write content and edit
operations need observable expected values. Command expectations must be
explicit structured predicates, not arbitrary substrings such as `find`.

Keep accepted JSON/bracket syntax explicit and reject malformed calls rather
than recovering a convenient valid fragment. Result integration must be tied
to the provided tool result and reject a negated answer containing the target
word. Retain this as recorded-output scoring until a separate sandbox executor
verifies filesystem/process outcomes. No real shell or file operation is
needed for these scorer contract checks.

Closure includes all 45 positive episode fixtures, the 45 extra-call foils,
the 21 parameter foils, wrong order/count, malformed syntax, missing content,
wrong second-call parameters, negated answers and empty-output cases. Any
missing expectation must be reported as missing coverage, not a perfect score.

## D. Confirmation curriculum after these audit gates

The R23.5 pilot already rejects blank/extra-prose outputs, enforces teaching
separation and persists full provenance. Preserve its result and its manifest.
The next confirmation is a separately frozen DEV instrument, not a rerun on
already inspected pilot questions.

Proposed bounded confirmation: six fresh rules (two append, two prepend, two
date-order/separator rules), three teaching examples per rule, four independent
qualification inputs and eight fresh confirmation inputs per rule. Select
literal affixes/date permutations and all inputs before any model execution.
Prevent rule and input overlap with the completed pilot where claiming fresh
rules. Require a unique teaching-consistent rule in the declared grammar.

Keep the pilot's optimizer, eight-vector width, 24 updates, seeds 0/1/2,
generation limits, final-checkpoint rule and initial/zeroed/swapped controls.
No width/layer search, early stopping or choosing seeds by confirmation scores.
State reduction and alternative coupling stay separate experiments.

Use the qualification inputs only for a predeclared context-effect admission
check. Report every admitted/rejected rule and its scores; do not replace
failed rules with better-looking ones. Training uses the three teaching
examples only. Confirmation outputs remain untouched until settings and
admission decisions are fixed. A nonpositive confirmation denominator remains
undefined, even if the qualification effect was positive. Preserve raw/context
and unedited model-written-text controls, all failures and actual byte counts.

Report per-rule results and seed variability, using rules as the sampling unit.
Each rule has eight distinct confirmation inputs repeated across three seeds;
those 24 answers are not 24 independent inputs. The existing 30% recovery
reference remains descriptive for this small confirmation; no full-study or
population-wide hypothesis verdict follows from choosing the best rule. The
same-model one-sentence summary is labeled precisely and is not treated as a
test of every possible structured text harness.

## Approval boundary and sequence

Approval of this plan means: activate A1; prepare/review A2's response contract;
implement and structurally validate B and C as separate versioned DEV repairs;
freeze D after those admission checks. A2's actual accepted-alias corpus and
any new numeric threshold require review when materialized. No new performance
claim may reuse old-version scores as if the instrument were unchanged.

The guard repair already applied changes enforcement, not measurement. A1's
patch is ready to apply; B/C/D are concrete designs, not completed code or runs.
Original instruments remain active and unchanged pending human sign-off.
