# Curriculum and evaluation audit — September 12, 2026

**Verdict: BLOCK new scientific conclusions on the defective eval-v2/R20
instruments until versioned repairs pass admission.** The completed R23.5
pilot remains a separate, narrow persistence result. Its normalized exact
scorer does not have the legacy empty-answer bug.

The user requested an audit and improvements before fresh-pattern confirmation.
This audit inspects current code and public data, runs deterministic checks,
and prepares a repair candidate. **No model experiment, remote job, sealed
catalog access or meta-test ran.** Historical score files and frozen
instruments were not modified. The eval guard was repaired locally.

Evidence: [machine-readable audit](2026-09-12_curriculum_eval_audit.json),
[versioned repair plan](../experiments/eval-audit-repair-v1/REPAIR_PLAN.md),
[unapplied eval-v2.3 patch](../experiments/eval-audit-repair-v1/eval_v2_3.patch),
and [candidate shadow checks](../experiments/eval-audit-repair-v1/SHADOW_VALIDATION.json).

## Findings, ordered by consequence

### 1. [P1][CORRECTNESS] A silent model receives 100% on eval-v2

**Location:** `src/oczy/eval_v2/scoring.py:118–138`; live caller
`src/oczy/experiments/organism_curriculum/run_curriculum.py:249–250`.
**Authority:** the frozen curriculum requires correct responses; its matcher
documents meaningful exact/contains/sense matching, not acceptance of silence.

**Evidence:** `probe_matches("", probe, episode)` accepts **120/120 shipped
probes**, comprising 20 contains-mode and 100 sense-mode probes across all six
stages. Empty string is a substring of every target, including through the
sense-mode fallback. A model that emits only EOS therefore can receive full
accuracy. An empty expected target is another invalid case to reject.

The semantic fallback also conflates different senses of the same word:
`fashion` passes an expected `machine learning model`, `git` passes
`tree branch`, and `spreadsheet` passes `biology cell` with semantic mode on.
Both sides need only intersect the union of all senses; they need not match
the same sense. The default nonempty sense matcher also accepts incidental
token overlap, so fixing empty answers alone does not validate free-text
semantics.

**Guard/test gap:** current curriculum/split validators report **zero errors
and zero warnings**. All 28 focused existing scoring/split/protocol/manifest
tests pass; they do not test these adversarial cases. This does not establish
that historical model outputs were empty or quantify historical inflation.

**Remedy/closure:** A1's versioned nonempty guard is ready, with blank 0/120,
expected-answer 120/120 and 0/960 nonempty comparison changes. A2 requires a
separate sense/alias contract and wrong-sense/negation/all-labels response
distribution. **Confidence: high.** Instrument repair remains unapplied.

### 2. [P1][SPEC/CORRECTNESS] R20 scores answers its teaching cannot determine

**Location:** `src/oczy/experiments/meta_cortex/taskgen.py:259–355,
427–495, 590–719, 873–974, 1052–1108`.
**Authority:** Research/20 requires learning from the supplied events and
composition of learned operations; it does not require guessing hidden random
assignments. The generator itself says contextual same-rule/transfer queries
paraphrase taught mappings.

**Evidence:** all 105 public DEV tasks were checked, preserving the original
training/tuning split and targets:

| Defect | Training | Tuning | Total |
|---|---:|---:|---:|
| Untaught contextual same-rule lookup | 37/75 | 7/12 | 44/87 |
| Untaught contextual transfer lookup | 25/60 | 5/10 | 30/70 |
| Untaught independent FSM transfer edge | 52/52 | 7/7 | 59/59 |
| Contextual composition second operand undefined | 30/30 | 5/5 | 35/35 |
| FSM composition requires an untaught action | 30/30 | 5/5 | 35/35 |
| Transformation transfer target underdetermined | 1/60 | 2/10 | 3/70 |
| Transformation composition target underdetermined | 1/30 | 1/5 | 2/35 |

The new transformation check enumerates all **289 rules in the public
generator grammar**, using only teaching examples to filter candidates. Two
tasks retain 16 compatible conditional rules. For example, teaching
`xi→grimxi`, `mu→grimmu`, `kappa→grimkappa` never reveals the vowel-start suffix:
both `alpha→alphadax` and `alpha→alphafep` are consistent with those lessons.
Five scored probes are underdetermined. Witness rules/outputs are retained.

The contextual composition code uses the first output as a second input even
though the two vocabularies are disjoint, then silently substitutes a different
symbol with `.get(..., symbols[-1])`. One training task also requests five
distinct teaching events from four available facts and silently duplicates one.
FSM teaching provides neither action assignments nor an initial goal, while
the family description claims retained-goal behavior.

**Guard/test gap:** deterministic hashes and split fingerprints bind these
tasks but do not certify support or identifiability. Correctly preserving
an invalid target does not make it learnable. Strong oracle scores could not
repair missing information in the actual teaching condition.

**Remedy/closure:** construct probes from covered facts/typed paths, teach
every required branch/action/goal, reject duplicate events, and require a
teaching-only derivation or unique-target certificate before running a model.
Use a new DEV generator/instrument lineage and recompute calibration before
later decisions. **Confidence: high.** No original targets were changed.

### 3. [P1][SPEC/CORRECTNESS] Specificity can penalize the taught correct answer

**Location:** `src/oczy/experiments/meta_cortex/taskgen.py:821–833,
1111–1132`.
**Authority:** Research/20 defines specificity as preservation on unrelated
rules, not reverting a learned answer to a conflicting hidden target.

**Evidence:** **23 public tasks** have byte-identical user questions in a
teaching event and specificity probe but different targets: **9 transformation
tasks and 14 FSM tasks**. Shared v3/v4 bare-answer system instructions are
excluded from this comparison because they do not change task scope.
Example: `Apply the rule to: lambda` is taught as `ltwemmbdtwem`, but
specificity expects `lambda`. The FSM specificity generator picks another
hidden goal as the next-state target while presenting the same state/input
question and no new goal or graph.

**Failure:** retaining a taught answer can count as loss of specificity. A
learner cannot know that the hidden scoring category has changed.
**Guard/test gap:** distinct ProbeKind labels and target hashes do not enter
the model-visible context, so they cannot disambiguate these questions.
**Remedy/closure:** explicit observable unchanged scope plus contradiction
checks across teaching/primary/specificity; flipping the hidden category alone
must not change the correct target. **Confidence: high.**

### 4. [P1][INTEGRITY] Runtime instrument code was outside the effective freeze

**Location:** prior `scripts/eval_guard.py:16–55`, current guard at
`scripts/eval_guard.py:14–103`; `eval/v2/__init__.py:48–84` and manifest.
**Authority:** AGENTS.md freezes scoring, episodes, thresholds and baselines
per version. SPRINT's frozen-instrument requirement includes scoring code.

**Evidence:** the eval-v2 manifest contains six data files and its loader, but
does not hash the active scorer in `src/oczy/eval_v2/scoring.py`, the validator,
the split implementation or the curriculum protocol driver. The old git guard
checked only committed revision differences and missed that moved scorer
path, as well as the newer R20/R24/tool instrument paths. It could also silently
substitute a different range when an explicit requested ref was invalid.

**Repair applied:** guard checks committed, staged, unstaged and untracked
changes; preserves deleted/renamed protected paths; covers current instrument
source paths; and rejects invalid explicit ranges. Learner-only changes remain
permitted. The documented fallback is retained only for an implicit default
whose upstream is absent. This is enforcement code, not a scoring change.

**Closure evidence:** 13 new regression cases fail against the old guard;
all **21 guard tests pass** after the repair. Tests use real temporary git
repositories, including staged-then-reverted content and protected-file rename.
A1's candidate manifest rejects simulated runtime scorer drift in an isolated
copy. Runtime source binding awaits A1 approval. **Confidence: high.**

### 5. [P2][CORRECTNESS] Tool scoring accepts invalid parameters and extra calls

**Location:** `src/oczy/experiments/tool_calling_curriculum/scoring.py:62–80`.
**Authority:** Experiment 08's implemented substrate claims ordered-chain and
parameter correctness; this runner scores recorded outputs, not real effects.

**Evidence:** prefix-only sequence comparison ignores every extra call after
the required prefix. An appended unexpected call passes **45/45 episode
fixtures**. Substring parameter checks accept deliberately wrong parameter
strings in **21/21 parameter-bearing episodes**, including
`WRONG/config.toml.bak` for `config.toml`. Later calls have no per-call parameter
contract; `write` tasks often do not score file contents. Answer keyword
presence alone does not establish grounding or successful execution.

**Guard/test gap:** validation checks counts, IDs and nonempty expectations,
not adversarial outputs. No shell or filesystem action was executed for this
audit; these counts diagnose the scorer only.
**Remedy/closure:** complete call-sequence comparison, typed per-call parameter
and content contracts, explicit malformed/extra/wrong-order/negation foils,
and a distinct executor-outcome gate for real tool competence. **Confidence:
high.** No tool scoring was altered.

### 6. [P2][EVIDENCE] Legacy reports and split names support narrower claims

**Location:** `src/oczy/experiments/organism_curriculum/dataset.py:253–305`,
`run_curriculum.py:492–528`; `correction-benchmark/src/correction_benchmark/scorer.py:56–68`.
**Authority:** distinguish unseen-question transfer from unseen-rule learning;
preserve evidence needed to inspect measured results.

The eval-v2 split correctly implements its declared category-stratified
question split. **19 episodes span dev and holdout** (8 stage-1, 7 stage-2,
2 stage-3, 2 stage-5); teaching rules are not held out. This is conforming for
that protocol, but cannot substantiate R20-style unseen-rule generalization.
The standard report writer saves aggregate pre/post accuracies and teaching
episode outputs but omits raw probe generations and full runtime/scorer
provenance. Some historical results therefore cannot be re-scored faithfully.

The separate legacy correction-benchmark scorer also returns **1.0 for an
empty category** (`Scorer.transfer_score(())`). No such empty-category problem
was inferred for the nonempty R20 pilot; this is a separate legacy boundary.

**Remedy/closure:** precise split labels; a separately frozen whole-rule split
for whole-rule claims; raw per-probe outputs, full condition/provenance/counts;
undefined/absent status for empty metrics. **Confidence: high.** Existing
question split and historical score files remain unchanged.

## Conformance matrix and scope

| Surface / requirement | Authority and implementation | Evidence / disposition |
|---|---|---|
| eval-v2.2 data integrity | AGENTS; eval/v2 manifest/loader | Existing file hashes pass; runtime source coverage incomplete. A1 candidate closes selected runtime-source coverage. |
| eval-v2.2 execution order/category split | Approved July 11 v2.2 protocol; dataset/driver | 28 focused tests pass; category allocations reproduce. No claim that whole rules are held out. |
| R20 learnable held-out/composition tasks | Research/20; public taskgen | MISMATCH: unsupported/ambiguous targets and contradictory specificity. |
| R20 scorer isolation | normalized-exact/v1; calibration.py | Blank and extra prose rejected; this scorer is distinct from the defective eval-v2 matcher. No scorer changes. |
| R24-v3 toy rule identifiability | Registered toy algebra; toy_catalog_v3.py | All 192 public candidate rules uniquely determined after A/B/C. No sealed partition was built or loaded. |
| R24 three independent lessons | Existing result already limits minimality | A and C alone identify all 192 rules; B aliases A. This does not invalidate the registered fixed-three-event result. |
| R23.5 pilot persistence | Approved pilot-v1 manifest and results | Identifiable within declared families; exact scorer and recorded controls intact. Two patterns/four inputs cannot establish broad generalization. |
| R23.5 recovery/bytes | Pilot contract | Date denominator zero remains undefined; numeric state exceeds raw text. No compression claim. |
| Tool curriculum | Experiment 08 implemented recorded-output substrate | Scoring implemented despite stale “unimplemented” index; no claim of live tool execution. Concrete scorer defects above. |
| Older topical/adaptive curricula | curriculum.py, curriculum_dataset.py, plastic-cortex generator | Inventory/source review only. Seeded order or model-targeted training curriculum is not independent held-out evidence; no model/checkpoint loaded. |

## Improvements prepared for the next curriculum

The [repair plan](../experiments/eval-audit-repair-v1/REPAIR_PLAN.md) makes each
change and admission gate explicit. A1 is an applicable, shadow-checked patch;
the R20/tool/confirmation portions are designs awaiting implementation and
approval. They must not be described as completed instrument repairs.

The next curriculum should progress through identifiable single rules,
observable scope changes, typed composition, interference/retention and
cross-session reload. A later stage is not assumed harder merely from its
name. Measure a per-stage distribution with frozen no-context, retrieval and
oracle baselines before using difficulty as a scheduling signal. Adapt lesson
order only inside training; never adapt scored items or thresholds to the
optimizing loop's successes.

For serialization confirmation, use fresh rules and inputs, a distinct
qualification set for contextual uplift, fixed optimization and complete
reporting of ineligible rules. Count rules/inputs, not repeated seed answers,
as independent experimental units. A failed one-sentence text refiner is not
a refutation of every text-harness baseline. The draft R23.5 universal kill
language exceeds what a finite pattern/layer/budget experiment can establish;
that draft is not an active authority for a universal scientific verdict.

## Verification ledger

| Command / check | Result | Meaning |
|---|---|---|
| `.venv/bin/python scripts/audit_curricula_evals.py --public-root /home/nyanpasu/.local/state/oczy/remote-queue/campaigns/r20-int8-dev-calibration-v6/analysis-a8c98d6/instrument/instrument/public --instrument experiments/r23.5-serialization-dev/instruments/v4 --output experiments_logs/2026-09-12_curriculum_eval_audit.json` | Exit 0 | Audit completed; it intentionally reports defects rather than claiming admission. Source/input hashes are in JSON. |
| `.venv/bin/python -m pytest scripts/tests/test_eval_guard.py -q --tb=no` before guard repair | Exit 1; 13 failed, 8 passed | New cases reproduce guard defects against old code. |
| `.venv/bin/python -m pytest scripts/tests/test_eval_guard.py -q` after repair | Exit 0; 21 passed | Committed/workspace path enforcement and explicit-ref handling. |
| `.venv/bin/python -m pytest src/oczy/experiments/organism_curriculum/tests/test_scoring_semantic.py src/oczy/experiments/organism_curriculum/tests/test_split.py src/oczy/experiments/organism_curriculum/tests/test_stage_protocol.py src/oczy/experiments/organism_curriculum/tests/test_manifest_integrity.py -q` | Exit 0; 28 passed | Existing coverage passes despite reproduced scoring defects. |
| `.venv/bin/python scripts/prepare_eval_v2_3_candidate.py --output experiments/eval-audit-repair-v1` | Exit 0 | Candidate only: 120 blank/whitespace rejections, 120 expected passes, 960 nonempty comparisons unchanged; source drift rejected. |
| `git apply --check experiments/eval-audit-repair-v1/eval_v2_3.patch` | Exit 0 | Patch applies cleanly; not applied. |
| `.venv/bin/python scripts/eval_guard.py HEAD...HEAD` | Exit 0 | Current workspace has no active protected instrument changes. |
| ` .venv/bin/ruff check scripts/eval_guard.py scripts/tests/test_eval_guard.py scripts/audit_curricula_evals.py scripts/prepare_eval_v2_3_candidate.py`; `git diff --check` | Exit 0 | Static/whitespace checks; not a scientific result. |

The initial audit compared full message tuples and missed specificity
contradictions because v3/v4 add a common answer-format system message.
The corrected checker compares the actual user questions; both diagnostic
reports are preserved, with the `_initial.json` report superseded. Candidate
preparation initially failed to import the root `eval` package, before model
or instrument execution; fixing the script's import path resolved it. No
failed attempt was represented as a completed scientific experiment.

## Formalization opportunities

1. **Teaching sufficiency:** for the declared finite grammar H and teaching E,
   require all h consistent with E to agree on each scored target. Existing
   exhaustive enumeration already finds concrete witnesses; use it as an
   admission invariant. This is bounded verification, not proof over arbitrary
   real-world rules. Cost: small for 289 transformation rules.
2. **Typed graph composition:** every needed edge/action/goal is either taught
   or derivable by a public rule; every intermediate output has the next
   operation's input type. Small graph traversal plus target perturbation
   checks are sufficient; no solver installation is needed.
3. **Observable-input consistency:** equal task scope, prior teaching and
   model-facing query imply equal targets, independent of hidden probe category.
   A dictionary/counterexample check catches this class cheaply.
4. **Artifact closure:** declared manifests cover every directly owned
   scoring/split/protocol dependency; a changed source byte rejects loading.
   Real filesystem/git tests are a better first step than a heavyweight proof.

## Open questions and residual risk

No broad real-model difficulty or power study ran. No human-validated alias
corpus exists yet for a stricter natural-language sense scorer. The pilot's
finite-grammar uniqueness does not imply unrestricted rule identifiability.
R24's public algebra audit does not inspect or re-adjudicate sealed results.
The local guard repair is not committed or deployed, and protection lists
still require maintenance as instrument ownership moves. Repaired instrument
scores require new version/provenance and must not overwrite historical tables.

The next approval is concrete: the linked versioned repair plan and ready
eval-v2.3 patch. AGENTS.md requires **“Get human sign-off”** before changing an
eval. The earlier permission to audit and proceed does not specify these newly
discovered scoring/target changes; the proposal makes their scope reviewable.
