# R20 task-support repair — design and evidence (v1)

**Status: SIGNED 2026-10-04 (S1 approve, user via manager) and ADOPTED.**
The human sign-off is recorded in
[§S of SIGNOFF_CHAIN.md](SIGNOFF_CHAIN.md#s-signatures), and the adopted
instrument was frozen as `meta_cortex/v3` at gate G1 on 2026-10-05 (record:
[`2026-10-05_r20_g1_instrument_freeze.md`](../../experiments_logs/2026-10-05_r20_g1_instrument_freeze.md),
`definition_sha256`
`ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee`).

The sign-off covers **instrument construction and adoption only**. It grants no
scientific verdict, selects no threshold from results, and does not authorize
meta-test access. This directory's evidence was produced with no model run, no
meta-test, no calibration, and no sealed file opened — that remains true of the
evidence below.

The gate sequence continues at **G2** (oracle capability screen on the v3
instrument). G3 is learnability, G4 is calibration/power, and **G5 still requires
explicit human sign-off on the exact candidate manifest hash.**

Human sign-off chain: [SIGNOFF_CHAIN.md](SIGNOFF_CHAIN.md).

## What the defect was

The 2026-09-12 public DEV audit
([coverage audit](../../experiments_logs/2026-09-12_dev_teaching_coverage.json),
[curriculum/eval audit](../../experiments_logs/2026-09-12_curriculum_eval_audit.md))
found that the historical generator (`src/oczy/experiments/meta_cortex/taskgen.py`,
lineage `oczy/meta-cortex/taskgen/v1-dev`) scores answers its teaching cannot
determine. Recorded counts, all reproduced exactly by the dry run below:

| Defect (train + tuning) | Count |
|---|---|
| Contextual same-rule lookup not taught | 44 / 87 |
| Contextual transfer lookup not taught | 30 / 70 |
| FSM transfer edge not taught | 59 / 59 |
| Contextual composition second operand undefined | 35 / 35 |
| Contextual composition first lookup not taught | 18 / 35 |
| FSM composition requires an untaught action | 35 / 35 |
| FSM action mapping never taught | 35 / 35 tasks |
| FSM goal never taught | 35 / 35 tasks |
| Transformation transfer target underdetermined | 3 / 70 |
| Transformation composition target underdetermined | 2 / 35 |
| Specificity probe contradicts its own teaching | 23 |
| Silently duplicated teaching event | 1 task |

## What the repair is

A **separate generator lineage**, `oczy/meta-cortex/taskgen/v2-dev`, implemented
in [`src/oczy/experiments/meta_cortex/taskgen_v2.py`](../../src/oczy/experiments/meta_cortex/taskgen_v2.py).
The v1 generator, its rendered tasks, its thresholds and every recorded v1
score are untouched (`taskgen.py` and `contracts.py` are eval-guard protected;
this run modifies neither).

Per defect class, the v2 constructor behavior:

1. **Untaught lookups.** Every same-rule and transfer probe is drawn from the
   task's own teaching pairs/transitions. Family C "transfer" is redefined as
   paraphrase transfer over taught transitions (matching what family A already
   meant by transfer); multi-step transfer is measured by the composition
   probes, which traverse two taught edges. v1's "guess an independently
   sampled edge" transfer is gone.
2. **Undefined contextual composition.** Outputs are drawn from the symbol
   vocabulary, so a first lookup's output is a legal second lookup key (typed
   chain), and both lookups of the chain are taught. v1 fed a disjoint
   `_OUTPUTS` token back as a symbol and then silently substituted
   `symbols[-1]`.
3. **Untaught actions / absent goal.** Family C teaching now includes the final
   state's action mapping and the goal. The composition probe states the goal
   and the goal-terminal convention (`halt`) and scores `"<final state>
   <action>"`. A registered paired counterfactual changes the goal (in the
   teaching and in the probe) and the correct answer changes — the decision is
   goal-dependent, not a hidden goal-free lookup. `halt` is never a state
   action, so the answer strictly depends on the goal.
4. **Underdetermined transformation targets.** Conditional tasks must teach
   both branches (vowel-initial and consonant-initial operands) before either
   branch is scored, and construction refuses (``SupportError``) any scored
   target that is not agreed on by every rule of the public 289-rule grammar
   consistent with the teaching.
5. **Contradictory specificity targets.** Specificity probes state their
   unchanged scope explicitly in the question ("Unrelated rule …", "Unrelated
   machine …"), and the checker enforces that no observable question carries
   two different targets anywhere in a task (teaching included). v1 scored a
   hidden category flip — e.g. `Apply the rule to: lambda` taught as
   `ltwemmbdtwem` but scored as `lambda`.
6. **Silently duplicated teaching events.** Teaching facts are sampled without
   replacement; the declared event count always equals the number of distinct
   teaching records.

Every scored probe carries an **offline support certificate**
(``SupportBundle``) naming the exact teaching sentences and derivation that
determine its answer. Certificates are audit metadata: never rendered, never
model-facing, never given to the learner.

## Independent verification

[`scripts/r20_task_support_check.py`](../../scripts/r20_task_support_check.py)
is deliberately separate from both generators. It parses **rendered teaching
and probe text only**, re-derives every scored target from that text (including
an independent enumeration of the 289-rule transformation grammar), and
checks the certificates against the rendered text rather than trusting them.
It also runs mutation checks: deleting a required teaching fact or flipping a
target must make admission fail. The two mutation families are **not equally
discriminating** and are counted separately below: the delete-a-fact checks can
genuinely fail, while the flip-target check cannot fail once a certificate
verifies (the checker's derivation reads the recorded expected answer only in
its pre-learning-baseline branch, and flip-target rows are generated only for
non-baseline certificates, so on every flip row the derivation is independent
of the flipped answer and a flipped target always disagrees) — the real
guarantee for target fidelity is the independent derivation disagree-check.
Two caveats bound that claim. A pre-learning-baseline probe ever certified as
non-baseline would break the independence just described: its flip-target row
would be judged by a derivation that reads the recorded expected answer, so
that row would not be independent of the flipped answer. And a flip-target row
is emitted per non-baseline certificate only when that certificate reaches the
mutation stage: a certificate that fails earlier (probe not found, target
mismatch, absent required teaching fact, failed or disagreeing derivation)
still counts in the certificate totals but emits no flip-target row, so the
identity `flip_target == certificates - baseline_not_required` holds only for
a run in which every non-baseline certificate reaches the mutation stage —
true for this run (758 == 933 - 175) and for the tests' clean fixture, not for
every configuration.

## Reproduce

```bash
.venv/bin/python scripts/r20_task_support_check.py dry-run \
  --output experiments/r20-task-support-repair-v1/DRY_RUN.json \
  --public-root /home/nyanpasu/.local/state/oczy/remote-queue/campaigns/r20-int8-dev-calibration-v6/analysis-a8c98d6/instrument/instrument/public
.venv/bin/python -m pytest src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py \
  src/oczy/experiments/tests/test_meta_cortex_taskgen.py -q
```

The dry run exits nonzero unless (a) the unchanged v1 generator reproduces
every recorded 2026-09-12 defect count exactly, and (b) every defect count is
zero on the v2 lineage, and (c) every support certificate verifies and every
mutation is detected. The optional `--public-root` cross-checks the in-memory
v1 rebuild against the materialized public instrument (identical catalog
digest `c0034bcde0d21e05d151b6a08d34f8cad95c7e4317c8eb919ec3d2fa3fdabb79`).

Current result (see [DRY_RUN.json](DRY_RUN.json)):

| Check | Result |
|---|---|
| v1 defect counts vs recorded audit | reproduced exactly for the 11 defect classes pinned in `EXPECTED_V1_DEFECTS` ([`scripts/r20_task_support_check.py`](../../scripts/r20_task_support_check.py)) — i.e. every row of the defect table above except "Contextual composition first lookup not taught", which the audit records (18 / 35) but the dry run deliberately does not pin |
| v2 defect counts | 0 in every class |
| v2 support certificates — verified (derivation-backed) | 758 / 758 |
| v2 support certificates — pre-learning baseline | 175 of 933 total, `baseline_not_required` by design (no derivation is claimed for them) |
| v2 mutation checks — delete a required fact / teaching set | 688 / 688 detected (discriminating: these can fail) |
| v2 mutation checks — flip target | 758 / 758 detected (non-discriminating by construction, see above; 758 is every non-baseline certificate of this run — 933 − 175 — each of which reached the mutation stage) |
| materialized instrument cross-check | identical catalog digest |

## What this repair does NOT do

- It does not change v1, its recorded scores, its thresholds, or any historical
  result. "Before" numbers are counts of construction defects, not re-scored
  model outputs.
- It does not run a model, a training step, or a full R20 rerun; no meta-test,
  calibration, or sealed task file was opened.
- It does not apply the eval-v2 repairs A2 (sense contract) or C (tool-output
  scoring), and does not touch `eval/`, `src/oczy/eval_v2/`, or the tool
  curriculum. Those remain separately unapplied and out of scope here.
- It does not adopt the versioned batch decoder; that is a separate signable
  item in [SIGNOFF_CHAIN.md](SIGNOFF_CHAIN.md).
- New task semantics invalidate reuse of the old calibration for new
  decisions: distributions, margin, power/sample size and candidate sign-off
  must be rebuilt after adoption (gate sequence in SIGNOFF_CHAIN.md).

## Files

| File | Role |
|---|---|
| `../../src/oczy/experiments/meta_cortex/taskgen_v2.py` | repaired generator, lineage `oczy/meta-cortex/taskgen/v2-dev` |
| `../../scripts/r20_task_support_check.py` | independent checker + dry-run CLI |
| `../../src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py` | focused tests |
| `DRY_RUN.json` | dry-run report (before/after counts, certificates, distributions) |
| `SIGNOFF_CHAIN.md` | what needs human signing and the post-sign-off gate sequence |
