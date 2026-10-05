# R20 human sign-off chain — task-support repair and decoder adoption

> **SIGNED 2026-10-04** (user authorization relayed by the manager). Verbatim
> signatures and the conditions attached to them are recorded in
> [§S — Signatures](#s-signatures) below. The scope is unchanged and remains
> **instrument construction and adoption only**: no scientific verdict, no
> threshold selection from results, and no meta-test authorization is implied by
> any signature here. S1 and S2 authorize proceeding to G1 only. S3 is
> conditional and its condition was discharged before adoption (evidence:
> [`../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md`](../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md)).

This document states exactly what needs a human decision, what evidence
accompanies each item, and what happens after sign-off. Before 2026-10-04 no
autonomous session could grant any of it (AGENTS.md: "Get human sign-off";
project-scope-registry hard limits: no meta-test access, no full R20 rerun before
sign-off, frozen scores intact, historical failures keep reproducing).

## S. Signatures

Recorded verbatim from the user authorization relayed by the manager on
2026-10-04 (kanban card `t_5a7b48de`):

| Item | Verdict | Recorded |
|---|---|---|
| S1 | approve | `S1 approve (user via manager, 2026-10-04)` |
| S2 | approve | `S2 approve (user via manager, 2026-10-04)` |
| S3 | adopt-with-version-bump | `S3 adopt-with-version-bump (user via manager, 2026-10-04)` |

**What these signatures cover — instrument construction and adoption only.**

Two distinct version axes are in play, and the records below use them
consistently:

- **Task-generator lineage:** `oczy/meta-cortex/taskgen/v1-dev` (the historical
  generator) → `v2-dev` (the task-support repair, adopted by S1).
- **Instrument:** `meta_cortex/v2` (frozen, all historical scores) →
  `meta_cortex/v3` (the adopted DEV instrument, frozen at gate G1 on 2026-10-05).

1. **S1 approve.** The `oczy/meta-cortex/taskgen/v2-dev` lineage may be adopted
   as the new DEV instrument lineage and versioned into a frozen `meta_cortex/v3`
   instrument. The v1 generator stays frozen; **v1-lineage and v2-lineage task
   scores are not comparable as a causal learner improvement**, and therefore
   neither are `meta_cortex/v2` and `meta_cortex/v3` results.
2. **S2 approve.** The versioned batch decoder
   (`generation_v2.py`, qualified in
   `2026-09-13_decoder_parity_dev_v1.md`) may be adopted **only inside a new
   versioned batched experiment**, with its own provenance. Old organ behavior
   and all old scores remain unchanged.
3. **S3 adopt with a version bump — conditional, condition discharged.** The
   uncommitted `organ.py` `skip_special_tokens=True` decode change is adopted
   **with a version bump**, conditional on first demonstrating in the experiment
   logs exactly which prior recorded scores the decode change alters, and on
   recording any comparability break as a limitation with the change quarantined
   to v2-lineage runs.

**Conditions attached by the user, carried forward verbatim:**

- S3's condition: demonstrate in the experiment logs exactly which prior recorded
  scores the decode change alters; if any historical-score comparability breaks,
  record it as a limitation and keep the change quarantined to v2-lineage runs.
- G1 may proceed on S1 + S2. G5 still requires explicit human sign-off on the
  exact candidate manifest hash.

**Explicitly NOT authorized by these signatures (unchanged):** no meta-test
access; no model runs beyond DEV; no score or threshold changes; no reuse of v1
calibration numbers for v2-decisions (i.e. for decisions about the v2-lineage
instrument `meta_cortex/v3`); no full R20 rerun; no change to `eval/`,
`src/oczy/eval_v2/`, or the tool curriculum. The eval-v2 A2 (sense/alias
contract) and C (tool-output scoring) repairs remain unapplied and unsigned.

---

## A. What needs signing

### S1 — Adopt the R20 task-support repair as a new DEV instrument lineage

Approve `oczy/meta-cortex/taskgen/v2-dev`
([`taskgen_v2.py`](../../src/oczy/experiments/meta_cortex/taskgen_v2.py)) as the
task-support-repaired public DEV generator, replacing v1 **for future runs
only**. Concretely, the signer approves:

1. The lineage split: v1 (`taskgen.py`, its rendered tasks, thresholds and all
   recorded scores) stays frozen and unchanged; v2 is a new versioned lineage
   with its own deterministic stream bound to its own schema string.
2. The six construction repairs and their semantics: taught-only lookups,
   typed two-step contextual chains, taught FSM action mapping and goal with a
   goal-dependent composition decision (registered paired goal counterfactual),
   branch-covered and uniquely determined transformation targets,
   explicit-scope specificity probes, and duplicate-free teaching.
3. The semantic redefinitions those repairs require, which are the parts a
   reviewer should read hardest:
   - family C "transfer" means paraphrase transfer over taught transitions
     (as family A already meant); path transfer is measured by the composition
     probes;
   - family C composition is a goal-dependent decision with the goal and the
     `halt` convention stated in the probe (v1 claimed goal retention while
     teaching neither goal nor actions);
   - contextual outputs are drawn from the symbol vocabulary so composition is
     well-typed;
   - specificity probes state their unchanged scope in the question and score
     that stated scope (v1 scored a hidden category flip).
4. Acceptance that v1 and v2 scores are **not comparable as a causal learner
   improvement** (REPAIR_PLAN B): new task semantics invalidate reuse of the
   old calibration for new decisions.

Evidence for S1:

| Evidence | Path |
|---|---|
| Defect counts before/after, certificates, mutation checks, distributions | [`DRY_RUN.json`](DRY_RUN.json) |
| Independent checker (rendered-text derivation + 289-rule grammar enumeration) | [`scripts/r20_task_support_check.py`](../../scripts/r20_task_support_check.py) |
| Focused tests (12) | [`src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py`](../../src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py) |
| Design and reproduction commands | [`README.md`](README.md) |
| Original audit that defined the defects | [`2026-09-12_curriculum_eval_audit.md`](../../experiments_logs/2026-09-12_curriculum_eval_audit.md), [`2026-09-12_dev_teaching_coverage.json`](../../experiments_logs/2026-09-12_dev_teaching_coverage.json) |
| Repair design this implements | [`experiments/eval-audit-repair-v1/REPAIR_PLAN.md`](../eval-audit-repair-v1/REPAIR_PLAN.md) §B |

What signing S1 does **not** authorize: meta-test access, any model run, any
score change, any threshold change, or reuse of v1 calibration numbers.

### S2 — Adopt the versioned batch decoder through a new versioned experiment

Approve adoption of the versioned batch path
([`generation_v2.py`](../../src/oczy/experiments/meta_cortex/generation_v2.py),
qualified in [`2026-09-13_decoder_parity_dev_v1.md`](../../experiments_logs/2026-09-13_decoder_parity_dev_v1.md):
new batch path matches scalar on 32/32 single-item and 32/32 padded-batch
responses including a learned bank; the original batch path matches 21/32; the
ablation isolates fixed continuation positions) **only inside a new versioned
batched experiment**. Concretely:

1. Old organ behavior and all old scores remain unchanged; the original batch
   path is not retro-qualified.
2. Scalar-only learning failures are not explained away by the batch repair.
3. Any experiment adopting the versioned batch decoder does so as its own
   versioned run with its own provenance, not by editing a historical run.

Evidence for S2: the decoder-parity DEV record above (QUALIFIED, bounded
repair) and its artifacts under `experiments_logs/artifacts/`.

### S3 — The uncommitted `organ.py` decode change (SIGNED: adopt with a version bump)

The working tree carried an **uncommitted** change to
[`organ.py`](../../src/oczy/experiments/meta_cortex/organ.py) that decodes
generated ids with `skip_special_tokens=True` (scalar and batch paths). It
changes the answer *text* the frozen exact-text scorer sees, so it is
measurement-adjacent. It was not produced or applied by the task-support work.

**Verdict recorded 2026-10-04: adopt WITH a version bump**, conditional on the
comparability demonstration the user required. That condition was discharged on
2026-10-05 and is recorded in
[`../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md`](../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md):
the change alters the answer *text* of 11 of 68 recorded probe outputs and moves
**0 recorded scores**. The change is therefore quarantined to v2-lineage runs:
no v1-lineage score may be re-read through the new decoder.

### S4 — Successor instrument authorization (2026-10-05)

**Recorded 2026-10-05: user via manager.** Verbatim entry:

> User via manager 2026-10-05: Option A funded, successor named
> `meta_cortex/v4-r20`, max_new_tokens move approved, evidence push to remote
> approved.

Relayed on kanban card `t_37e96ee1`, superseding the open questions on
`t_5670e505`. What this authorization covers:

1. **Option A funded** — build the successor instrument and re-run G2.
2. **Name** — the user asked for "v4"; plain `meta_cortex/v4` collides with the
   approved `experiments/r23.5-serialization-dev/instruments/v4` (manifest
   `d58adb749f4112f2…`) and `meta_cortex/v3` is doubly claimed, so the successor
   id is the scope-qualified **`meta_cortex/v4-r20`**. The collision is recorded
   in the successor's `DEFINITION.json` under `naming`.
3. **`max_new_tokens` moved 32 → 128** — a **signed field relative to v3**.
   Justified by the observed truncation (10 of 15 v3 oracle generations cut
   mid-preamble at 32) and bounded at 4x the longest observed preamble. Recorded
   in `DEFINITION.json` under `max_new_tokens_change`. **v3 and v4-r20 scores are
   NOT comparable as a causal improvement** (same rule as the v2→v3 transition).
4. **Prompt amendments carried in** — the 2026-09-11 Amendment A (bare-answer
   system instruction) and Amendment B (complete transformation oracle rule
   descriptions) are in the successor's prompt registry.
5. **Evidence push to the remote approved** — commit and push the experiment
   evidence, logs and instrument definitions to `origin`.

**What this authorization does NOT cover (unchanged):** no meta-test access; no
threshold, margin or power selection (that is G4, signed at G5); no edit to
`eval/`, `src/oczy/eval_v2/` or the tool curriculum; no change to
`taskgen.py` / `taskgen_v2.py`; no reopening of any frozen v1/v2/v3 score. G5
still requires explicit human sign-off on the exact candidate manifest hash.

**Outcome of the funded work (same session).** The successor froze at
G1-equivalent (`definition_sha256`
`fd2d8db9012683d4955879481c7f5a91eac16d3c82cde018846605c9378f1031`) and **G2
PASSED 7/15** (criterion `oracle_context correct > 0`; v3 had failed 0/15). See
[`2026-10-05_r20_g1_v4_r20_instrument_freeze.md`](../../experiments_logs/2026-10-05_r20_g1_v4_r20_instrument_freeze.md)
and
[`2026-10-05_r20_g2_v4_r20_oracle_screen.md`](../../experiments_logs/2026-10-05_r20_g2_v4_r20_oracle_screen.md).
The sequence stopped before G3: the pass is real but the transformation family's
oracle 0/5 is an ambiguity the card requires a human to rule on.

---

## B. What is explicitly NOT in this chain

- **No meta-test access.** The meta-test remains blocked until the whole
  post-sign-off sequence below completes and a candidate is signed on its exact
  manifest hash.
- **No thresholds, margins, or power numbers are being chosen here.** They are
  recomputed after adoption (gate G4) and signed at G5.
- **No full R20 rerun** may start on this document alone.
- **eval-v2 repairs A2 (sense/alias contract) and C (tool-output scoring)
  remain unapplied and unsigned**, as does any change to `eval/`,
  `src/oczy/eval_v2/`, or the tool curriculum.
- **No sealed generator, sealed catalog, or meta-test task file** was opened in
  producing S1's evidence (the dry run reads only the public DEV splits).
- The R20 local-reproduction gap (four nonzero state hashes still differ after
  the model-path hash fix) is unresolved and is not covered by any signature
  here.

---

## C. Post-sign-off gate sequence

Order is fixed; a gate that fails stops the sequence and is recorded as a
blocker (nulls and refutations are results).

| Gate | What must pass | Evidence it produces | Failure meaning |
|---|---|---|---|
| G1 Freeze | Version the v2-dev lineage into a frozen instrument: `meta_cortex/v3` identity, materialized public DEV splits, task fingerprints, leakage/support audit, manifest hash; add the new files to the eval guard's protected list | instrument manifest + guard test that a source byte change rejects loading | instrument not frozen: no further gate |
| G2 Oracle capability screen | The contextual/oracle capability screen on the new DEV version (can the frozen organ express the task under oracle control?) | oracle screen report | articulation block (mouth–cortex protocol failure), **not** a cortex refutation |
| G2′ Oracle capability screen (successor) | The same screen re-run on a successor instrument that carries the approved 2026-09-11 prompt amendments and the authorized `max_new_tokens` (S4). Criterion unchanged: `oracle_context correct > 0` | oracle screen report | articulation block; the successor instrument stays frozen either way |
| G3 Learnability recheck | DEV-side learnability on v2 tasks (teaching fits, no-update comparator on the same battery) | learnability report | learner diagnosis; instrument stays |
| G4 Calibration and power | Fresh DEV repeatability distributions, margin, power/sample-size for the *new* task semantics; v1 calibration may not be reused | calibration + power report | no powered design: no candidate |
| G5 Candidate sign-off | Explicit human sign-off on the exact candidate manifest hash, margin, and task count (plus S2's decoder version for any batched run) | signed manifest hash | meta-test stays blocked |
| G6 Meta-test | One-shot held-out meta-test with all parameters frozen and only fast/slow cortex state mutable; causal controls (no-update, retrieval/context, oracle, random/untrained, feedback-shuffled, zeroed, swapped), trace deletion, multiple seeds, trajectories, CIs, per-byte accounting | adjudicated result (accept/diagnose/refute) | recorded per §9 of CURRENT_STATE.md |

Between G5 and G6, the meta-test protocol itself is unchanged from
Research/20's registered design: retrieval excluded from the primary arm,
retrieval/vanilla/oracle comparators still reported, no episode-specific fixes.

---

## D. How to sign

Reply with the item id and verdict, e.g. `S1 approve`, `S2 approve`, `S3
revert` — plus any conditions. Approval of S1 and S2 means: proceed to G1,
nothing more. If a condition is attached (e.g. "adopt S1 but keep family C
transfer as v1 path transfer"), the condition is recorded here as an amendment
and the affected checks are re-run and re-offered before G1 freezes.

Provenance of this document: prepared 2026-10-04 by an autonomous session on
kanban card `t_b2e4657a`; evidence listed in §A was produced by the commands in
[`README.md`](README.md) with exit codes and counts recorded in
[`DRY_RUN.json`](DRY_RUN.json).
