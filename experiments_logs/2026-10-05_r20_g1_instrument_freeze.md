# R20 gate G1 — freeze the `meta_cortex/v3` DEV instrument (2026-10-05)

**Gate:** G1 (freeze) — **PASSED**.
**Classification:** DEV_INSTRUMENT_FREEZE.
**Authorization:** user sign-off S1 + S2 + S3 relayed by the manager
2026-10-04, recorded in
[`SIGNOFF_CHAIN.md`](../experiments/r20-task-support-repair-v1/SIGNOFF_CHAIN.md).
Scope of that sign-off: instrument construction and adoption only.

No model was run, no optimizer step was taken, no threshold, margin or power
value was chosen, no v1 score was touched, and no sealed or meta-test payload
was generated or opened.

## What G1 froze

`meta_cortex/v3` is the `oczy/meta-cortex/taskgen/v2-dev` lineage (the
task-support repair) frozen as a DEV instrument, differing from `meta_cortex/v2`
in exactly two ways:

1. **Task semantics** — public DEV tasks come from
   `taskgen_v2.build_dev_catalog_v2` instead of `taskgen.build_dev_catalog`.
2. **Decoder** — `skip_special_tokens=True` (S3, adopted with a version bump
   after its condition was discharged in
   [`2026-10-05_r20_signoff_s3_comparability.md`](2026-10-05_r20_signoff_s3_comparability.md)).

Everything else is inherited **byte-for-byte** from the frozen v2 instrument, and
the freeze asserts that rather than assuming it: if any registry hash drifted, it
fails closed.

| Role | Frozen value |
|---|---|
| `definition_sha256` | `ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee` |
| `dev_view_sha256` | `593090c405758617ae9f750cf9c1075e999c92f8c7b677affcb04c3855b3ca26` |
| `calibration_view_sha256` | `102b96af65b764812a051c3d88e3f238887367ebc5f6fd8adc60346431deb711` |
| catalog digest | `33210eb474a25925815c1112e2d6c6ea889b349c8d137fbcbe0aacac5f035219` |
| support-bundle digest | `1195e78362671aecc692d08030c34c6692bd9e00fe73c76023027fa1961a3a62` |
| DEV seed table | `c3672ed78f3a0dbd3f9da7ab3b8cc2f1d6370a9a8f3753b81edeccefd4c2fc90` |
| probe counts | `4c643a8808481355f5463e1ec7efd1f3fa53e845a679bb4c3ece96c1ae0348e5` |
| source commit | `9031ce2ca95022f95cdad515e2071855eef1d8ac` |

Materialized task counts: **90** `meta_train`, **15**
`meta_validation_tuning`, **90** `meta_validation_calibration` (3 families ×
30/5/30, root seed 20260709, events 2–5).

Inherited v2 registry hashes, asserted unchanged:

| Registry | SHA-256 |
|---|---|
| prompt | `db624922bf4f67e3e1011b5530ade4111b479ef05391ca0a61e375cac2339735` |
| scorer | `e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742` |
| endpoint | `669d6130075e429264a6c9eec470bbf82663559433781adcffa6ed942c9ca796` |

Organ identity bound to the instrument is the historical
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`
(`Qwen/Qwen2.5-0.5B-Instruct`, INT8 weight-only). This hash was **reproduced on
this machine**, not assumed — see the organ-identity probe below.

## Commands, exit codes and artifacts

### 1. Frozen organ identity reproduced locally

```bash
R=/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity
OCZY_REMOTE_CPU_ONLY=1 OCZY_MODEL_DIR=/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
HF_HUB_OFFLINE=1 PYTHONPATH=/home/nyanpasu/Desktop/code/kinoSoft/oczy/src \
bwrap --die-with-parent --unshare-net --tmpfs / --ro-bind /usr /usr --symlink usr/bin /bin \
  --symlink usr/lib /lib --symlink usr/lib64 /lib64 --ro-bind /etc /etc --ro-bind /home /home \
  --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind $R/model /kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
  --bind $R $R --chdir /home/nyanpasu/Desktop/code/kinoSoft/oczy -- \
  $R/runtime/bin/python -m oczy.experiments.meta_cortex._organ_identity_probe
```

Exit code **0**. Output: `{"organ_hash": "a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea"}`.
Runtime: Python 3.12.13, torch 2.10.0+cpu, torchao 0.17.0, transformers 5.0.0,
tokenizers 0.22.2 — the recorded historical runtime. The hash is stable across
the probe's before/after check.

### 2. Freeze + independent verification + checker

```bash
.venv/bin/python scripts/freeze_r20_v3_instrument.py \
  --output experiments/r20-taskgen-v3-dev/instrument \
  --report /tmp/g1_report_final.json
```

Exit code **0**, 0.6 s. This command materializes the instrument, then re-verifies
the frozen tree independently, then runs the independent checker over the
materialized public tasks.

Report: `/tmp/g1_report_final.json`
(`schema: oczy/r20-g1-freeze/v1`, `passed: true`).

Artifact: `experiments/r20-taskgen-v3-dev/instrument/` — 11 hash-listed public
files, 604 KB:

```
DEFINITION.json
public/DEV_VIEW.json
public/CALIBRATION_VIEW.json
public/audits/leakage_summary.json
public/chat_template.txt
public/endpoints.json
public/generator.json
public/probe_counts.json
public/prompts.json
public/scorers.json
public/seeds.json
public/tasks/meta_train.jsonl
public/tasks/meta_validation_calibration.jsonl
public/tasks/meta_validation_tuning.jsonl
```

### 3. Determinism

The freeze was materialized three times into independent directories
(`/tmp/g1_freeze_1/instrument`, `/tmp/g1_repro/instrument` and
`experiments/r20-taskgen-v3-dev/instrument`). All three produced
`definition_sha256`
`ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee`, and
`diff -r /tmp/g1_freeze_1/instrument experiments/r20-taskgen-v3-dev/instrument`
reported zero differences (exit 0, 0 lines). The freeze is a deterministic
function of the lineage, not of the directory or of the run.

### 4. Independent re-verification of the committed tree

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; \
  from oczy.experiments.meta_cortex.instrument_v3 import verify_v3_definition; \
  print('VERIFIED', verify_v3_definition(Path('experiments/r20-taskgen-v3-dev/instrument')).definition_sha256)"
```

Exit code **0**: `VERIFIED ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee`.

`verify_v3_definition` checks, failing closed on any of them: schema and
identity; the definition self-hash recomputed from canonical JSON; every listed
file's SHA-256 and byte size; no unlisted file anywhere in the tree; no sealed
payload and no `sealed/` directory; all three registries equal to their frozen v2
values; both view self-hashes recomputed and both views binding this exact
definition; the DEV view referencing no held-back file and the calibration view
referencing no train/tuning file; and — importantly — a **fresh rebuild of the v2
lineage** whose rendered task bytes must equal the frozen task files exactly.

### 5. Guard tests: a source byte change rejects loading

```bash
.venv/bin/python -m pytest src/oczy/experiments/tests/test_meta_cortex_instrument_v3.py -q
```

Exit code **0**: **18 passed**. Coverage includes a single-byte flip in each of
the three task files, each registry, `probe_counts.json` and the leakage summary
(all rejected with `File hash mismatch`), a mutated `DEFINITION.json` count
(rejected on self-hash), a re-signed definition that claims a different scorer
registry (rejected on `registry drift` — proving the v2 registry binding is
enforced independently of the self-hash), an unlisted extra file, a `sealed/`
directory, and overwrite refusal.

### 6. Regression surface

```bash
.venv/bin/python -m pytest src/oczy/experiments/tests/test_meta_cortex_taskgen.py \
  src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py \
  src/oczy/experiments/tests/test_meta_cortex_instrument.py \
  src/oczy/experiments/tests/test_meta_cortex_generation_text.py \
  scripts/tests/test_eval_guard.py -q
```

Exit code **0**: **228 passed** — including the v1 generator tests, the v2 support
tests, the v2 instrument tests and the generation-text tests that cover the
`skip_special_tokens` decode paths.

With the 18 new v3 guard tests added to the same command, the final combined run
was exit code **0**, **256 passed**.

### 7. Eval guard protection (a G1 requirement)

The frozen tree, the v2 generator lineage, the v3 instrument module, the organ
decode path, the identity probe, the checker and the freeze CLI were added to
`scripts/eval_guard.py`'s protected set.

```bash
.venv/bin/python scripts/eval_guard.py --allow     # exit 1: EVAL_CHANGE_APPROVED unset
```

Exit code **1**, refusing, as designed. The guard now lists every frozen v3 file
(`experiments/r20-taskgen-v3-dev/instrument/...`) and the sign-off record itself.
`EVAL_CHANGE_APPROVED=1` remains the only way past it, which is the point: the
optimizing loop must not be able to edit the instrument it is measured by.

Guard tests: `.venv/bin/python -m pytest scripts/tests/test_eval_guard.py -q`
→ exit code **0**, **31 passed**, including five new cases that tamper with the
frozen v3 tree and the sign-off record.

### 8. Lint

```bash
.venv/bin/python -m ruff check src/oczy/experiments/meta_cortex/instrument_v3.py \
  src/oczy/experiments/meta_cortex/_organ_identity_probe.py \
  scripts/freeze_r20_v3_instrument.py \
  src/oczy/experiments/tests/test_meta_cortex_instrument_v3.py
```

Exit code **0**: `All checks passed!`

## Independent checker results (over the frozen v3 public tasks)

The checker is `scripts/r20_task_support_check.py`
(SHA-256 `4e501038bbab53951ef74aa2e744ec7c84c03b6e6fd1074369d36c19de7e06cb`),
which is separate from both generators and re-derives every scored target from
rendered text.

- Tasks parsed: **105**; parse defects: **0**.
- All **22** defect classes **0**, i.e. `all_defect_counts_zero: true`:
  `composition_second_operand_undefined`, `contextual_composition_chain_untyped`,
  `contextual_composition_first_lookup_untaught`,
  `contextual_composition_second_lookup_untaught`,
  `contextual_same_rule_lookup_untaught`, `contextual_transfer_lookup_untaught`,
  `duplicate_teaching_event`, `fsm_action_mapping_untaught`,
  `fsm_composition_action_untaught`, `fsm_composition_edge_untaught`,
  `fsm_composition_goal_untaught`, `fsm_composition_underivable`,
  `fsm_goal_untaught`, `fsm_same_rule_lookup_untaught`,
  `fsm_transfer_edge_untaught`, `question_target_contradiction`,
  `specificity_contradiction`, `specificity_scope_unstated`, `target_mismatch`,
  `transform_composition_underdetermined`, `transform_same_rule_underdetermined`,
  `transform_transfer_underdetermined`.
- Support certificates: **933** total, **758** verified (derivation-backed),
  **175** pre-learning baselines (`baseline_not_required` by design), **0**
  failed.
- Mutation checks: **1446/1446** detected, split by kind —
  **688/688** `delete_fact` (discriminating: these can fail) and **758/758**
  `flip_target` (**still vacuous by construction**; see limitations).

Leakage/support audit: **passed**. Pairwise fingerprint overlap is zero across
train/tuning/calibration; the `meta_test` domain is empty because v3 carries no
sealed payload.

## What this gate does NOT establish

G1 is an **instrument freeze**. It establishes that the measuring instrument is
frozen, reproducible, hash-bound, leakage-audited and protected. It establishes
nothing about the science. Specifically:

1. **No capability claim.** Nothing here shows the frozen organ can express any
   v3 task. The sampled oracle was 2/3 on three tuning tasks under the *v2*
   tasks; v3 has not been screened at all. That is gate G2.
2. **No learnability claim.** No teaching was run on v3 tasks. The v2 dry run
   measured construction defects, not learner behaviour. That is gate G3.
3. **No calibration, margin, power or threshold.** v3 carries the v2
   `confidence_level` (0.95) and `target_power` (0.80) *only as declared
   view fields*; no margin, sample size or threshold is computed, chosen or
   implied. The calibration view says so in a dedicated field. v1 calibration
   numbers may not be reused. That is gate G4, and G5 requires explicit human
   sign-off on the exact candidate manifest hash.
4. **v1 and v3 scores are not comparable.** The task semantics changed by
   design. The freeze records this in `generator.json` under `comparability`; it
   is not a caveat about the freeze, it is a fact about the lineage.
5. **The freeze is DEV-only and cannot be used to reach the meta-test.** There is
   no sealed directory, no meta-test seed commitment, no `meta_test_authorized`
   flag, and the verifier rejects both a `sealed/` directory and any sealed file
   entry. The meta-test remains BLOCKED behind G5.
6. **Not a remote research campaign.** Everything here is local, on the working
   tree, at source commit `9031ce2`, with 85+ uncommitted files in the tree and
   no clean-source source archive (`source_archive_sha256` is empty). A remote run
   under `infrastructure/kaggle/RESEARCH_GUIDE.md` would need a clean commit-
   addressed bundle.

## Limitations (carried forward, unresolved)

Restated as required; none of these is fixed by G1:

- **R20 local reproduction: four nonzero state hashes still differ.** The
  untrained, trained, feedback-shuffled and state-swapped state hashes from the
  2026-09-11 local reproduction do not match the historical remote run. Exact
  numerical-state reproduction remains **unresolved**, and no replacement shard
  exists. G1 does not touch it and inherits it.
- **eval-v2 A2 and C repairs remain unsigned.** The sense/alias contract (A2)
  and tool-output scoring (C) are still unapplied. No capability claim on eval-v2
  stands until they land. This run changed nothing in `eval/`,
  `src/oczy/eval_v2/` or the tool curriculum.
- **758 flip-target mutation checks are still vacuous.** They are counted
  separately from the 688 discriminating delete-a-fact checks and are reported
  as non-discriminating. A flipped target cannot fail once its certificate
  verifies, so 758/758 is not evidence of fidelity; the real guarantee is the
  independent derivation disagree-check.
- **The 175 baseline certificates carry no derivation claim** and are exempt by
  design. They are reported separately rather than folded into 933/933.
- **DEV within-domain fingerprint duplicates are permitted** for finite-state
  assignments when the assignment space is smaller than the task count. That is
  the v1/v2 lineage's inherited limitation, not a new one, and it does not affect
  the cross-domain firewall (which is zero).
- **The independent checker validates construction, not comprehension.** A
  certificate proves the teaching text determines the scored target; it says
  nothing about whether a frozen 0.5B organ can use the teaching.

## Next gate

**G2 — oracle capability screen on the new DEV version**: can the frozen organ
express a v3 task under oracle control? Per the sign-off chain, a G2 failure is
an articulation block (a mouth–cortex protocol failure), **not** a cortex
refutation, and it stops the sequence.

G2 was **not** run in this session. It requires model calls, and the card
authorizes proceeding "only as far as G1 (freeze) and G2/G3 if they pass
cleanly". G1 passed; the decision to spend the compute on G2, and the scheduling
for it, are recorded here for the manager rather than taken unilaterally.