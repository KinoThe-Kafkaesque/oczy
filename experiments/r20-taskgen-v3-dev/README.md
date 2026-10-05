# R20 `meta_cortex/v3` DEV instrument (frozen)

**Status: FROZEN 2026-10-05 at gate G1.** `definition_sha256`
`ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee`.

This is the adopted task-support-repaired DEV instrument
(`oczy/meta-cortex/taskgen/v2-dev` lineage), frozen under the human sign-off
recorded in
[`../r20-task-support-repair-v1/SIGNOFF_CHAIN.md`](../r20-task-support-repair-v1/SIGNOFF_CHAIN.md)
§S. The sign-off authorizes **instrument construction and adoption only**.

## What this instrument is

`meta_cortex/v3` differs from the previous `meta_cortex/v2` instrument in exactly
two ways:

1. **Task semantics.** Public DEV tasks come from the task-support repair
   (`taskgen_v2.build_dev_catalog_v2`), which removes untaught lookups, undefined
   composition, untaught FSM actions and goals, underdetermined transformation
   targets and contradictory specificity targets. Every scored probe carries an
   offline support certificate naming the exact teaching facts that determine its
   answer.
2. **Decoder.** Generated ids are decoded with `skip_special_tokens=True`
   (sign-off S3, adopted with a version bump after its comparability condition
   was discharged on 2026-10-05).

The prompt registry, scorer registry, endpoint registry, seed derivation, cortex
geometry and organ identity are inherited **byte-for-byte** from v2. The freeze
asserts this rather than assuming it: `materialize_v3_definition` fails closed if
any registry hash differs from its frozen v2 value, and `verify_v3_definition`
re-checks it on every load.

**v1 and v3 scores are not comparable.** The task semantics changed by design;
this is recorded in `instrument/public/generator.json` under `comparability`.

## What this instrument is NOT

- **Not a candidate.** No calibration, margin, power number or threshold was
  computed or chosen. The calibration view carries v2's declared `confidence_level`
  and `target_power` as *view fields only* and says so explicitly.
- **Not meta-test capable.** There is no `sealed/` directory, no meta-test seed
  commitment, and `meta_test_authorized: false`. The verifier rejects both a
  `sealed/` directory and any sealed file entry. The meta-test remains blocked
  behind gate G5 (explicit human sign-off on the exact candidate manifest hash).
- **Not evidence of capability.** Freezing an instrument says nothing about
  whether the frozen organ can express a task.

## Layout

```
instrument/
  DEFINITION.json                       self-hashed; lists all 11 public files
  public/DEV_VIEW.json                  train + tuning only
  public/CALIBRATION_VIEW.json          held-back calibration tasks
  public/audits/leakage_summary.json    leakage/support audit (passed)
  public/generator.json                 v2 lineage, root seed, comparability note
  public/seeds.json                     versioned DEV seed table
  public/prompts.json  public/scorers.json  public/endpoints.json
  public/probe_counts.json  public/chat_template.txt
  public/tasks/meta_train.jsonl               (90 tasks)
  public/tasks/meta_validation_tuning.jsonl   (15 tasks)
  public/tasks/meta_validation_calibration.jsonl (90 tasks)
```

## Reproduce and verify

Freeze (refuses to overwrite an existing directory):

```bash
.venv/bin/python scripts/freeze_r20_v3_instrument.py \
  --output <new-dir> --report <report.json>
```

Verify an existing freeze (fails closed on any tamper):

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; \
  from oczy.experiments.meta_cortex.instrument_v3 import verify_v3_definition; \
  print(verify_v3_definition(Path('experiments/r20-taskgen-v3-dev/instrument')).definition_sha256)"
```

The freeze is deterministic: two independent materializations produce the same
`definition_sha256`.

The organ identity bound here is
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`, reproduced
locally under the recorded historical runtime (Python 3.12.13, torch 2.10.0+cpu,
torchao 0.17.0, transformers 5.0.0, tokenizers 0.22.2).

## Guarding

The frozen tree, its generator lineage (`taskgen_v2.py`), the instrument module
(`instrument_v3.py`), the organ decode path, the identity probe, the independent
checker and the freeze CLI are all in `scripts/eval_guard.py`'s protected set.
Changing any byte requires `EVAL_CHANGE_APPROVED=1 --allow` — an explicit human
decision, not an automatic one.

## Evidence

- Gate G1 record:
  [`../../experiments_logs/2026-10-05_r20_g1_instrument_freeze.md`](../../experiments_logs/2026-10-05_r20_g1_instrument_freeze.md)
- S3 decode comparability record:
  [`../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md`](../../experiments_logs/2026-10-05_r20_signoff_s3_comparability.md)
- Repair design and pre-adoption evidence:
  [`../r20-task-support-repair-v1/README.md`](../r20-task-support-repair-v1/README.md)

## Limitations

Carried forward unresolved: the R20 local-reproduction gap (four nonzero state
hashes still differ, no replacement shard); the unsigned eval-v2 A2 and C
repairs; and the 758 flip-target mutation checks, which remain vacuous by
construction and are reported separately from the 688 discriminating
delete-a-fact checks. See the G1 record's limitations section.