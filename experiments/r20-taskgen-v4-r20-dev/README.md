# R20 `meta_cortex/v4-r20` DEV instrument (frozen successor)

**Status: FROZEN 2026-10-05** at a G1-equivalent freeze.
`definition_sha256`
`fd2d8db9012683d4955879481c7f5a91eac16d3c82cde018846605c9378f1031`.

This is the **successor** to the frozen [`meta_cortex/v3`](../r20-taskgen-v3-dev/)
instrument. It exists because v3 failed gate G2 (oracle context 0/15, an
articulation block) while carrying neither of the two prompt amendments the user
approved on 2026-09-11.

Authorized by the user decision relayed by the manager 2026-10-05 (kanban card
`t_37e96ee1`), recorded in
[`../r20-task-support-repair-v1/SIGNOFF_CHAIN.md`](../r20-task-support-repair-v1/SIGNOFF_CHAIN.md)
§S4. The scope is **instrument construction only**.

## Why this id and not `meta_cortex/v4`

The user asked for "v4". Plain `meta_cortex/v4` is already taken by the approved
`experiments/r23.5-serialization-dev/instruments/v4` (manifest `d58adb749f4112f2…`),
and `meta_cortex/v3` is doubly claimed (r23.5 v3 amendment `5c944abc…` versus the
G1-frozen R20 lineage `ab99c173…`). The successor id is therefore the
scope-qualified **`meta_cortex/v4-r20`**. Both collisions are recorded in
`instrument/DEFINITION.json` under `naming`.

## What this instrument is

`meta_cortex/v4-r20` differs from `meta_cortex/v3` in exactly **three** ways,
each explicitly authorized:

1. **Amendment A** — the identical bare-answer system instruction
   (`Return only the requested answer token or string. Do not add an explanation,
   label, or surrounding quotation marks.`) is prepended to every public DEV
   probe: all six probe kinds, all three families, **933/933 probes**.
2. **Amendment B** — the 35 `rule_transformation` `oracle_context` headers are
   rewritten from the shorthand `Rule: <template> with parameters 'p1' and 'p2'`
   form to the approved complete English descriptions. Worked examples,
   operands, expected answers, splits, seeds and scoring are unchanged.
3. **`max_new_tokens` 32 → 128** — a **signed field changed relative to v3**,
   justified by the observed truncation (10 of 15 v3 oracle generations cut
   mid-preamble at 32) and bounded at 4x the longest observed preamble. Recorded
   under `max_new_tokens_change`.

The **task content is byte-identical** to the pinned v3 public DEV view: the
catalog digest `33210eb4…` and support-bundle digest `1195e783…` are unchanged,
and the materializer's per-probe diff proves `non_probe_task_fields_changed = 0`
and `probe_payloads_changed_outside_amendment_b = 0`.

The scorer and endpoint registries are inherited **byte-for-byte** from the
frozen v2/v3 instrument and the freeze fails closed on drift. The prompt registry
deliberately is **not** inherited — carrying the amendments is the point — and
the verifier rejects a re-signed definition that claims the v3 prompt registry.

**v1/v2/v3 and v4-r20 scores are not comparable.** The prompt registry, the
`max_new_tokens` field and the versioned DEV seed table all changed. This is
recorded in `instrument/public/generator.json` under `comparability`.

## What this instrument is NOT

- **Not a candidate.** No calibration, margin, power number or threshold was
  computed or chosen. The calibration view carries the inherited declared
  `confidence_level` (0.95) and `target_power` (0.80) as *view fields only*.
- **Not meta-test capable.** No `sealed/` directory, no meta-test seed
  commitment, `meta_test_authorized: false`. The verifier rejects both a
  `sealed/` directory and any sealed file entry. The meta-test remains blocked
  behind gate G5.
- **Not evidence of capability.** The separate G2 oracle screen is what speaks to
  capability; see the evidence below.

## Layout

```
g1_report.json                     the G1-equivalent freeze report
materialization/
  MANIFEST.json                    self-hashed amendment manifest (base hashes,
                                   amendment record, per-probe amendment diff)
  TASKS.json                       the amended task records
instrument/
  DEFINITION.json                  self-hashed; lists all 11 public files
  public/DEV_VIEW.json             train + tuning only
  public/CALIBRATION_VIEW.json     held-back calibration tasks
  public/audits/leakage_summary.json  leakage/support audit (passed)
  public/generator.json            v2-dev lineage, base pin, comparability note
  public/seeds.json                versioned DEV seed table
  public/prompts.json  public/scorers.json  public/endpoints.json
  public/probe_counts.json  public/chat_template.txt
  public/tasks/meta_train.jsonl               (90 tasks)
  public/tasks/meta_validation_tuning.jsonl   (15 tasks)
  public/tasks/meta_validation_calibration.jsonl (90 tasks)
```

## Reproduce and verify

Freeze (fail-closed on a wrong base; refuses to overwrite):

```bash
.venv/bin/python scripts/freeze_r20_v4_r20_instrument.py \
  --base-public-root experiments/r20-taskgen-v3-dev/instrument/public \
  --output <new-dir>/instrument --repro-output <other-dir>/repro \
  --report <new-dir>/g1_report.json
```

Verify an existing freeze (fails closed on any tamper):

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; \
  from oczy.experiments.meta_cortex.instrument_v4_r20 import verify_v4_r20_definition; \
  print(verify_v4_r20_definition(Path('experiments/r20-taskgen-v4-r20-dev/instrument')).definition_sha256)"
```

Materialize (or re-verify) only the amendment bundle:

```bash
.venv/bin/python scripts/materialize_r20_v4_r20.py \
  --public-root experiments/r20-taskgen-v3-dev/instrument/public \
  --output <new-dir>/materialization
```

Exit code **3** means the base is not the approved, pinned v3 public DEV view.

The freeze is deterministic: two independent materializations produce the same
`definition_sha256` (`diff -r` clean).

The organ identity bound here is
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`, reproduced
locally under the recorded historical runtime (Python 3.12.13, torch 2.10.0+cpu,
torchao 0.17.0, transformers 5.0.0, tokenizers 0.22.2).

## Guarding

The frozen tree, its amendment manifest, its gate report, the instrument module
(`instrument_v4_r20.py`), the materializer, the freeze CLI and the G2 successor
screen are all in `scripts/eval_guard.py`'s protected set. Changing any byte
requires `EVAL_CHANGE_APPROVED=1 --allow` — an explicit human decision, not an
automatic one.

## Evidence

- G1-equivalent freeze record:
  [`../../experiments_logs/2026-10-05_r20_g1_v4_r20_instrument_freeze.md`](../../experiments_logs/2026-10-05_r20_g1_v4_r20_instrument_freeze.md)
- G2 successor oracle screen (**PASSED 7/15**, partial):
  [`../../experiments_logs/2026-10-05_r20_g2_v4_r20_oracle_screen.md`](../../experiments_logs/2026-10-05_r20_g2_v4_r20_oracle_screen.md)
- The v3 G2 failure this successor answers:
  [`../../experiments_logs/2026-10-05_r20_g2_oracle_screen.md`](../../experiments_logs/2026-10-05_r20_g2_oracle_screen.md)
- The 2026-09-11 amendments carried in:
  [`../../experiments_logs/2026-09-11_campaign_r20_dev_output_path.md`](../../experiments_logs/2026-09-11_campaign_r20_dev_output_path.md)
- Base instrument: [`../r20-taskgen-v3-dev/README.md`](../r20-taskgen-v3-dev/README.md)

## Limitations

Carried forward unresolved: the R20 local-reproduction gap (four nonzero state
hashes still differ, no replacement shard); the unsigned eval-v2 A2 and C
repairs; the 758 flip-target mutation checks, vacuous by construction and
reported separately from the 688 discriminating delete-a-fact checks. Specific to
this successor: **the transformation oracle family is 0/5 under oracle control**
even though the gate passed on its registered criterion, and the G2 run was a
local run on an uncommitted working tree, not a clean-source remote campaign.
See both records' limitations sections.
