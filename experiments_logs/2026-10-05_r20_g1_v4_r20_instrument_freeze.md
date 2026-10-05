# R20 successor `meta_cortex/v4-r20` — G1-equivalent freeze (2026-10-05)

**Gate:** G1-equivalent (successor freeze) — **PASSED**.
**Classification:** DEV_INSTRUMENT_FREEZE.
**Authorization:** user decision relayed by the manager 2026-10-05 (kanban card
`t_37e96ee1`, authoritative — supersedes the open questions on `t_5670e505`):
Option A funded, successor named `meta_cortex/v4-r20`, the `max_new_tokens` move
approved, and the evidence push to the remote approved. The scope is
**instrument construction only**: no scientific verdict, no threshold selection
from results, and no meta-test authorization is implied.

No model was run, no optimizer step was taken, no threshold/margin/power value
was chosen, no v1/v2/v3 score was touched, and no sealed or meta-test payload was
generated or opened.

## Why a successor instrument exists

`meta_cortex/v3` passed G1 and then **failed G2 0/15** (oracle context), an
articulation block. The recorded diagnosis
([`2026-10-05_r20_g2_oracle_screen.md`](2026-10-05_r20_g2_oracle_screen.md)) was
that v3 inherited v2's frozen prompt registry byte-for-byte and therefore
carries **neither** of the two prompt amendments the user approved on
2026-09-11 for the v2 instrument:

- **Amendment A** — an identical bare-answer system instruction on every public
  DEV probe.
- **Amendment B** — complete oracle rule descriptions for the transformation
  oracle header, replacing the shorthand `Rule: <template> with parameters 'p1'
  and 'p2'` form.

The successor carries both amendments **and** raises `max_new_tokens` from the
signed v3 value of 32 to 128, which the user approved for the same reason (the
G2 v3 artifact showed oracle generations truncated mid-preamble at 32).

## Naming: why `meta_cortex/v4-r20` and not `meta_cortex/v4`

The user said "name it v4". Plain `meta_cortex/v4` is already taken by
`experiments/r23.5-serialization-dev/instruments/v4` (user-approved amendment B,
manifest `d58adb749f4112f2920f08e0dbe0ce5c6a1ee7711f562faee4e41072eccc8764`), and
`meta_cortex/v3` is doubly claimed (the r23.5 v3 amendment `5c944abc…` versus the
G1-frozen R20 lineage `ab99c173…`). The successor id is therefore the
scope-qualified `meta_cortex/v4-r20`, which honors the "v4" intent
unambiguously. This is recorded in `DEFINITION.json` under `naming`, including
both collisions and the scope directory `experiments/r20-taskgen-v4-r20-dev/`.

## What G1-equivalent froze

| Role | Frozen value |
|---|---|
| `definition_sha256` | `fd2d8db9012683d4955879481c7f5a91eac16d3c82cde018846605c9378f1031` |
| `dev_view_sha256` | `9363cff3b07994f8c8f21d04809c555f54855d6b9fab20ac0143d1e63a6d4f23` |
| `calibration_view_sha256` | `4d1f27d69f62366c5064092ab1c67c29dfd038d6e4cebb1141154588a8eb2111` |
| catalog digest | `33210eb474a25925815c1112e2d6c6ea889b349c8d137fbcbe0aacac5f035219` (**unchanged from v3**) |
| support-bundle digest | `1195e78362671aecc692d08030c34c6692bd9e00fe73c76023027fa1961a3a62` (**unchanged from v3**) |
| amended prompt registry | `d8bd0b263216f83df83d258850d014b338627e09a355111a87b538b9ab401a54` |
| base (v3) prompt registry | `db624922bf4f67e3e1011b5530ade4111b479ef05391ca0a61e375cac2339735` |
| DEV seed table | `049ce5ff52f1223ad6facf15c200524e8c4569b77de386e03fe572cdcb0dfa3f` (v3: `c3672ed7…`; **the table is versioned by instrument id/version, so every seed differs by construction**) |
| probe counts | `4c643a8808481355f5463e1ec7efd1f3fa53e845a679bb4c3ece96c1ae0348e5` (**unchanged from v3**) |
| `max_new_tokens` | **128** (v3 signed value: 32) |
| source commit | `9031ce2ca95022f95cdad515e2071855eef1d8ac` |

Materialized task counts: **90** `meta_train`, **15**
`meta_validation_tuning`, **90** `meta_validation_calibration` — identical to v3.

Base pinned by bytes, not by name: `meta_cortex/v3` `definition_sha256`
`ab99c173…`, `dev_view_sha256` `593090c4…`, `catalog_sha256` `33210eb4…`, and the
three public task files at `1f1a4118…` / `f4037013…` / `13e17d93…`.

Inherited registry hashes, asserted unchanged rather than assumed (the prompt
registry is deliberately **not** in this set — carrying the amendments is the
point of the successor):

| Registry | SHA-256 |
|---|---|
| scorer | `e5d746d0477c489157d1699e2ae73dfcc8ac92998719de1a06d92fcff4b1c742` |
| endpoint | `669d6130075e429264a6c9eec470bbf82663559433781adcffa6ed942c9ca796` |

Organ identity bound to the instrument is the historical
`a342431c0fdb02bf1bbed95255795ad52df3e799c821318c6206021a46a3f9ea`
(`Qwen/Qwen2.5-0.5B-Instruct`, INT8 weight-only) — reproduced locally on this
machine on 2026-10-05 by the G2 successor screen, not assumed.

## The definition diff vs v3 (exactly three changes)

1. **Prompt registry — Amendment A.** Every public DEV probe (all six probe
   kinds, all three families) receives the system message
   `Return only the requested answer token or string. Do not add an explanation,
   label, or surrounding quotation marks.` as its first message. The system
   message stays first when a teaching transcript is prepended.
2. **Prompt registry — Amendment B.** Every `rule_transformation`
   `oracle_context` header is rewritten from the shorthand form to the approved
   complete English description; the worked examples, operands, expected answers,
   splits, seeds and scoring are unchanged.
3. **`max_new_tokens` 32 → 128.** This **changes a signed field relative to
   v3**. It is recorded in `DEFINITION.json` under `max_new_tokens_change` with
   the authorization, the justification (the G2 v3 artifact showed 10 of 15
   oracle generations truncated mid-preamble at 32; 128 is 4x the longest
   observed preamble and remains bounded), and the comparability rule.
   **v3 and v4-r20 scores are NOT comparable as a causal improvement** — the
   same rule applied to the v2→v3 transition.

Nothing else changed. The task *content* is byte-identical to the pinned v3
public DEV view; only probe messages changed. One further value differs by
construction rather than by decision: the DEV **seed table** is derived from
`<seed-schema>|<instrument-id>|<instrument-version>|…`, so versioning the
instrument id changes every derived seed (`049ce5ff…` versus v3's `c3672ed7…`).
That is the seed derivation's own versioning rule, not an extra amendment: no
seed *schedule* changed, only the values it hashes to. It is why the successor's
`dev_view_sha256` differs from v3's even beyond the prompt registry, and it is
one more reason v3 and v4-r20 runs are not comparable.

## Commands, exit codes and artifacts

### 1. Freeze + independent verification + checker + amendment surface

```bash
.venv/bin/python scripts/freeze_r20_v4_r20_instrument.py \
  --base-public-root experiments/r20-taskgen-v3-dev/instrument/public \
  --output experiments/r20-taskgen-v4-r20-dev/instrument \
  --repro-output <scratch>/repro_repo \
  --report experiments/r20-taskgen-v4-r20-dev/g1_report.json
```

Exit code **0**, 0.6 s. Report `passed: true`
(`schema: oczy/r20-g1-v4-r20-freeze/v1`).

Artifact: `experiments/r20-taskgen-v4-r20-dev/instrument/` (11 hash-listed
public files) plus `experiments/r20-taskgen-v4-r20-dev/materialization/`
(`MANIFEST.json`, `TASKS.json`) and `g1_report.json`.

### 2. The new fail-closed materializer (G1-lineage analogue)

```bash
.venv/bin/python scripts/materialize_r20_v4_r20.py \
  --public-root experiments/r20-taskgen-v3-dev/instrument/public \
  --output experiments/r20-taskgen-v4-r20-dev/materialization
```

Exit code **0**: `meta_cortex/v4-r20: fb9c8e6658e997e37f33aa25e8e7f5477b9f7db2a7a55153083b9f2444c15699`.

Fail-closed checks it performs before writing anything:

- the base must be the approved v3 public DEV view (instrument id/version,
  taskgen lineage, prompt registry, and the signed `max_new_tokens`=32 all
  pinned; `dev_view_sha256` and `definition_sha256` pinned);
- the base `DEFINITION.json` must self-verify to the pinned hash, list the
  pinned task/registry bytes, and contain no `sealed/` directory;
- a fresh in-memory rebuild of the `oczy/meta-cortex/taskgen/v2-dev` lineage must
  reproduce the base task bytes exactly (the base is a *record* of the generator,
  not a drifted snapshot);
- the loaded base view must equal that rebuild record-for-record;
- the amended tasks must differ from the base in exactly the approved ways, with
  per-probe assertions that the system message is Amendment A, that everything
  after it is unchanged, and that only `rule_transformation`
  `oracle_context` headers change.

**Wrong-base refusals, exit 3** (the v2-lineage materializer refused the v3
lineage with exit 3 on the prior card; this one refuses a wrong base as loudly):

| Base given | Result |
|---|---|
| `…/r20-dev-calibration-v1/instrument/public` (a v1 instrument) | `REFUSED (wrong base): Base must be the approved 'meta_cortex/v3' public DEV instrument, got instrument_id 'meta_cortex/v1'` — exit **3** |
| a byte-tampered copy of the v3 tree (`max_new_tokens` 32→64) | `REFUSED (wrong base): Base max_new_tokens must be the signed v3 value 32, got 64` — exit **3** |
| a copy with `instrument_id` changed to `meta_cortex/v2` | `REFUSED (wrong base): Base must be the approved 'meta_cortex/v3' …` — exit **3** |
| the unmodified v2-lineage materializer on the v3 base | still refuses: `Not the approved v2 public DEV instrument` — nonzero (regression-checked in the test suite) |

Amendment diff recorded in the manifest (self-verified, then independently
re-measured from the frozen tree):

| Quantity | Value |
|---|---|
| system messages added (Amendment A) | **933 / 933 probes** |
| oracle headers rewritten (Amendment B) | **35** |
| non-probe task fields changed | **0** |
| probe payloads changed outside Amendment B | **0** |
| oracle targets independently re-derived from the amended description text | **35 / 35** |

### 3. Independent checker over the pinned base tasks

```bash
.venv/bin/python scripts/r20_task_support_check.py dry-run \
  --output <report.json> --public-root experiments/r20-taskgen-v3-dev/instrument/public
```

Exit code **0** (invoked from the freeze CLI; script SHA-256
`4e501038bbab53951ef74aa2e744ec7c84c03b6e6fd1074369d36c19de7e06cb`).
Tasks parsed **105**; parse defects **0**; all **22** defect classes **0**;
support certificates **933** total, **758** verified, **175**
`baseline_not_required`, **0** failed; mutation checks **1446/1446** detected
(**688/688** discriminating `delete_fact`, **758/758** `flip_target`, still
vacuous by construction). These are exactly the recorded G1 numbers, which is
the point: the successor's task content is unchanged, so the v3 construction
verdict carries over and was re-measured rather than assumed.

Leakage/support audit: **passed**. Pairwise fingerprint overlap is zero across
train/tuning/calibration; the `meta_test` domain is empty.

### 4. Determinism

The freeze was materialized twice into independent directories; both produced
`definition_sha256` `fd2d8db9…`, and `diff -r` reported zero differences (exit
**0**). The freeze is a deterministic function of the lineage, not of the
directory or the run.

### 5. Guard tests: a source byte change rejects loading

```bash
.venv/bin/python -m pytest src/oczy/experiments/tests/test_meta_cortex_instrument_v4_r20.py -q
```

Exit code **0**: **26 passed**. Coverage includes a single-byte flip in each of
the three task files and five registries (all rejected with `File hash
mismatch`), a mutated `DEFINITION.json` count (rejected on self-hash), a
re-signed definition claiming a different scorer registry (rejected on
`registry drift`), a re-signed definition carrying the **v3** prompt registry
(rejected — proving the amendments must actually be present), a re-signed
definition with `max_new_tokens` 64 (rejected), the four wrong-base classes
above, an unlisted extra file, a `sealed/` directory, overwrite refusal, and
per-probe checks that Amendment A is on all 933 probes and Amendment B touches
only the 35 transformation oracle headers.

```bash
.venv/bin/python -m pytest scripts/tests/test_r20_v4_r20.py -q
```

Exit code **0**: **14 passed** — the materializer's own fail-closed behaviour,
including that the **v2-lineage materializer still refuses the v3 base**.

### 6. Regression surface

```bash
.venv/bin/python -m pytest \
  src/oczy/experiments/tests/test_meta_cortex_taskgen.py \
  src/oczy/experiments/tests/test_meta_cortex_taskgen_v2_support.py \
  src/oczy/experiments/tests/test_meta_cortex_instrument.py \
  src/oczy/experiments/tests/test_meta_cortex_instrument_v3.py \
  src/oczy/experiments/tests/test_meta_cortex_generation_text.py \
  scripts/tests/test_eval_guard.py scripts/tests/test_dev_prompt_repair.py \
  scripts/tests/test_r20_g2_oracle_screen.py -q
```

Exit code **0**: **272 passed** — the v1 generator tests, the v2 support tests,
the v2 and **v3** instrument tests (so the frozen v3 instrument still verifies),
the generation-text tests, the guard tests, the 2026-09-11 amendment tests, and
the **v3 G2** screen tests (so the historical G2 failure still reproduces as
recorded).

### 7. Eval guard protection

The frozen successor tree, its amendment manifest, its gate report, the new
instrument module, the materializer, the freeze CLI and the G2 successor screen
were added to `scripts/eval_guard.py`'s protected set.

```bash
.venv/bin/python scripts/eval_guard.py --allow     # exit 1: EVAL_CHANGE_APPROVED unset
```

Exit code **1**, refusing, as designed.
`scripts/tests/test_eval_guard.py` → exit code **0**, **40 passed**, including
five new cases that tamper with the frozen successor tree.

### 8. Lint

```bash
.venv/bin/python -m ruff check \
  src/oczy/experiments/meta_cortex/instrument_v4_r20.py \
  scripts/materialize_r20_v4_r20.py scripts/freeze_r20_v4_r20_instrument.py \
  scripts/probe_r20_v4_r20_oracle.py \
  src/oczy/experiments/tests/test_meta_cortex_instrument_v4_r20.py \
  scripts/tests/test_r20_v4_r20.py scripts/eval_guard.py scripts/tests/test_eval_guard.py
```

Exit code **0**: `All checks passed!`

## What this gate does NOT establish

G1-equivalent is an **instrument freeze**. It establishes that the successor
measuring instrument is frozen, reproducible, hash-bound, leakage-audited and
protected, and that the approved amendments are actually carried. It establishes
nothing about the science:

1. **No capability claim.** Nothing here shows the frozen organ can express a
   v4-r20 task. That is gate G2 (run separately, see
   [`2026-10-05_r20_g2_v4_r20_oracle_screen.md`](2026-10-05_r20_g2_v4_r20_oracle_screen.md)).
2. **No learnability claim.** No teaching was run. That is gate G3.
3. **No calibration, margin, power or threshold.** The successor carries v2/v3's
   declared `confidence_level` (0.95) and `target_power` (0.80) *only as declared
   view fields*; no margin, sample size or threshold is computed, chosen or
   implied. That is gate G4, and G5 requires explicit human sign-off on the exact
   candidate manifest hash.
4. **v3 and v4-r20 scores are not comparable.** The prompt registry and
   `max_new_tokens` changed by design; v1/v2/v3 scores are not comparable to
   v4-r20 for the same reason.
5. **DEV-only; cannot reach the meta-test.** No sealed directory, no meta-test
   seed commitment, no `meta_test_authorized` flag; the verifier rejects both a
   `sealed/` directory and any sealed file entry. The meta-test remains BLOCKED
   behind G5.
6. **Not a remote research campaign.** Everything is local, on the working tree
   at source commit `9031ce2`, with a large uncommitted set and no clean-source
   source archive (`source_archive_sha256` is empty).
7. **Not a task-support verdict.** The checker verdict is inherited from the v3
   base by construction; the successor's task content is byte-identical, which
   the materializer and the amendment-surface check both prove. Nothing here
   re-adjudicates the task-support repair.

## Limitations (carried forward, unresolved)

Restated as required; none of these is fixed by this freeze:

- **R20 local reproduction: four nonzero state hashes still differ.** The
  untrained, trained, feedback-shuffled and state-swapped state hashes from the
  2026-09-11 local reproduction do not match the historical remote run. Exact
  numerical-state reproduction remains **unresolved**, and no replacement shard
  exists. This freeze does not touch it and inherits it.
- **eval-v2 A2 and C repairs remain unsigned.** The sense/alias contract (A2)
  and tool-output scoring (C) are still unapplied. No capability claim on eval-v2
  stands until they land. This run changed nothing in `eval/`,
  `src/oczy/eval_v2/` or the tool curriculum.
- **758 flip-target mutation checks are still vacuous.** They are counted
  separately from the 688 discriminating delete-a-fact checks and are reported as
  non-discriminating.
- **The 175 baseline certificates carry no derivation claim** and are exempt by
  design.
- **DEV within-domain fingerprint duplicates are permitted** for finite-state
  assignments when the assignment space is smaller than the task count; that is
  the inherited v1/v2 lineage limitation and it does not affect the cross-domain
  firewall (zero).
- **The independent checker validates construction, not comprehension.** A
  certificate proves the teaching text determines the scored target; it says
  nothing about whether a frozen 0.5B organ can use the teaching.
- **`max_new_tokens`=128 is an approved signed-field move, not a repair of the
  articulation block.** It removes a *truncation* confound; it does not by itself
  make the mouth able to express a task. Whether it helps is exactly what G2
  measures.
- **The amendment surface is measured on rendered text, not on model behaviour.**
  That the descriptions are the approved ones and yield the scored answers is
  proven; that the frozen organ *uses* them is not.

## Next gate

**G2 — oracle capability screen on `meta_cortex/v4-r20`.** Per the sign-off
chain, a G2 failure is an articulation block (a mouth–cortex protocol failure),
**not** a cortex refutation, and it stops the sequence. G2 was run in the same
session; see
[`2026-10-05_r20_g2_v4_r20_oracle_screen.md`](2026-10-05_r20_g2_v4_r20_oracle_screen.md).
