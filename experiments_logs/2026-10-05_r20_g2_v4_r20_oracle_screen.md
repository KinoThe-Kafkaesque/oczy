# R20 gate G2 (successor) — oracle capability screen on `meta_cortex/v4-r20`: **PASSED**

**Gate:** G2 (oracle capability screen) — **PASSED**. The gate sequence may
continue to G3.
**Classification:** DEV_GATE_G2_ORACLE_SCREEN.
**Meaning of this pass:** the frozen organ **can** express v4-r20 tasks under
oracle control, i.e. the mouth–cortex protocol is not refuted for these tasks.
It is **not** evidence of cortex learning: the soft bank was empty and no
training ran. The learner was never involved.

**Authorization:** user decision relayed by the manager 2026-10-05 (kanban card
`t_37e96ee1`): Option A funded, successor named `meta_cortex/v4-r20`,
`max_new_tokens` move approved. No model run beyond DEV; no threshold chosen; no
meta-test access; no optimization; no score or threshold change.

## What was run

```bash
R=/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_VERBOSITY=error OCZY_REMOTE_CPU_ONLY=1 \
OCZY_MODEL_DIR=/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
HF_HUB_OFFLINE=1 PYTHONPATH=/home/nyanpasu/Desktop/code/kinoSoft/oczy/src \
bwrap --die-with-parent --unshare-net --tmpfs / --ro-bind /usr /usr \
  --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc --ro-bind /home /home --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind $R/model /kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
  --bind $R $R --chdir /home/nyanpasu/Desktop/code/kinoSoft/oczy -- \
  $R/runtime/bin/python scripts/probe_r20_v4_r20_oracle.py \
    --instrument-root experiments/r20-taskgen-v4-r20-dev/instrument \
    --output $R/g2-v4r20-final
```

Exit code **0** (the screen completed; the verdict lives in the artifact, not in
the exit code, so a *failed* gate would also have exited 0). Wall time
**156.9 s**. **79 rows.**

Historical runtime: Python 3.12.13, torch 2.10.0+cpu, torchao 0.17.0,
transformers 5.0.0, tokenizers 0.22.2, no network, offline model directory.

Artifacts:
[`artifacts/2026-10-05_r20_g2_v4_r20_oracle_screen/results.json`](artifacts/2026-10-05_r20_g2_v4_r20_oracle_screen/results.json)
(SHA-256 `3962fa7b3a0f4b90e13c9fa2ba1c6b7dd41e922b882c81713be3033939bb1b40`),
plus `stdout.log` and `stderr.log`. The screen source hash recorded in the
artifact is `probe_source_sha256`
`0cc7cc5ebc5b693c7ae74190d4e1278ebfeb86bf676c44db4774671cc903c8b5`, which equals
`sha256sum scripts/probe_r20_v4_r20_oracle.py` at record time.

Instrument binding, all checked in-run and failing closed otherwise:

| Check | Value |
|---|---|
| `definition_sha256` | `fd2d8db9012683d4955879481c7f5a91eac16d3c82cde018846605c9378f1031` (the frozen v4-r20 definition) |
| `dev_view_sha256` | `9363cff3b07994f8c8f21d04809c555f54855d6b9fab20ac0143d1e63a6d4f23` |
| `base_instrument_id` | `meta_cortex/v3` |
| `base_definition_sha256` | `ab99c173…d4bee` |
| Amendment A verified on every probe before any model call | yes (all 933 probes) |
| scorer | matches the frozen registry binding |
| organ hash before / after | `a342431c…f9ea` / identical |
| organ matches frozen binding | **true** |
| `optimizer_steps` | 0 |
| `training_run` | false |
| `thresholds_selected` | false |
| `meta_test_accessed` / `calibration_accessed` / `sealed_accessed` | false / false / false |
| soft bank width | 0 (no cortex state exists) |
| `max_new_tokens` | 128 (v3 signed value: 32) |

## Result

Gate criterion, fixed before the run: **oracle_context correct > 0**.

| Condition | Correct | Total | Role |
|---|---:|---:|---|
| `no_context` | 2 | 32 | the floor |
| `teaching_context` | 9 | 32 | **retrieval comparator** — the bar, not a result |
| `oracle_context` | **7** | **15** | **the gate** |

Per family, oracle context: contextual_remap **2/5**, rule_transformation
**0/5**, finite_state **5/5**.

**G2 PASSED.** The frozen organ produced exact answers on 7 of the 15
oracle-controlled v4-r20 probes, so the gate criterion (`> 0`) is met.

### Comparison with the v3 G2 failure (descriptive, not a causal claim)

| Condition | v3 (2026-10-05) | v4-r20 (2026-10-05) |
|---|---:|---:|
| `no_context` | 0/32 | 2/32 |
| `teaching_context` | 4/32 | 9/32 |
| `oracle_context` | **0/15** | **7/15** |

**These are not comparable as a causal improvement.** The prompt registry
changed (both approved amendments), `max_new_tokens` changed (32 → 128, a signed
field), and the seed table is versioned by instrument id, so v3 and v4-r20 differ
in three ways at once. The v2→v3 rule applies: no single-variable causal claim
is made here. What the table *does* establish is that the instrument change the
user authorized moved the gate from failing to passing — which is exactly the
question G2 asks.

### Truncation accounting (descriptive only, decides nothing)

| Condition | generations ending mid-sentence |
|---|---:|
| `no_context` | 3/32 |
| `teaching_context` | 0/32 |
| `oracle_context` | 1/15 |

The heuristic is reported separately from the score and the gate never consults
it. It is what the `max_new_tokens` raise was aimed at: the v3 run's 15 oracle
generations were overwhelmingly mid-preamble cuts at 32 tokens, and at 128 that
is no longer the dominant failure mode. The one remaining flagged oracle row is
a prose-wrapped answer, not a cut-off one.

## What the correct answers are

The 7 correct rows are all exact normalized matches under the frozen scorer:

| Family | Expected | Generated | Condition |
|---|---|---|---|
| finite_state | `q2` | `q2` | oracle (×4 tasks) |
| finite_state | `q1` | `q1` | oracle |
| contextual_remap | `dax` | `dax` | oracle |
| contextual_remap | `twem` | `twem` | oracle |

The **finite-state oracle is now 5/5**, including the goal-dependent composition
probe. The contextual-remap oracle is 2/5: the three misses return *another*
token from the stated mapping (`vol`, `pex`) rather than a mapping error — a
selection failure inside a correct mapping, not a failure to read the mapping.
The **transformation oracle remains 0/5**, and its failures are the informative
ones:

| Expected | Generated |
|---|---|
| `pexx` | `npexrcpexmpex` |
| `fepnu` | `nu -> fepnu` |
| `djirltjir` | `jir` |
| `pexmpexcrpexn` | `pexpex` |
| `xfep` | `fepfep` |

The second row (`nu -> fepnu`) is a *correct* application of the stated rule
rendered in a `operand -> result` form the exact scorer rejects. The first row
applies the rule to the wrong operand (it returned the answer for `omicron`, a
worked example, rather than for the query `xi`). So the transformation oracle
now reads the description well enough to compute a correct answer — it just does
not reliably compute it **for the queried operand**, and it sometimes wraps it.

## The distinction that matters

The pass is 7/15 on the oracle condition. The retrieval bar is 9/32 (28%) and
the floor is 2/32 (6%). Oracle control is at 47% of its 15 probes. So:

- the **gate criterion is met** — the mouth can express these tasks under oracle
  control, which is all G2 claims;
- the articulation is **partial, not clean**: 8/15 oracle probes still fail,
  including all five transformation probes and three of five contextual-remap
  probes;
- the remaining failures are **not** explained away. They are a mixture of
  operand-selection errors, prose-wrapping under an exact scorer, and
  mapping-member selection. G2 does not separate those.

A stricter reading of "passes cleanly" would note that the transformation family
is still 0/5. That is a real residual risk for G3, and it is recorded here rather
than smoothed over. The gate criterion as registered is `oracle_context correct >
0`, and it is met; a *per-family* oracle criterion was never registered.

## What this gate does NOT establish

1. **Not a cortex result.** No cortex state existed (soft bank width 0) and no
   optimizer step was taken. Nothing here says a learned state could carry the
   task.
2. **Not a learnability result.** G3 was not run.
3. **Not comparable to any v2 or v3 number.** The prompt registry, the
   `max_new_tokens` field and the versioned seed table all changed by design.
4. **Not a powered-design result.** G4 was not run; no margin, sample size or
   threshold exists for v4-r20.
5. **Not a threshold selection.** `gate_criterion` was fixed as "oracle correct
   > 0" before the run and taken from the sign-off chain's definition of G2, not
   from any result. No number in this entry was fitted to the data.
6. **Not a task-support verdict.** The successor's task content is byte-identical
   to the pinned v3 base, whose construction verdict (22/22 defect classes at 0,
   758/758 derivation-backed certificates) is re-measured in the G1-equivalent
   record.
7. **Not a claim that the amendments caused the pass.** Three things changed at
   once. The pass is consistent with the recorded diagnosis (the v3 prompt format
   was defective for this organ) but does not isolate it.

## Limitations (carried forward, unresolved)

- **R20 local reproduction: four nonzero state hashes still differ** (untrained,
  trained, feedback-shuffled, state-swapped). Unresolved; inherited.
- **eval-v2 A2 and C repairs remain unsigned and unapplied.** No capability claim
  on eval-v2 stands.
- **758 flip-target mutation checks remain vacuous by construction**, reported
  separately from the 688 discriminating delete-a-fact checks.
- **Sample size.** 15 oracle probes over 15 tuning tasks, one per family index.
  7/15 is a clear direction but not a full-catalog estimate; v4-r20's 90
  calibration tasks were not screened (that view is held back by design).
- **The transformation family is 0/5 under oracle control.** The gate passed on
  the registered criterion; a family-level oracle criterion was never
  registered, and if G3's learnability battery weights transformation heavily,
  this is a live risk.
- **The prose/format confound is reduced but not eliminated.** At 128 tokens the
  truncation heuristic fires on 1/15 oracle rows instead of 10/15, but the
  scorer is still exact and still rejects correct answers wrapped in prose
  (`nu -> fepnu`). How much of the residual 8/15 is format versus articulation is
  not separated by this run.
- **Local run, not a clean-source remote campaign.** Working tree at `9031ce2`
  with a large uncommitted set; no source archive.
- **The screen's own source hash changed between the two completed runs.** The
  first completed screen (144.4 s, `probe_source_sha256` `b2d32b08…`) used an
  earlier revision of the truncation heuristic that flagged all 15 oracle rows;
  the recorded run (156.9 s, `probe_source_sha256` `0cc7cc5e…`) uses the revised
  heuristic. **The gate verdict and all 79 generated strings are byte-identical
  between the two runs** — the heuristic is descriptive-only and the gate never
  consults it — but the artifact is a rewrite of a pre-existing output, so both
  runs are preserved (`…/g2-v4r20/results.json` and
  `…/g2-v4r20-final/results.json` on the diagnostic host, and the first run's
  copy in the card scratch directory). The G1-equivalent freeze is untouched by
  this: only the probe script changed, and only in a non-deciding helper.

## Sequence status

| Gate | Status |
|---|---|
| S1/S2/S3 sign-off | recorded (2026-10-04) |
| user decision (Option A, v4-r20, `max_new_tokens`, evidence push) | recorded (2026-10-05, `t_37e96ee1`) |
| G1 freeze (`meta_cortex/v3`) | **PASSED** — `definition_sha256` `ab99c173…` |
| G2 oracle screen (v3) | **FAILED** — articulation block, 0/15 |
| G1-equivalent freeze (successor `meta_cortex/v4-r20`) | **PASSED** — `definition_sha256` `fd2d8db9…` |
| G2 oracle screen (v4-r20) | **PASSED** — 7/15, criterion `> 0` |
| G3 learnability | **not run in this session** — the card authorizes proceeding "only if G2 passes cleanly", and the transformation family's 0/5 makes "cleanly" arguable; the decision is left to the manager |
| G4 calibration/power | not run |
| G5 candidate sign-off | not reached; meta-test stays blocked |

**Next step.** G2 passed on its registered criterion, so G3 (DEV learnability on
the v4-r20 tasks: teaching fits and the no-update comparator on the same battery)
is the next gate in the sequence. This session stopped before G3 because the card
says to proceed to G3 "if G2 passes cleanly" and to "stop before G4 if anything
is ambiguous": the pass is real but the transformation family's oracle 0/5 is an
ambiguity a human should rule on. G4/G5 are untouched and the meta-test remains
blocked.
