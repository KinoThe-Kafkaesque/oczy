# R20 gate G2 — oracle capability screen on `meta_cortex/v3`: **FAILED**

**Gate:** G2 (oracle capability screen) — **FAILED**. This stops the gate
sequence.
**Classification:** DEV_GATE_G2_ORACLE_SCREEN.
**Meaning of this failure:** an **articulation block** (a mouth–cortex protocol
failure). Per the sign-off chain, this is **not** a cortex refutation and **not**
a statement about the learner. The learner was never involved: the soft bank was
empty and no training ran.

**Authorization:** user sign-off S1 + S2 + S3 relayed by the manager
2026-10-04, recorded in
[`SIGNOFF_CHAIN.md`](../experiments/r20-task-support-repair-v1/SIGNOFF_CHAIN.md)§S.
No model run beyond DEV; no threshold chosen; no meta-test access; no
optimization; no score or threshold change.

## What was run

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
TRANSFORMERS_VERBOSITY=error OCZY_REMOTE_CPU_ONLY=1 \
OCZY_MODEL_DIR=/kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
HF_HUB_OFFLINE=1 PYTHONPATH=/home/nyanpasu/Desktop/code/kinoSoft/oczy/src \
bwrap --die-with-parent --unshare-net --tmpfs / --ro-bind /usr /usr \
  --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
  --ro-bind /etc /etc --ro-bind /home /home --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind $R/model /kaggle/input/models/qwen-lm/qwen2.5/transformers/0.5b-instruct/1 \
  --bind $R $R --chdir /home/nyanpasu/Desktop/code/kinoSoft/oczy -- \
  $R/runtime/bin/python scripts/probe_r20_v3_oracle.py \
    --public-root experiments/r20-taskgen-v3-dev/instrument/public \
    --output $R/g2-v3
```

Exit code **0** (the screen completed; a *failed gate* is a result, not an
execution error, so the verdict lives in the artifact, not the exit code).
Wall time 431.4 s. **79 rows.**

Historical runtime: Python 3.12.13, torch 2.10.0+cpu, torchao 0.17.0,
transformers 5.0.0, tokenizers 0.22.2, no network, offline model directory.

Artifacts:
[`artifacts/2026-10-05_r20_g2_oracle_screen/results.json`](artifacts/2026-10-05_r20_g2_oracle_screen/results.json)
(SHA-256 `611f69b1732b66c6b0aecb675c162096948ba586fe69ca676e00d5e02343cfe1`),
plus `stdout.log` and `stderr.log`.

Instrument binding, all checked in-run and failing closed otherwise:

| Check | Value |
|---|---|
| `dev_view_sha256` | `593090c405758617ae9f750cf9c1075e999c92f8c7b677affcb04c3855b3ca26` (the frozen v3 view) |
| `definition_sha256` | `ab99c17358a3af3efb577563b8cd33639519084fa88648c3bc1e05dce86d4bee` |
| scorer | matches the frozen registry binding |
| organ hash before / after | `a342431c…f9ea` / identical |
| organ matches frozen binding | **true** |
| `optimizer_steps` | 0 |
| `training_run` | false |
| `thresholds_selected` | false |
| `meta_test_accessed` / `calibration_accessed` / `sealed_accessed` | false / false / false |
| soft bank width | 0 (no cortex state exists) |

## Result

Gate criterion, fixed before the run: **oracle_context correct > 0**.

| Condition | Correct | Total | Role |
|---|---:|---:|---|
| `no_context` | 0 | 32 | the floor |
| `teaching_context` | 4 | 32 | **retrieval comparator** — the bar, not a result |
| `oracle_context` | **0** | **15** | **the gate** |

Per family, oracle context: contextual_remap 0/5, rule_transformation 0/5,
finite_state 0/5.

**G2 FAILED.** The frozen organ produced no exact answer on any of the 15
oracle-controlled v3 probes.

## The distinction that matters

The 4 correct responses are **all** in `teaching_context`, and **none** in
`oracle_context`. This is the retrieval bar doing exactly its job: with the
teaching transcript present, the model can echo a taught answer (3× `twem`,
1× `q2`). Under oracle control — where the mapping is stated in full — it
produced 0 exact answers.

A descriptive breakdown (**not** a score, not a threshold, no substring credit
applied anywhere in the gate):

| Condition | Exact | Generated text *contains* the expected token |
|---|---:|---:|
| `no_context` | 0/32 | 1/32 |
| `teaching_context` | 4/32 | 18/32 |
| `oracle_context` | 0/15 | 4/15 |

The organ frequently emits the right token wrapped in prose — e.g. for the
finite-state oracle it answers "From the given transition graph, we can see that
the graph is structur…". The frozen scorer is exact normalized equality with
substring matching explicitly rejected, so prose fails. This is a
**response-format** failure layered on top of whatever articulation failure
exists, and the two are not separable from this run alone.

## Diagnosis (offered, not established)

The most likely mechanism is **not** the task-support repair and **not** the
cortex. It is that v3 inherited v2's frozen prompt registry byte-for-byte, and
that registry lacks the two prompt amendments that were approved on 2026-09-11
for the *v2* instrument:

- **Amendment A** (a bare-answer system instruction on every probe) and
  **Amendment B** (complete oracle rule descriptions for transformations) are
  recorded in
  [`2026-09-11_campaign_r20_dev_output_path.md`](2026-09-11_campaign_r20_dev_output_path.md).
  Under those amendments the v2 oracle sampled 2/3 correct; before them it was
  0/3.

v3's oracle prompts visibly carry the un-amended form: the transformation oracle
says `Rule: composition with parameters 'reverse' and 'pex'` followed by worked
examples, i.e. the *shorthand* that Amendment B existed to replace, and no probe
carries the bare-answer system instruction from Amendment A. The G1 freeze
asserted that the prompt registry equals v2's frozen value, and it does — which
is exactly why v3 cannot have inherited the v3/v4 amendments. Adopting them would
change the prompt registry hash, which is precisely the change G1 was built to
refuse without a new human sign-off.

So the honest reading is: **G2 failed against a prompt format that a previous
human-approved repair already identified as defective for this organ, and that
repair is not part of v3.** This is a strong argument that G2 should be re-run
under a separately frozen v4 with the amendments — but that is a new instrument
version requiring its own human sign-off, not something this card authorizes.

Two alternatives are not distinguishable from this run and are not claimed:
(a) the un-amended prompt format alone is responsible; (b) the task-support
repair's richer probes make the task harder than v2's. Nothing here separates
them.

## What this gate does NOT establish

1. **Not a cortex refutation.** No cortex state existed. This says nothing about
   whether a learned state could carry the task.
2. **Not a task-support verdict.** The v3 tasks are sound by construction: all 22
   defect classes are 0, 758/758 support certificates verify, and the leakage
   audit passed. A sound task that a frozen organ cannot answer under oracle
   control is an interface finding, not a task defect.
3. **Not comparable to any v2 number.** v1/v2/v3 scores are not comparable by
   construction.
4. **Not a learnability result.** G3 was not run.
5. **Not a powered-design result.** G4 was not run; no margin, sample size or
   threshold exists for v3.
6. **Not a threshold selection.** `gate_criterion` was fixed as "oracle correct
   > 0" before the run and chosen from the sign-off chain's definition of G2, not
   from any result. No number in this entry was fitted to the data.

## Limitations (carried forward, unresolved)

- **R20 local reproduction: four nonzero state hashes still differ** (untrained,
  trained, feedback-shuffled, state-swapped). Unresolved; inherited.
- **eval-v2 A2 and C repairs remain unsigned and unapplied.** No capability claim
  on eval-v2 stands.
- **758 flip-target mutation checks remain vacuous by construction**, reported
  separately from the 688 discriminating delete-a-fact checks.
- **Sample size.** 15 oracle probes over 15 tuning tasks, one per family index.
  The zero is unambiguous in direction, but this is not a full-catalog estimate;
  v3's 90 calibration tasks were not screened (that view is held back by design).
- **The prose/format confound is unresolved.** "Cannot express the task" and
  "expresses it in prose that the exact scorer rejects" are not separated here.
- **The first G2 attempt was OOM-killed (exit 137)** under host memory pressure
  (swap 100% full, 8.5 GB available) — an infrastructure failure, not a result.
  It was re-run to completion with `OMP_NUM_THREADS=2`. Only the completed run
  is reported; the killed attempt produced no artifact.
- **Local run, not a clean-source remote campaign.** Working tree at
  `9031ce2ca95022f95cdad515e2071855eef1d8ac` with a large uncommitted set; no
  source archive.

## Sequence status

| Gate | Status |
|---|---|
| S1/S2/S3 sign-off | recorded |
| G1 freeze | **PASSED** — `definition_sha256` `ab99c173…` |
| G2 oracle screen | **FAILED — articulation block; sequence stops here** |
| G3 learnability | not run (blocked by G2) |
| G4 calibration/power | not run (blocked) |
| G5 candidate sign-off | not reached; meta-test stays blocked |

**Next step requires a human decision**, not an autonomous one: either freeze a
separately versioned `meta_cortex/v4` that includes the 2026-09-11 prompt
amendments and re-run G2 under it, or accept that the mouth–cortex protocol is
blocked for the v3 lineage.