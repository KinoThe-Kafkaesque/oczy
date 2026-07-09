# Remote research compute guide

This is the required workflow for using Kaggle compute for Oczy research. The
remote service is an execution substrate, not an authority over the experiment.
It may accelerate a pre-registered workload, but it may not change the
instrument, select thresholds, inspect meta-test during development, or turn an
infrastructure smoke result into evidence for the cortex hypothesis.

## Approved state and fixed references

As of 2026-07-09:

| Resource | Status | Fixed reference |
|---|---|---|
| Kaggle CPU | Verified | private CPU kernel path in [`cpu-smoke/`](cpu-smoke/) |
| Kaggle GPU | Verified | 2×Tesla T4 via `NvidiaTeslaT4`; [`t4-smoke/`](t4-smoke/) |
| P100 | Blocked | Kaggle PyTorch 2.10 CUDA image lacks sm_60 kernels |
| L4 request | Blocked | request silently allocated an incompatible P100 |
| TPU | Not wired | quota exists, but no XLA/JAX parity probe or runner exists |
| Language organ | Version pinned and remotely verified | `qwen-lm/qwen2.5/transformers/0.5b-instruct/1` |

The pinned Qwen artifact is version 1, 999,604,126 bytes, with:

- model type `qwen2`;
- hidden size 896;
- 24 transformer layers and 14 attention heads;
- `config.json` SHA-256
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`;
  and
- `model.safetensors` SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

Changing the model variation, quantization, tokenizer, chat template, or model
version changes the language organ and therefore requires a new pre-registered
experiment version. It is not a compute-only substitution.

The remote model probe passed on 2×T4 with zero trainable Qwen parameters, no
parameter gradients, a finite input-embedding gradient, and an unchanged
parameter fingerprint. See [`RESULTS.md`](RESULTS.md) for exact timings and
hashes.

## Prepared building blocks

| Building block | Purpose |
|---|---|
| [`run_cortex_smoke.py`](run_cortex_smoke.py) | CPU/CUDA gradient, frozen-state, and throughput plumbing without a real LM |
| [`run_qwen_model_probe.py`](run_qwen_model_probe.py) | Locate and hash the attached Qwen artifact; verify frozen-parameter and input-gradient behavior |
| [`prepare_source_bundle.py`](prepare_source_bundle.py) | Create a commit-addressed source archive and private Kaggle dataset metadata |
| [`prepare_research_kernel.py`](prepare_research_kernel.py) | Generate a private, internet-off CPU/T4 kernel with source/model/provenance checks |
| [`RESULTS.md`](RESULTS.md) | Verified hardware results and accelerator nulls |

Generated bundles and kernels belong in a temporary directory or the ignored
`infrastructure/kaggle/build/` directory. Do not hand-edit a generated
`run.py`; change the generator or job arguments and regenerate it.

## Which compute to use

| Work | Default compute | Reason |
|---|---|---|
| Instrument materialization, split/leakage audits, distribution checks | CPU | deterministic and no frozen-organ gradient workload |
| Scorer tests, report aggregation, bootstrap CIs | CPU | GPU overhead adds no value |
| Oracle articulation and frozen-Qwen interface checks | 2×T4 | model forward/backward dominates |
| Developmental outer-loop training | T4, initially one developmental seed per GPU | independent seeds avoid needless gradient synchronization |
| Large single-seed training after profiling | T4 with DDP | only when a measured single-seed workload benefits |
| Immutable meta-test | Same validated substrate as development | compute changes must not become an unregistered variable |
| TPU | Do not schedule | current PyTorch runner would fall back to CPU; an explicit XLA/JAX implementation and parity run are required first |

Small 64×64 cortex-only math can be faster on CPU. The verified 2.016× T4
gain appeared when the differentiable frozen-organ interface used width 896.
Profile the real workload before fan-out; “GPU available” is not itself a
reason to consume GPU quota.

## End-to-end workflow

### 1. Freeze the research intent locally

Before preparing remote artifacts:

1. identify the research specification and exact phase;
2. verify the relevant instrument manifest locally;
3. confirm that the optimizing code and measuring instrument are separated;
4. define the one variable changed by this run;
5. fix seed lists, conditions, stopping rule, output schema, and kill criteria;
6. run unit/smoke tests locally; and
7. commit the exact source intended for execution.

For `meta_cortex/v1`, development may use meta-train and meta-validation only.
Generating or running a meta-test kernel requires the frozen instrument
manifest hash and a human sign-off identifier.

### 2. Build an immutable source dataset

Use a clean worktree. The preparer refuses dirty state by default.

```bash
COMMIT=$(git rev-parse HEAD)
BUILD="infrastructure/kaggle/build/source-${COMMIT:0:12}"

uv run python infrastructure/kaggle/prepare_source_bundle.py \
  --revision "$COMMIT" \
  --output "$BUILD"

jq . "$BUILD/source_manifest.json"
kaggle datasets create --path "$BUILD"
```

The dataset is private by default. Its slug contains the first 12 commit
characters, while the manifest records the full commit and archive SHA-256.
Do not use `--allow-dirty-worktree` for a scored, oracle, developmental, or
meta-test run. That option exists only for explicitly labelled infrastructure
development; the manifest preserves the dirty-state warning.

Do not update an old source dataset in place. Create a new commit-addressed
dataset so a kernel cannot silently receive newer code under the same source
reference.

### 3. Verify the pinned model setup

The kernel metadata must contain exactly:

```json
"model_sources": [
  "qwen-lm/qwen2.5/transformers/0.5b-instruct/1"
]
```

Internet remains disabled. Run the model probe after a Kaggle image change or
before the first real experiment batch:

```bash
kaggle kernels push \
  --path infrastructure/kaggle/qwen-t4-probe \
  --accelerator NvidiaTeslaT4 \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-language-organ-t4-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-language-organ-t4-probe \
  --path reports/kaggle/qwen-t4-probe \
  --force
```

The probe must report the expected file hashes, an actual supported CUDA
device, zero trainable model parameters, no model-parameter gradients, a
finite non-zero input-embedding gradient, and an unchanged parameter
fingerprint.

Experiment code must load from `OCZY_MODEL_DIR` (set by the generated
bootstrap) or an explicit `--model-path`, with `local_files_only=True`. It must
never fall back to Hugging Face network resolution.

### 4. Generate a phase-specific kernel

Read the values produced in step 2:

```bash
SOURCE_MANIFEST="$BUILD/source_manifest.json"
SOURCE_COMMIT=$(jq -r .commit "$SOURCE_MANIFEST")
SOURCE_SHA=$(jq -r .archive.sha256 "$SOURCE_MANIFEST")
SOURCE_DATASET=$(jq -r .dataset_id "$SOURCE_MANIFEST")
```

Example: generate the Experiment 09 developmental job after its module exists:

```bash
JOB="infrastructure/kaggle/build/meta-cortex-development-seed-0"

uv run python infrastructure/kaggle/prepare_research_kernel.py \
  --output "$JOB" \
  --kernel-id abdellahkadem/oczy-meta-cortex-development-seed-0 \
  --title "Oczy Meta Cortex Development Seed 0" \
  --phase development \
  --profile t4 \
  --source-dataset "$SOURCE_DATASET" \
  --source-commit "$SOURCE_COMMIT" \
  --source-archive-sha256 "$SOURCE_SHA" \
  --module oczy.experiments.meta_cortex.train_outer \
  --arg=--instrument \
  --arg=v1 \
  --arg=--developmental-seed \
  --arg=0
```

The generator embeds the job spec into `run.py` and writes a reviewable
`job_spec.json` plus `kernel-metadata.json`. It automatically attaches the
pinned Qwen model to oracle, development, and meta-test phases. Instrument and
analysis jobs remain model-free unless a model source is explicitly supplied.

For a meta-test job, generation additionally requires:

```bash
--instrument-manifest-sha256 <64-hex-hash> \
--human-signoff-id <recorded-human-approval>
```

These fields are provenance gates, not permission for an agent to approve its
own instrument.

### 5. Review before submission

Review all three generated files. Confirm:

- source commit and archive hash match the uploaded private dataset;
- kernel and dataset slugs include the intended seed/run identity;
- `is_private` is true and `enable_internet` is false;
- the phase and module are correct;
- only the intended arguments changed from the matched control;
- T4 jobs request `NvidiaTeslaT4` explicitly;
- the pinned Qwen version is unchanged;
- no API key, OAuth file, local path, expected answer, or meta-test content is
  embedded; and
- the timeout is small enough to stop a runaway job.

### 6. Submit, monitor, and retrieve

```bash
kaggle kernels push \
  --path "$JOB" \
  --accelerator NvidiaTeslaT4 \
  --timeout 21600

kaggle kernels status abdellahkadem/oczy-meta-cortex-development-seed-0
kaggle kernels logs abdellahkadem/oczy-meta-cortex-development-seed-0
kaggle kernels output \
  abdellahkadem/oczy-meta-cortex-development-seed-0 \
  --path reports/kaggle/meta-cortex-development-seed-0 \
  --force
```

Check `kaggle quota --csv` before fan-out. Start with one seed. Expand to the
pre-registered seed set only after the first artifact passes provenance,
hardware, model, and output-schema checks.

The requested accelerator is not evidence of the allocated accelerator. Trust
the runtime hardware record in `remote_run_provenance.json` and the experiment
report. The L4→P100 fallback already demonstrated this failure mode.

### 7. Validate and promote the artifacts

Every completed remote run must produce:

1. `remote_run_provenance.json` from the bootstrap;
2. the experiment's versioned result JSON;
3. developmental curves and checkpoints for development runs only;
4. language-organ before/after hashes;
5. instrument manifest hash where applicable;
6. actual hardware/framework versions;
7. task and seed counts;
8. conditions including nulls and matched baselines; and
9. trace-deletion/fixed-width audits required by the research spec.

Before interpretation, verify:

- `source_manifest.commit` and archive SHA match the planned run;
- model artifact and frozen-parameter hashes match;
- actual hardware matches the approved profile;
- the remote module and arguments match `job_spec.json`;
- all expected seed/condition artifacts exist;
- no checkpoint or report contains forbidden raw traces or meta-test leakage;
  and
- the result reproduces locally on a small fixture where practical.

Copy the durable result summary into a new dated file under
`experiments_logs/`, include the Kaggle kernel URL/version and exact commands,
and update the ledger. Keep pulled bulk artifacts under `reports/kaggle/`; they
are intentionally ignored by Git.

## Failure policy

Treat a run as **BLOCKED/INVALID**, not as a scientific null, when:

- source, archive, model, or instrument hashes differ;
- Kaggle allocates the wrong hardware;
- a P100 or unsupported compute capability appears;
- internet-backed model fallback occurs;
- the model gains parameter gradients or changes;
- a required artifact is missing or corrupt;
- development accesses meta-test;
- the runner changes more than the registered variable; or
- fewer independent tasks/seeds run than the claim requires.

Treat a correctly executed negative behavioral result as a real null and log
it prominently. Infrastructure failure and hypothesis refutation are different
outcomes.

## TPU admission gate

Having TPU quota does not make TPU an approved executor. TPU may be added only
after a separate versioned probe demonstrates:

1. an explicit `torch_xla` or JAX path—never CPU fallback;
2. parity with the CPU/T4 tensor update and scorer fixtures;
3. frozen Qwen parameter and tokenizer identity;
4. stable static shapes for the compiled step;
5. actual TPU topology and device utilization in provenance; and
6. a measured advantage after compilation time is included.

Until then, use CPU and the verified 2×T4 path.

## Short preflight checklist

- [ ] Research spec and one changed variable identified.
- [ ] Instrument/version/threshold distribution frozen and authorized.
- [ ] Exact source committed; worktree clean.
- [ ] Source bundle and archive SHA reviewed.
- [ ] Qwen model source is the pinned version-1 reference.
- [ ] Private kernel, internet disabled, timeout set.
- [ ] Correct CPU/T4 phase selected; no TPU/P100/L4 assumption.
- [ ] First seed passes before fan-out.
- [ ] Actual hardware and all hashes verified from pulled artifacts.
- [ ] Nulls, errors, commands, URL/version, and artifacts logged durably.
