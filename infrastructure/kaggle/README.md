# Remote CPU compute pool for offline cortex work

This directory provides a private, reproducible **CPU-only** compute pool
using **Kaggle Kernels** and **Google Colab CLI** for offline developmental
work around Research/20 and Experiment 09.

Use [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for every real research run. It
defines source/model pinning, CPU profile generation, submission/retrieval,
provenance, meta-test gates, and the CPU-only contract.

The checked-in kernels are deliberately **infrastructure smoke tests**, not
cortex experiments. They run a synthetic learned-writer -> fixed fast/slow state
-> latent coupler -> frozen differentiable organ path. The state remains 64x64
and the organ interface uses Qwen2.5-0.5B-Instruct's real hidden width (896),
but no Qwen weights are loaded for the smoke task. The jobs do not use `eval/v2`,
`meta_cortex/v1`, retrieval, episode-conditioned code, or a real language
model, and their losses cannot be cited as evidence for H-META-CORTEX.

## Provider availability

| Provider | Profile | Status | Since |
|---|---|---|---|
| Kaggle CPU | `cpu` | Active | 2026-07-10 |
| Colab (CLI 0.6.0) | `cpu` | Active | 2026-07-11 |
| GPU (T4/P100/L4) | --- | Archived -- do not submit | 2026-07-09 |
| TPU | --- | Not wired | --- |

## What is wired

| Profile | Kaggle kernel | Requested machine | Intended use |
|---|---|---|---|
| CPU | `abdellahkadem/oczy-cortex-cpu-smoke` | Kaggle CPU | instrument materialization, leakage/distribution audits, scoring, aggregation |
| CPU | `abdellahkadem/oczy-qwen-cpu-probe` | Kaggle CPU | frozen real-model and input-gradient verification on CPU |

All kernels are private, have internet disabled, and set `enable_gpu: false`.
The smoke kernel writes `cortex_smoke_report.json` to `/kaggle/working`. The
report records the hardware actually allocated, source hash, throughput,
held-out loss, finite-gradient checks, and before/after hashes of every frozen
parameter. See [`RESULTS.md`](RESULTS.md) for the verified CPU smoke result and
the acceptance contract.

### CPU-only contract

Active runners and generated bootstrap code enforce a strict CPU-only
contract:

- Kernel metadata sets `enable_gpu: false` and `enable_tpu: false`.
- The generated bootstrap sets `CUDA_VISIBLE_DEVICES=""` before importing
  torch and verifies the env var is still empty at startup, with no CUDA
  or NVML runtime query performed.
- No active code calls `torch.cuda.*` for device selection, model placement, or
  profiling. Legacy report fields (`cuda_available`, `cuda_device_count`,
  `torch_cuda_version`) remain as constant `false`/`0`/`null` values for schema
  compatibility.
- Colab CLI jobs never pass `--gpu` or `--tpu`; the scheduler argv omits them.

## Local preflight

```bash
kaggle --version
kaggle kernels list --mine --page-size 1 --csv
kaggle quota --csv
colab --help
colab sessions

uv run python infrastructure/kaggle/run_cortex_smoke.py \
  --device cpu \
  --output /tmp/oczy-cortex-local-smoke.json
```

The `--device` argument accepts only `cpu`. Authentication uses Kaggle's OAuth
credentials at `~/.kaggle/credentials.json` or the Colab CLI OAuth flow
(`colab login`). Do not copy credentials into the repository or a kernel source
directory.

## Submit and retrieve (Kaggle)

CPU kernels do not use `--accelerator`. The metadata already specifies
`enable_gpu: false`.

```bash
# Cortex CPU smoke
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels logs abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force
```

```bash
# Qwen CPU probe
kaggle kernels push \
  -p infrastructure/kaggle/qwen-cpu-probe \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels logs abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-cpu-probe \
  --path reports/kaggle/qwen-cpu-probe \
  --force
```

A run counts as wired only when its JSON artifact has `passed: true`,
`cuda_available: false`, and `cuda_device_count: 0`. See
[`RESULTS.md`](RESULTS.md) for the full acceptance contract.

## Parallel batch scheduling

For many-jobs fan-out, the **parallel scheduler** (`parallel_scheduler.py`)
manages a batch of remote CPU jobs (Kaggle kernels and/or Colab sessions) with
additive provider capacity, crash-safe durable state, and automatic resume.

### Batch schemas

#### v1 (``oczy/kaggle-parallel-batch/v1``) — Kaggle only

```json
{
  "schema_version": "oczy/kaggle-parallel-batch/v1",
  "jobs": [
    {
      "name": "smoke-seed-0",
      "kernel_dir": "build/smoke-seed-0",
      "output_dir": "reports/kaggle/smoke-seed-0"
    },
    {
      "name": "smoke-seed-1",
      "kernel_dir": "build/smoke-seed-1",
      "output_dir": "reports/kaggle/smoke-seed-1"
    }
  ]
}
```

Each ``kernel_dir`` must be produced by ``prepare_research_kernel.py`` with
``kernel-metadata.json`` and ``job_spec.json``. Validation rejects any kernel
that is not private, CPU-only, or whose title would create a different Kaggle
slug than its ``id`` field.

#### v2 (``oczy/remote-parallel-batch/v2``) — mixed providers

Each job carries a ``provider`` field (``"kaggle"`` or ``"colab"``):

```json
{
  "schema_version": "oczy/remote-parallel-batch/v2",
  "jobs": [
    {
      "name": "kaggle-seed-0",
      "provider": "kaggle",
      "kernel_dir": "build/kaggle-seed-0",
      "output_dir": "reports/kaggle-seed-0"
    },
    {
      "name": "colab-run-0",
      "provider": "colab",
      "script": "scripts/train.py",
      "arguments": ["--seed", "42"],
      "output_dir": "reports/colab/run-0",
      "timeout": 3600
    }
  ]
}
```

Colab job fields:
- **script** (required) — path to the Python script, resolved relative to the
  manifest file's directory.
- **arguments** (optional) — list of string CLI arguments.
- **output_dir** (required) — local directory for pulled stdout/stderr/result.
- **timeout** (optional) — per-job wall-clock timeout in seconds.

### Provider-agnostic run and status

```bash
# Run a batch (v1 or v2)
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json

# Check status without submitting
uv run python infrastructure/kaggle/parallel_scheduler.py status \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| ``--max-parallel N`` | omitted | Omit for additive provider capacity (kaggle_max + learned Colab). Explicit N caps total concurrency for backward compatibility (must be >= 1). |
| ``--kaggle-max N`` | 10 | Max concurrent Kaggle kernels (hard max 10) |
| ``--colab-max N`` | 10 | Colab AIMD capacity ceiling (additive probe, not a hard quota) |
| ``--colab-cooldown SEC`` | 60 | Seconds to wait after a Colab 412 rejection before retrying |
| ``--push-timeout SEC`` | 21600 | Kaggle kernel run-time limit |
| ``--job-timeout SEC`` | 21600 | Max wall-clock wait per job |
| ``--poll-interval SEC`` | 30 | Seconds between status polls |
| ``--watch-batch`` | off | Live watch mode: keep the scheduler alive after all jobs finish, re-read the batch file when it changes, and merge unseen job names as new pending jobs. Exit with Ctrl-C. |
| ``--watch-interval SEC`` | 30 | Seconds between batch-file change checks when idle in watch mode (must be > 0 when ``--watch-batch`` is set) |

**Omitting ``--max-parallel`` means additive provider capacity.** The scheduler
imposes no global concurrency cap — Kaggle jobs fill up to ``--kaggle-max``
(hard-capped at 10) and Colab jobs fill up to the AIMD-learned limit
(``--colab-max`` ceiling). For example, ``--kaggle-max 10`` plus a learned
Colab limit of X allows 10 Kaggle + X Colab jobs to run concurrently.

**Explicit ``--max-parallel N`` caps total concurrency** across all providers
for backward compatibility. When provided, N must be >= 1.

**``--colab-max`` sets a ceiling, not a guaranteed capacity.** The Colab
AIMD controller starts admissions at 1 and probes upward — actual available
slots depend on account-level session limits and external sessions. See
[RESULTS.md](RESULTS.md) for the discovered `learned_limit` and probe
behavior.

### Lifecycle

The ``--state`` file is written atomically after every transition:

1. **pending** — ready to submit
2. **submitting** — push/launch in progress
3. **running** — submitted, waiting for remote completion
4. **collecting** — complete, downloading output
5. **succeeded** — output collected

A job enters **failed** on push error, kernel error status, output collection
failure, or timeout.

**Resume**: kill the process (or reboot) and re-run the same command.
Interrupted ``submitting``/``collecting`` jobs are converted back to
``pending``/``running``, already-``running`` kernels are never resubmitted,
and Colab running jobs on restart (no local process handle) are failed as
interrupted with best-effort session stop. Only new jobs from an updated
batch manifest are added.

The JSON summary is printed on completion (exit code 0 if all succeeded,
1 if any failed). The final state file is preserved for post-hoc inspection.

### Live watch mode

``--watch-batch`` keeps the scheduler alive after all jobs reach a terminal
state. The batch file is periodically re-read (every ``--watch-interval``
seconds when idle); any unseen job names are merged as new ``pending`` jobs.
Existing jobs are never modified -- their definitions and lifecycle states are
preserved exactly. Malformed or partially-written reloads are logged to stderr
and retried on the next watch cycle without terminating the daemon.

```bash
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  infrastructure/kaggle/my-batch.json \
  --state /tmp/parallel-state.json \
  --watch-batch \
  --watch-interval 15
```

The scheduler runs until interrupted with Ctrl-C. On interruption, state is
persisted atomically and a JSON summary is printed (exit code 0 if all jobs
succeeded, 1 if any failed). Without ``--watch-batch``, the scheduler
terminates when all jobs finish -- unchanged from previous behavior.

### Colab session lifecycle differences

Colab jobs have a different lifecycle from Kaggle kernels:

- **argv**: ``colab run --keep --session <name> --timeout <sec> -- SCRIPT ARGS...``
  The ``--`` separator ensures script paths/arguments starting with ``--`` are
  not consumed as CLI flags. The ``--keep`` flag means the session daemon
  persists on the backend; the scheduler's ``stop()`` call explicitly
  unassigns the VM (``colab stop --session <name>``).
- **Output**: ``colab`` writes ``stdout.log``, ``stderr.log``, and
  ``result.json`` (containing ``ok``, ``error``, ``exit_code``, ``status``,
  ``session``) into ``output_dir``.
- **Session cleanup**: On success, error, timeout, and capacity rejection,
  ``stop()`` is called to free the backend VM slot. The local ``Popen`` is
  killed and reaped.
- **External session accounting**: The scheduler queries ``colab sessions``
  (cached once per loop iteration) to count account-wide active sessions.
  External sessions count toward the admission gate — they reduce slots
  available to scheduler jobs even if the scheduler did not start them.
- **Restart recovery**: On restart, ``detect_orphaned_sessions`` intersects
  known scheduler session names with the backend session list and stops any
  leaked sessions. A ``sessions()`` probe runs once at restart so external
  sessions can drain without false failure.

### Colab AIMD admission control

Colab capacity is probed dynamically, not hardcoded:

- Starts admission at 1 (``COLAB_AIMD_START``).
- **On success** (first confirmed running poll): learned_limit += 1 (capped
  at ``--colab-max``).
- **On HTTP 412 TooManyAssignmentsError**: learned_limit reduced to the
  account-wide active Colab session count (including external sessions,
  min 1). A cooldown period (``--colab-cooldown``) prevents immediate
  retry.
- **On admission gate block** (external sessions fill capacity): job stays
  pending — it does not count as a capacity rejection or increment the
  rejection counter. Only an actual 412 process exit increments the
  consecutive-rejection counter.
- After 10 consecutive 412 rejections the job fails (prevents infinite
  retry when external sessions permanently occupy all capacity).

The learned_limit is persisted in the v2 state file as
``colab_learned_limit`` and restored on restart. **It is not a hardcoded
maximum** — it self-corrects on the next 412 and continues probing.

### Verified concurrent execution (Kaggle)

Two CPU smoke kernels
(``abdellahkadem/oczy-scheduler-cpu-smoke-1`` and
``abdellahkadem/oczy-scheduler-cpu-smoke-2``) were submitted with
``max_parallel=2`` on 2026-07-10. Both succeeded in one attempt and
reported ``passed: true``, ``cuda_available: false``. The scheduler
recorded submission timestamps differing by approximately 1.85 s while
completion times overlapped, confirming genuine concurrent remote
execution on Kaggle CPU. See [`RESULTS.md`](RESULTS.md) for the
complete evidence.

### Verified queue-starvation fix (Colab, 2026-07-11)

The first live four-job Colab run exposed a bug that caused the admission gate
to increment ``capacity_rejections`` on every poll cycle when a pending job was
blocked by full capacity. After ``COLAB_MAX_CAPACITY_REJECTIONS`` (10) blocks,
the job was failed even though no ``TooManyAssignmentsError`` had occurred —
it was normal queueing. The fix: admission gate blocks (``cached_external_active
>= effective_limit``) do not touch ``capacity_rejections``. See
[`RESULTS.md`](RESULTS.md) for the full evidence.

## Source bundling and kernel generation (Kaggle)

The source and kernel preparation commands used by the full workflow are:

```bash
uv run python infrastructure/kaggle/prepare_source_bundle.py --help
uv run python infrastructure/kaggle/prepare_research_kernel.py --help
```

The kernel generator accepts `--profile cpu` (the only active profile) and
produces a private, internet-off kernel with a CPU-only bootstrap. The
bootstrap verifies a commit-addressed source archive, discovers the attached
version-pinned model, disables network-backed model resolution, records runtime
provenance, and then executes one Python module. See
[`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for the end-to-end workflow.

## Archived GPU material

Historical GPU verification (T4, P100, L4, and the T4-based Qwen model probe)
is preserved under [`archive/gpu/`](archive/gpu/). That material — including
kernel metadata, the 2×T4 throughput comparison, P100/L4 compatibility nulls,
and the T4 model probe results — is historical evidence only. GPU kernels and
metadata under the archive must not be resubmitted. See
[`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md) for the full historical
record.
