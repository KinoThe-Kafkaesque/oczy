# Remote CPU compute verification — 2026-07-10/11

**Result class:** infrastructure verification, not a Research/20 experiment

**Active remote profile:** CPU only (Kaggle Kernels + Colab CLI 0.6.0)

## Active tasks

| Task | Slug / provider | Profile | Local status | Remote status |
|---|---|---|---|---|
| Cortex smoke (Kaggle) | `abdellahkadem/oczy-cortex-cpu-smoke` | `cpu` | PASS | verified 2026-07-10 (v4) |
| Generated bootstrap probe (Kaggle) | `abdellahkadem/oczy-cpu-bootstrap-probe` | `cpu` | PASS | verified 2026-07-10 (v4) |
| Qwen CPU probe (Kaggle) | `abdellahkadem/oczy-qwen-cpu-probe` | `cpu` | PASS | verified 2026-07-10 (v1) |
| Colab CLI 0.6.0 CPU sessions | Colab (installed CLI) | `cpu` | PASS | verified 2026-07-11 |

The `cpu-smoke` task is the infrastructure plumbing verification: a synthetic
learned-writer -> fixed fast/slow state -> latent coupler -> frozen differentiable
organ path with a 64x64 cortex state and a width-896 frozen-organ interface. No
Qwen weights are loaded, no `meta_cortex/v1` code runs, and the result is not
evidence for H-META-CORTEX.

The `qwen-cpu-probe` task verifies the pinned Qwen2.5-0.5B-Instruct model
artifact on CPU: frozen-parameter hashes, zero trainable parameters, finite
input-embedding gradient, and no parameter-fingerprint change. The kernel
metadata lives in [`qwen-cpu-probe/`](qwen-cpu-probe/) and uses
`enable_gpu: false`. Remote acceptance verified on 2026-07-10 (v1).

The Colab CLI 0.6.0 was installed and the `colab sessions` command confirmed
the OAuth-authenticated backend was reachable. Safe allocation probes (see
below) confirmed CPU session creation, execution, and cleanup.

## Acceptance contract

A remote run counts as accepted only when **all** of the following hold:

1. The remote service reports completion (Kaggle `status complete` / Colab
   exit code 0 with no capacity-rejection markers).
2. The pulled JSON artifact has `passed: true` (Kaggle) or `result.json`
   `ok: true` (Colab).
3. For Kaggle kernels: `remote_run_provenance.json` or the report JSON records
   `cuda_available: false` and `cuda_device_count: 0`. For Colab: argv never
   contains `--gpu` or `--tpu`.
4. Source hash, model hashes (for probe jobs), and frozen-parameter hashes
   match the locally verified values.
5. The kernel metadata used `enable_gpu: false`, `enable_tpu: false`, and
   `enable_internet: false` (Kaggle); Colab uses CPU-only CLI flags.

Any run that reports CUDA availability, a non-empty `CUDA_VISIBLE_DEVICES`, or
a GPU device name is **BLOCKED** — it means the CPU-only contract was violated.

## Verified CPU smoke result (Kaggle, 2026-07-10)

The CPU smoke kernel
[`oczy-cortex-cpu-smoke` v4](https://www.kaggle.com/code/abdellahkadem/oczy-cortex-cpu-smoke)
completed remotely on a Kaggle x86_64 CPU. The report recorded:

- 64x64 fast and slow state shapes;
- a 4x896 latent bank;
- 274,563 trainable smoke-path parameters and 3,473,792 frozen parameters;
- finite losses and gradients;
- held-out improvement of 42.11% (0.4211);
- elapsed: 2.4626 s; throughput: 1,039.6 episodes/s;
- identical before/after frozen-state hash
  `8be7ef2addbe83ce0766a360f9417d9c37a4baf43f97826e0a4faff79048d331`;
- `passed: true`;
- `cuda_available: false`;
- runner SHA `e262b17c8c0918e6464a79a889230bf68d8ed0083d578287cba625f1e0050c8c`;
- torch 2.10.0+cpu.

Pulled artifacts live locally under `reports/kaggle/` and are ignored by Git.

## Pinned Qwen model source

The official Qwen source
`qwen-lm/qwen2.5/transformers/0.5b-instruct/1` is version-pinned for all
model-bearing CPU jobs. The artifact is version 1, 999,604,126 bytes, with:

- model type `qwen2`;
- hidden size 896;
- 24 transformer layers and 14 attention heads;
- `config.json` SHA-256
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`;
- `model.safetensors` SHA-256
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

The historical T4-based model probe is preserved as archived evidence in
[`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md). The active `qwen-cpu-probe`
task confirmed the same model hashes on CPU (v1, 2026-07-10).

## Verified qwen-cpu-probe result (Kaggle, 2026-07-10)

The Qwen CPU probe kernel
[`oczy-qwen-cpu-probe` v1](https://www.kaggle.com/code/abdellahkadem/oczy-qwen-cpu-probe)
completed remotely on a Kaggle x86_64 CPU. The report recorded:

- model: Qwen2.5-0.5B-Instruct, 494,032,768 parameters, float32;
- zero trainable model parameters;
- torch 2.10.0+cpu; transformers 5.0.0;
- model load time: 2.3473 s;
- frozen forward/backward: 0.7218 s;
- input-embedding gradient: present, finite, norm 357.6543;
- model-parameter gradients: absent;
- parameter fingerprint unchanged
  `0ff56033b6a93d267b868cd98a2990b61a958af56aa9268a88974102691d9f5d`;
- model hashes match reference (`config.json` and `model.safetensors` unchanged);
- `passed: true`;
- `cuda_available: false`;
- runner SHA `3c8cddceea2d5ddd5e083db8e42af0d83bfffcf68d9d0363596c8c241d715fe2`.

## Verified generated bootstrap probe result (Kaggle, 2026-07-10)

The generated research-bootstrap kernel
[`oczy-cpu-bootstrap-probe` v4](https://www.kaggle.com/code/abdellahkadem/oczy-cpu-bootstrap-probe)
completed remotely on a Kaggle x86_64 CPU. This kernel exercises the full
`prepare_source_bundle.py` -> `prepare_research_kernel.py` pipeline: a
commit-addressed private source dataset, opaque archive extraction under
`/tmp/...` (not `/kaggle/working`), pinned Qwen model attachment, and the
generated `run.py` bootstrap with hardware and environment provenance checks.

Key evidence:

- clean committed source `9dfa484dc5ea0d48f06673ad27a6b64678ce7619`;
- source dataset `abdellahkadem/oczy-source-9dfa484dc5ea`;
- opaque archive `source.tar.gz.bin`, SHA-256
  `5a0d2990473ac35cf373be4af61f7c4066f113f73c5fc0874430cfd2ae7d77b1`,
  dirty=false;
- source extracted under a temporary directory (`/tmp/...`), not under
  `/kaggle/working`;
- pulled output contains exactly three files:
  `qwen_model_probe.json`, `remote_run_provenance.json`, and the kernel log;
- Qwen CPU probe within the bootstrap passed: float32 forward, finite
  input-embedding gradient norm 357.657470703125, no model-parameter gradients,
  parameter fingerprint unchanged;
- `passed: true`;
- `cuda_available: false`, `cuda_device_count: 0`;
- `CUDA_VISIBLE_DEVICES=""` verified before torch import;
- CPU profile with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
  `enable_gpu: false`, `enable_tpu: false`, `enable_internet: false`.

**Failure history and regression tests.** v1 failed with a raw JSON null
artifact (bootstrap serialization issue). v2 failed when Kaggle auto-extracted
the source tarball, bypassing the private-dataset extraction path. Both failures
are diagnosed, fixed, and covered by regression tests in the infrastructure
test suite. v3 ran successfully but left the extracted source in the pulled
output; v4 produces the compact three-file output documented above.

The generated bootstrap probes the same pinned Qwen2.5-0.5B-Instruct model
(hashes match the reference in the pinned model source section) and the same
CPU-only contract as the `cpu-smoke` and `qwen-cpu-probe` tasks. It is the
infrastructure proof that the full generated-job pipeline -- source bundling,
opaque archiving, kernel generation, extraction, model attachment, bootstrap
execution, and provenance logging -- works end to end on a remote CPU. It is
not evidence for the cortex hypothesis; it is a verified compute substrate.

## Verified Kaggle parallel scheduler result (2026-07-10)

The durable parallel scheduler
(``parallel_scheduler.py``, batch schema ``oczy/kaggle-parallel-batch/v1``)
was verified with two CPU smoke kernels submitted concurrently at
``max_parallel=2`` on 2026-07-10. This test validates the full scheduler
pipeline: batch loading, CPU-only validation, title/slug guard, bounded
concurrent submission, polling, output collection, durable state, and
lifecycle transitions.

**Evidence:**

| Kernel | State | Attempts | ``passed`` | ``cuda_available`` | ``submitted_at`` | ``completed_at`` |
|---|---|---|---|---|---|---|
| ``abdellahkadem/oczy-scheduler-cpu-smoke-1`` | succeeded | 1 | ``true`` | ``false`` | 2026-07-10 23:20:17.60 UTC | 2026-07-10 23:21:05.00 UTC |
| ``abdellahkadem/oczy-scheduler-cpu-smoke-2`` | succeeded | 1 | ``true`` | ``false`` | 2026-07-10 23:20:19.45 UTC | 2026-07-10 23:21:05.95 UTC |

**Concurrent execution proof.** The submission timestamps differ by
approximately 1.85 s (both submitted within one poll cycle at
``max_parallel=2``). Both remained active over the same approximately
45-second interval and completed approximately 0.96 s apart. This is the
expected pattern for two jobs running concurrently on separate remote CPU
machines. It confirms the scheduler submits multiple kernels before polling
the first and that Kaggle dispatched both during the same active interval.

**Lifecycle verification.** Both jobs flowed through the full lifecycle:
pending -> submitting -> running -> collecting -> succeeded, in one
attempt each. The durable state file was written atomically after every
transition and could be inspected with the ``status`` subcommand at any
point.

**Contract enforcement.** The scheduler rejected the original kernel
directories because the title "Oczy Scheduler CPU Smoke 1" generated the
Kaggle slug ``oczy-scheduler-cpu-smoke-1`` while the kernel metadata ``id``
was set to ``abdellahkadem/oczy-scheduler-smoke-1`` -- a mismatch that
would have caused Kaggle to create the kernel under a different slug and
all subsequent polling to fail. This guard was discovered, fixed, and
verified during the same session. See
`prepare_research_kernel.py` ``_title_slug()`` and
``parallel_scheduler.py`` ``_validate_kernel()``.

The state file and pulled kernel output are not checked into the
repository -- they are transient operator artifacts in a temporary
directory, preserved only as session log evidence. The scheduler
verification is infrastructure proof, not a scientific result.

## Verified Colab CLI result (2026-07-11)

The Colab CLI 0.6.0 and the v2 mixed-provider scheduler were verified live
on 2026-07-11 on a standard free-tier Colab account.

### Environment

- Colab CLI version: **0.6.0** (installed via pip).
- OAuth flow completed; `colab sessions` reached the backend.
- Standard free CPU tier (no Colab Pro/Pro+).

### Safe allocation probe

A safe account-capacity probe used `colab new --session <name>` with the
default CPU runtime. Three successive allocations succeeded. The fourth
allocation failed with:

- **HTTP 412 `TooManyAssignmentsError`** (`Precondition Failed`).

The three sessions that were actually allocated were then stopped, and
`colab sessions` confirmed **no active sessions**. The rejected fourth
attempt never created a session.

This establishes an observed capacity of three simultaneous standard CPU
sessions for this account at that time. It is not a permanent quota claim:
Colab does not expose a numeric limit, and availability can vary.

The scheduler therefore learns rather than hardcodes capacity. In the final
four-job run it reduced its limit after the fourth job received a 412, queued
that job until a slot freed, and increased the persisted `learned_limit` to 4
after the retry succeeded. A future fourth concurrent admission will probe
again and self-correct on another 412.

### Standard free CPU RAM

Standard free-tier Colab CPU sessions typically provide approximately
**12.7 GB** of system RAM. This value was not measured in this proof
but is the documented and user-observed typical allocation.

### Final scheduler live batch: four 30-second scripts

A v2 batch manifest with four Colab CPU jobs
(``oczy-colab-pool-v2-1`` through ``oczy-colab-pool-v2-4``) was run through
``parallel_scheduler.py`` with ``colab_max=10`` as an admission ceiling, each
running a 30-second Python script. Results:

| Job | Attempts | Result | Exit code | cpu_count | status |
|---|---|---|---|---|---|
| ``oczy-colab-pool-v2-1`` | 1 | ok true | 0 | 2 | succeeded |
| ``oczy-colab-pool-v2-2`` | 1 | ok true | 0 | 2 | succeeded |
| ``oczy-colab-pool-v2-3`` | 1 | ok true | 0 | 2 | succeeded |
| ``oczy-colab-pool-v2-4`` | 2 | ok true | 0 | 2 | succeeded |

- **First three** (jobs 1–3): submitted concurrently, each succeeded on the
  first attempt.
- **Fourth** (job 4): first attempt was capacity-rejected (412) and queued;
  it retried after a slot freed, then succeeded on attempt 2.
- **Each VM reported ``cpu_count=2``** in the runtime environment.
- **Final ``colab sessions``** confirmed no active sessions.

### Verified queue-starvation fix

The first live four-job Colab batch exposed a queue-starvation bug: the
admission gate in the scheduler's submission phase incremented
``capacity_rejections`` on every poll cycle when a pending job was blocked
solely because `cached_external_active >= effective_limit`. After
``COLAB_MAX_CAPACITY_REJECTIONS`` (10) blocks, the job was failed even
though no ``TooManyAssignmentsError`` process exit had occurred -- it was
normal queueing.

The fix: admission gate blocks (the ``continue`` at the ``if
cached_external_active >= effective_limit`` guard) do not touch
``capacity_rejections``. Only an actual 412 process exit (classified as
``COLAB_CAPACITY_REJECTED`` by `classify_colab_output`) increments the
counter. The test `test_fourth_job_queues_then_succeeds_no_capacity_rejection`
proves that with learned capacity=3 and four jobs, the fourth stays pending
across 12+ admission polls without incrementing ``capacity_rejections``,
then succeeds when a slot frees.

The regression tests in `test_colab_parallel_provider.py` confirm:
- Admission blocks never increment ``capacity_rejections``.
- ``capacity_rejections == 0`` for all four jobs in the queueing scenario.
- ``COLAB_MAX_CAPACITY_REJECTIONS`` (10) consecutive 412 rejections still
  fail a job (infinite retry guard).

### Colab lifecycle evidence

- **argv**: ``colab run --keep --session <name> --timeout <sec> -- <script> [args...]``
  The ``--`` separator ensures script paths or arguments starting with ``--``
  are forwarded correctly (tested: ``test_f3_separator_before_script_in_argv``).
- **No GPU/TPU flags**: argv never contains ``--gpu`` or ``--tpu`` (tested:
  ``test_colab_run_argv_omits_gpu_and_tpu``).
- **Output**: ``stdout.log``, ``stderr.log``, ``result.json`` (with ``ok``,
  ``error``, ``exit_code``, ``status``, ``session``).
- **Cleanup**: ``stop()`` called on success, error, timeout, and capacity
  rejection (tested: ``test_cleanup_stop_called_on_*``). Local Popen killed
  and reaped on timeout (tested: ``test_f4_kill_proc_on_timeout``).
- **External session accounting**: ``colab sessions()`` cached once per
  loop iteration (tested: ``test_f8_sessions_cached_once_per_loop_iteration``).
  External sessions reduce available slots (tested:
  ``test_external_sessions_block_admission``).
- **Restart recovery**: orphan sessions detected and stopped (tested:
  ``test_f5_orphan_recovery_stops_leaked_session``). A ``sessions()`` probe
  runs once at restart so external sessions can drain without false failure.
- **AIMD deferred on_success**: learned_limit does not increase on immediate
  exit (tested: ``test_f9_no_aimd_increase_on_immediate_exit``). Increases
  only after the first poll confirms the proc is RUNNING (tested:
  ``test_f9_aimd_increases_after_confirmed_running``).

## Infrastructure test suite

The verified targeted suite contains **240 behavioral tests** across three
files:

- ``test_kaggle_parallel_scheduler.py`` (73 tests): v1 batch loading,
  lifecycle, concurrency bounds, state persistence, title/slug guard,
  CLI exit codes, and resume semantics.
- ``test_colab_parallel_provider.py`` (119 tests): v2 batch/state schemas,
  mixed-provider slot invariants, Colab AIMD admission, capacity rejection
  lifecycle, session parsing/output collection, orphan detection, cleanup,
  external-session accounting, queue-starvation regression, and F1-F12
  reliability invariants.
- ``test_kaggle_preparation.py`` (48 tests): source bundling, generated
  kernels, bootstrap provenance, CPU-only metadata, and title/slug checks.

All use fake clients, fake processes, or deterministic clocks where remote
behavior is involved; none invoke Kaggle or Colab during the test suite.

New Colab or mixed-provider features must be added with tests following the
same pattern (fake client, deterministic clock, no network).

## Historical GPU evidence

GPU verification results (T4, P100, L4) from 2026-07-09 are preserved under
[`archive/gpu/`](archive/gpu/). That material -- including the 2xT4 throughput
comparison, P100/L4 compatibility nulls, and the T4 model probe -- is historical
evidence only. GPU kernels and metadata under the archive must not be
resubmitted. See [`archive/gpu/RESULTS.md`](archive/gpu/RESULTS.md) for the
full historical record.

## CLI/account state (Kaggle)

- Kaggle CLI version: **2.2.3** (upgraded from 2.1.2 via `uv tool`).
- OAuth credentials at `~/.kaggle/credentials.json` (mode `600`); no credential
  value is printed or copied.
- CPU-only jobs do not consume GPU or TPU quota.

## CLI/account state (Colab)

- Colab CLI version: **0.6.0** (installed via pip).
- OAuth authenticated via `colab login`.
- Standard free-tier CPU account (no Pro/Pro+).
- CPU sessions report ``cpu_count=2`` in the runtime environment.

## Exact active submission commands (Kaggle)

```bash
# Cortex CPU smoke
kaggle kernels push \
  -p infrastructure/kaggle/cpu-smoke \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-cortex-cpu-smoke
kaggle kernels output \
  abdellahkadem/oczy-cortex-cpu-smoke \
  --path reports/kaggle/cpu-smoke \
  --force

# Qwen CPU probe
kaggle kernels push \
  -p infrastructure/kaggle/qwen-cpu-probe \
  --timeout 900

kaggle kernels status abdellahkadem/oczy-qwen-cpu-probe
kaggle kernels output \
  abdellahkadem/oczy-qwen-cpu-probe \
  --path reports/kaggle/qwen-cpu-probe \
  --force
```

No `--accelerator` flag is used. CPU kernels set `enable_gpu: false` in their
metadata; the generated bootstrap also sets `CUDA_VISIBLE_DEVICES=""` before
importing torch.

## Exact scheduler commands (mixed batch)

```bash
# Run a mixed Kaggle/Colab batch (v2 manifest)
uv run python infrastructure/kaggle/parallel_scheduler.py run \
  infrastructure/kaggle/my-mixed-batch.json \
  --state /tmp/parallel-state.json \
  --kaggle-max 4 \
  --colab-max 4 \
  --colab-cooldown 60

# Status check
uv run python infrastructure/kaggle/parallel_scheduler.py status \
  infrastructure/kaggle/my-mixed-batch.json \
  --state /tmp/parallel-state.json
```
