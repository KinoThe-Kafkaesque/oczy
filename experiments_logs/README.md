# Experiment Logs

This directory holds dated records of experiments, evaluations, and design decisions for the Oczy project.

Each log should be timestamped and include:
- hypothesis or goal
- method
- results
- conclusion / next steps

## Curated campaign logs

When a batch of experiments is executed together as a remote campaign (multiple
jobs across kaggle/colab providers from a shared source commit), a single
curated campaign log consolidates the evidence. The filename format is
`YYYY-MM-DD_campaign_<short-commit>.md`.

A curated campaign log must include:
- **Goal** — what the campaign set out to adjudicate
- **Immutable source commits and providers** — commit hashes, provider(s), CPU/GPU contract
- **Lawful scope** — statement that no metrics, thresholds, baselines, or manifests were modified
- **Concurrency** — batch sequence and execution model (kernels, notebooks, cross-batch concurrency)
- **Per-run results** — per-experiment outcome table with primary metrics, seeds, and provider
- **Seed distributions** — cross-seed variance data for multi-seed experiments (e.g. Exp06, R18)
- **Nulls and refutations** — preserved as prominently as positives; metricless NULL distinguished from scientific NULL
- **Infrastructure blockers** — experiments that did not run, with reason (not a scientific verdict)
- **Non-runnable inventory** — which catalogued experiments did not produce results
- **Artifact provenance paths** — report paths relative to the campaign working directory
- **Infrastructure fixes** — what broke and was fixed between retry batches
- **Next steps** — concrete follow-up actions

Do not copy transient raw logs (`/tmp/` execution summaries, `*.json` batch
files) into the curated log. Reference them by relative path. The curated log
is the durable record; the raw summaries are ephemeral.

The ledger (`LEDGER.md`) classifies each campaign log as a single VALID row
pointing to the curated file, replacing per-experiment row references.
