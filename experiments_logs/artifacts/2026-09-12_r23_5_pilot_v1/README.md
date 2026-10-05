# R23.5 pilot v1 evidence archive

This is the immutable execution evidence for the September 12 DEV pilot.
See [the report](../../2026-09-12_campaign_r23_5_pilot_v1.md) and
[all scores/outputs](../../2026-09-12_r23_5_pilot_v1.json).

- `instrument/`: frozen manifest, teaching data, held-out probes and audit.
- `released_states/`: six initial and six trained numeric arrays plus their
  hash/shape/dtype manifest. NumPy loading must use `allow_pickle=False`.
- `released_text/`: unedited raw examples and model-written summaries plus
  their hash/byte/token manifest.
- `execution/`: four phase results/stdout logs, the complete optimization
  trajectory, reload rows and invocation/provenance records.
- `source/`: the source bytes bound by the frozen manifest, the preparer,
  namespace helper and report aggregator.
- `run_pilot.py`: archived orchestration; its absolute output path is already
  occupied and must not be reused for another run.

Execution was local, using uncommitted source based on main at
`e71268bc9cb01839995e5b3db69d0ad43a9ef04d` and exact source hashes.
This is not a clean commit-addressed remote execution. No model weights,
credentials or remote submission are included.

SHA256SUMS.json covers all archive files other than itself. The original
runtime/model are external; their verified identities are in the manifest
and execution records. A future rerun must preserve those identities and
the archived source bytes, use a new output directory, and retain every
attempt. The archive is evidence, not a new experiment authorization.
