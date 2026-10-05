# Decoder parity DEV v1 reproducibility archive

Completed September 13, 2026. Overlay `source/` on repository base
`e71268bc9cb01839995e5b3db69d0ad43a9ef04d`; place `instrument/` at
`experiments/decoder-parity-dev-v1`. The copied numeric bank is hash-bound,
so executing the materialized diagnostic does not require rerunning training.
Use its README command with a new output directory and the included selected
runtime provenance. Model weights and interpreter remain external dependencies
bound by `run/runtime_manifest.json`.

All 32 complete cases, 176 task outputs/scores, 208 decoded-string parity
verdicts including four-item batches, run command/log and independent audit
are included. The source overlay is a local hash-frozen uncommitted diagnostic,
not a remote clean-commit campaign. The candidate is separate from old organ
code. Runtime generation source and model generation configuration are copied
for the mechanism audit; they do not replace the pinned installed dependencies.

The copied report's relative links target its canonical repository location.
`SHA256SUMS` covers every archive file except itself. No model weights,
credentials, meta-test or unrelated job metadata are included.
