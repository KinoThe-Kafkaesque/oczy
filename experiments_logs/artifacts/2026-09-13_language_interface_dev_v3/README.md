# Language-interface DEV v3 reproducibility archive

Completed September 13, 2026. Overlay `source/` on repository base
`e71268bc9cb01839995e5b3db69d0ad43a9ef04d`; place `instrument/` at
`experiments/language-interface-dev-v3`. Use the run command in its README
with a new output directory and the included selected runtime provenance.
Model weights and the pinned interpreter remain external dependencies;
`run/runtime_manifest.json` binds their identity. All local execution sources
are hash-frozen uncommitted overlays, not a remote clean-commit campaign.

All 224 scored responses, full prompts, 16 live decoder comparisons,
execution command/log, model/runtime identities and independent audit are
included. The parent DEV-v2 run is needed only to rerun the paired historical
baseline analysis, not to execute these already-materialized prompts.
Relative links in the copied analytical report target its canonical location
in `experiments_logs/`.

`SHA256SUMS` covers every archive file except itself. No model weights,
credentials or unrelated job metadata are copied. No optimizer, meta-test,
remote launch or historical score change occurred.
