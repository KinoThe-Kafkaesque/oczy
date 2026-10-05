# Workbench verification — September 14, 2026

This is local DEV verification in Africa/Casablanca. It is separate from the
scientific verdicts and does not promote a capability.

- **122 focused tests pass** in 4.48 seconds: capability and context contracts,
  summaries, language/persistence controls, prompt repairs, teaching diagnostics,
  new diversity and regression contracts, audit tamper rejection, eval guard,
  workbench boundaries and versioned generation behavior.
- `node --check workbench/static/app.js` passes after the responsive fix.
- Workbench verification checks six source reports, both new instrument
  manifests and their bound ancestors, and released state files.
- Independent `sha256sum --check --quiet SHA256SUMS` passes for the diversity
  archive (72 entries) and regression archive (63 entries).
- Actual CLI and browser model inference produce identical seven-condition
  outputs for amber/berry/seed 0, including whitespace and failed answers.
  The exact CLI object is [live-cli.json](live-cli.json). The worker verifies
  the pinned model/runtime/state identities and performs zero optimizer steps.
- Browser live submission clears stale results, disables inputs while running,
  and renders the returned outputs. Saved trials, seed/context controls and
  action transcript selection were exercised. HTTP unit tests use a fake
  engine; actual model evidence comes from the separate live checks.
- Browser document width equals scroll width at 375, 1265 and 1425 pixels.
  The mobile capability entries now keep each result and limitation together;
  desktop retains three columns.
- The finish reviewer accepted all seven supplied viewport-segment captures,
  then scored its one responsive-reading fix resolved with `disposition: ship`.
  This final verdict covers that fix, not an independent whole-surface pass.
  See [full review](finish-review.md) and [fix verdict](finish-verdict.md).

## Verification limits

Full-page browser and CDP captures failed; viewport segments were used and
visually checked. Captured image sizes are 1425×990 desktop, 375×812 mobile,
and 1265×712 user-width. Capture files contain JPEG bytes under `.png` names.
They are local review evidence, not packaged UI assets. The reviewer did not
independently operate the browser or assess uncaptured dynamic states.

The one detector invocation returned no findings in degraded regex mode:
HTML/CSS parser dependencies were unavailable, so this is not a computed
contrast or complete static-analysis result. Keyboard and accessibility have
not received a dedicated full audit. The workbench serves only loopback;
external hosting and production deployment are not part of this verification.

The diversity controller's post-evaluation bookkeeping failure and missing
execution metadata remain disclosed in the research report and interface.
Scientific archives preserve the original source and the separate recovery.

The final bundle is sealed with `PACKAGE_MANIFEST.json`. Unpacked-bundle live
verification is recorded next to the exported archive, avoiding a self-referential
archive hash. Model weights and Python runtime are configured separately.
