# Experiment Summary — 2026-06-19 08:01 UTC

Seed: 0 | Sense matching: enabled | Consolidation: enabled

| Agent | Uptake | Transfer | Scope | Forget | Consol | Identity | Mem/Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| ContextOnlyAgent | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 612.75 |
| FastOnlyAgent | 0.6667 | 0.1667 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 12.00 |
| HippocampusOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 8828.00 |
| IdentityOnlyAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 451.00 |
| OrganismAgent | 0.6667 | 0.2500 | 0.1667 | 1.0000 | 1.0000 | 1.0000 | 12477.25 |
| ZeroMemoryAgent | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 61.00 |

## Key observations

- FastOnlyAgent is currently the most memory-efficient learner (12 bytes/Δ) despite no hippocampus, hypernetwork, or immune system.
- OrganismAgent achieves slightly higher average performance (transfer=0.25) but pays ~1000x more memory per delta, so its composite behavior-delta-per-byte score collapses.
- HippocampusOnly and IdentityOnly baselines fail to transfer in the simple word-association domain, showing the standalone organs are not yet compressing lessons.
- The central bottleneck is consolidation: the experience autoencoder and identity hypernetwork are storing large serialized objects rather than compact adapters.
