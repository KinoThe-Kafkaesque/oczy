# Identity Hypernetwork v1 Notes

## What is implemented

- `IdentityLatents`: four compact latent vectors (`z_user`, `z_domain`, `z_style`, `z_mistakes`) stored as NumPy arrays.
- `IdentityHypernetwork`: a tiny linear hypernetwork that maps the concatenated identity vector to a fixed concept vocabulary of adapter score deltas.
- A simple learning rule: a lesson updates the slice of the identity vector that corresponds to its `source`, moving it in the direction of the weight vector for the target concept so that future `generate_adapters()` scores shift toward the correction.

## Limitations

- **Fixed vocabulary**: `CONCEPT_VOCABULARY` is hard-coded. Any token outside the list cannot be directly learned (it falls back to the first known word in the label text).
- **No tokeniser / no sub-word handling**: real language is not a lookup table; this prototype handles only the small concept list.
- **Single linear projection**: output deltas are a linear map of the identity vector. There are no hidden layers, gates, or non-linear transformations.
- **No consolidation or forgetting**: lessons accumulate with no decay, replay, or anti-forgetting constraints.
- **Source-to-z mapping is hand-coded**: the agent does not learn which component should be updated; it uses keyword matching on `source`.
- **Small latent dimension**: 8-dim vectors are enough to demonstrate the idea but far too small for real identity modelling.
- **No scope control**: a correction could overgeneralise because the same concept scores are reused across all contexts.
- **No error signal**: the update always pushes the target score up; it does not compare predicted vs. corrected outputs.

## Next steps

1. Replace the fixed vocabulary with an embedding layer or a small frozen tokenizer so arbitrary tokens can become targets.
2. Add a non-linear hypernetwork head (still NumPy-only or later JAX/PyTorch) to produce richer adapter weights.
3. Introduce a proper loss and gradient update: compute `score_old(token)` vs. `score_target(correct_label)` and apply a small delta rule.
4. Split the concept space by domain/context to avoid overgeneralisation (``profile`` in project A vs. project B).
5. Add consolidation: periodically compress the identity latent and bound its magnitude.
6. Track lesson provenance and expose probes so the latent state can be inspected and rolled back.
