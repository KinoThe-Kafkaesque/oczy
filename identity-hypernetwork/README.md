# Identity Hypernetwork

Compact latent identity vectors (user, domain, style, mistakes) generated into LoRA/adapters by a hypernetwork.

## Quick start

```bash
uv sync
uv run pytest
```

## Project position in the stack

```text
activation / context

fast state

fast weights

neural memory

slow weights / priors
```

See `../experiments.txt` for the full architecture thesis.
