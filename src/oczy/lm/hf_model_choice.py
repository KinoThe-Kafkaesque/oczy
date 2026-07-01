"""Sprint 1 / S1.1 — the chosen HF substrate model.

Decision record: experiments_logs/2026-07-02_s1_1_model_selection.md
Benchmarked 2026-07-01/02 on this host (CPU, float32, 4 threads):
Qwen2.5-0.5B 82.8 ms/tok, Qwen2.5-1.5B 196.5 ms/tok, TinyLlama-1.1B
437 ms/tok (and EOS-after-1-token on plain prompts). All three passed the
plain-transformer KV-cache structure check; 0.5B wins on speed and RSS
with adequate instruction-following.
"""

HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MODEL_RATIONALE = (
    "Plain decoder-only transformer (per-layer (k,v) cache verified), "
    "82.8 ms/tok CPU float32 (2.4x faster than 1.5B), 2.7 GB RSS, "
    "instruct-tuned; fallback if quality-limited: Qwen/Qwen2.5-1.5B-Instruct."
)

# Fallback candidate, benchmarked and structurally verified but slower:
HF_MODEL_FALLBACK_ID = "Qwen/Qwen2.5-1.5B-Instruct"
