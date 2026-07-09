# Legacy substrate notice (Sprint 1 / S1.5)

**As of 2026-07-01, `cvec_driver.py` (`LlamaCVecDriver`, llama-cpp-python /
LFM2.5 GGUF) is a frozen legacy path.**

- Purpose: reproduce pre-Sprint-1 results only (experiments_logs entries up
  to 2026-07-01, lanes, the honest post-leakage baseline).
- Do not add features to it. Bug fixes only if they block reproduction.
- New work targets the HF/PyTorch substrate (`hf_driver.py`, Sprint 1 S1.2),
  which provides the two capabilities llama.cpp could not: direct
  `past_key_values` writes (Goal 1) and per-layer hidden access (Goal 2).
- Its tests stay in CI as regression cover for reproduction.

Rationale: see SPRINT.md ("Substrate" finding) — the llama.cpp binding
blocked Goal 1 and Goal 2, and a hybrid conv+attention model made KV-state
semantics ambiguous. Sprint 1 migrates to a plain small transformer via
HuggingFace.
