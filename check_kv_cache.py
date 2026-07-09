#!/usr/bin/env python3
"""KV-cache structure check for S1.1 model selection.

Verifies each candidate is a PLAIN decoder-only transformer:
- No hybrid conv/SSM layers
- No sliding-window-only attention
- Every layer yields a (k, v) tensor pair in past_key_values

Run: uv run python check_kv_cache.py
"""
from __future__ import annotations

import sys
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

CANDIDATES: list[tuple[str, str]] = [
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]

PROMPT = "The capital of France is"


def check_cache_structure(past_key_values: Any, n_layers: int, head_dim: int) -> dict:
    """Verify every layer yields a (k, v) tensor pair of expected shape."""
    errors = []

    if not isinstance(past_key_values, DynamicCache):
        # Legacy tuple-of-tuples format
        if not isinstance(past_key_values, tuple):
            return {"pass": False, "errors": [f"unexpected cache type: {type(past_key_values)}"]}
        pkv = past_key_values

    # DynamicCache has key_cache and value_cache lists
    if hasattr(past_key_values, "key_cache"):
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
    else:
        # Tuple of (k, v) pairs
        key_cache = [kv[0] for kv in past_key_values]
        value_cache = [kv[1] for kv in past_key_values]

    n_found = len(key_cache)
    if n_found != n_layers:
        errors.append(f"expected {n_layers} layers in cache, found {n_found}")

    for i in range(min(n_found, n_layers)):
        k = key_cache[i]
        v = value_cache[i] if i < len(value_cache) else None

        if k is None or v is None:
            errors.append(f"layer {i}: missing k or v (k={k is not None}, v={v is not None})")
            continue

        if k.ndim < 4:
            errors.append(f"layer {i}: key tensor has {k.ndim} dims, expected >= 4")
            continue

        # k.shape = (batch, num_heads, seq_len, head_dim)
        actual_head_dim = k.shape[3]
        if actual_head_dim != head_dim:
            errors.append(
                f"layer {i}: key head_dim={actual_head_dim}, expected {head_dim}"
            )

        if v.shape[3] != head_dim:
            errors.append(
                f"layer {i}: value head_dim={v.shape[3]}, expected {head_dim}"
            )

    return {"pass": len(errors) == 0, "errors": errors, "n_layers_found": n_found}


def check_architecture(config: Any) -> dict:
    """Inspect model config for hybrid/SSM/sliding-window red flags."""
    flags: list[str] = []
    warnings: list[str] = []

    arch = str(getattr(config, "architectures", ["unknown"])[0]).lower()

    # Red flags
    for kw in ["ssm", "mamba", "state_space", "hybrid", "liquid", "lfm"]:
        if kw in arch:
            flags.append(f"ARCHITECTURE RED FLAG: '{arch}' contains '{kw}'")

    has_sliding = getattr(config, "sliding_window", None)
    use_sliding = getattr(config, "use_sliding_window", False)
    if has_sliding is not None:
        warnings.append(f"sliding_window={has_sliding}")
    if use_sliding:
        flags.append("use_sliding_window=True — may be sliding-window-only")

    # Check for standard transformer indicators
    has_num_hidden = hasattr(config, "num_hidden_layers")
    has_num_attn = hasattr(config, "num_attention_heads")
    has_hidden = hasattr(config, "hidden_size")

    return {
        "architecture": arch,
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "head_dim": getattr(config, "head_dim", None),
        "has_standard_transformer_attrs": all([has_num_hidden, has_num_attn, has_hidden]),
        "red_flags": flags,
        "warnings": warnings,
        "pass": len(flags) == 0,
    }


def main() -> int:
    torch.set_num_threads(4)

    results: list[dict] = []

    for label, repo_id in CANDIDATES:
        print(f"\n{'='*60}\n[{label}] {repo_id}\n{'='*60}")

        try:
            tok = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            results.append({"label": label, "repo_id": repo_id, "error": str(e)})
            continue

        # --- Architecture check ---
        arch_result = check_architecture(model.config)
        print(f"  Architecture: {arch_result['architecture']}")
        print(f"  Layers: {arch_result['num_hidden_layers']}, "
              f"Heads: {arch_result['num_attention_heads']}, "
              f"Hidden: {arch_result['hidden_size']}")
        if arch_result["red_flags"]:
            for f in arch_result["red_flags"]:
                print(f"  RED FLAG: {f}")
        if arch_result["warnings"]:
            for w in arch_result["warnings"]:
                print(f"  WARNING: {w}")
        print(f"  Arch check: {'PASS' if arch_result['pass'] else 'FAIL'}")

        # --- KV-cache structure check ---
        inputs = tok(PROMPT, return_tensors="pt")
        head_dim = arch_result["head_dim"]
        if head_dim is None:
            n_heads = arch_result["num_attention_heads"] or 1
            h_size = arch_result["hidden_size"] or 0
            head_dim = h_size // n_heads if n_heads > 0 else 0
            print(f"  (inferred head_dim={head_dim} from hidden/{n_heads})")

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        cache_result = check_cache_structure(
            out.past_key_values,
            arch_result["num_hidden_layers"] or 0,
            head_dim,
        )
        print(f"  KV-cache layers: {cache_result['n_layers_found']}")
        if cache_result["errors"]:
            for e in cache_result["errors"]:
                print(f"  KV ERROR: {e}")
        print(f"  KV-cache check: {'PASS' if cache_result['pass'] else 'FAIL'}")

        results.append({
            "label": label,
            "repo_id": repo_id,
            "arch_check": arch_result,
            "cache_check": cache_result,
            "overall_pass": arch_result["pass"] and cache_result["pass"],
        })

        del model, tok
        import gc
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    all_pass = True
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: LOAD FAILED — {r['error']}")
            all_pass = False
        else:
            status = "PASS" if r["overall_pass"] else "FAIL"
            print(f"  {r['label']}: {status}")
            if not r["overall_pass"]:
                all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
