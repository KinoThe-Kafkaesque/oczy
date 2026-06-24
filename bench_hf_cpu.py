#!/usr/bin/env python3
"""Benchmark three small LMs on this machine.

For each model:
  1. Download via HF hub (cached at ~/.cache/huggingface/hub).
  2. Load on CPU with float32 (no GPU on this host).
  3. Tokenize a fixed prompt and run a warmup generate().
  4. Time N timed runs of greedy auto-regression for `--gen-tokens` new tokens.
  5. Capture:
     - prompt-encode latency (ms)
     - first-token latency (TTFT, ms) for the first generated token
     - per-token decode latency (ms/tok) for the rest
     - total wallclock (ms)
     - peak RSS (GB) during the timed run
     - a short generation sample.
  6. Print a comparison table at the end.

Run:
  .venv/bin/python $(pwd)/bench_hf_cpu.py --gen-tokens 100 --runs 3
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set CPU threading early so the whole run uses the same layout.  On
# the i7-1260P, torch defaults to 16 threads (all hyperthreads incl.
# E-cores) which has been measured to slow CPU-bound LM decode; the
# physical P-core count (4) is usually the sweet spot and gives
# reproducible numbers.
DEFAULT_THREADS = 4


MODELS: list[tuple[str, str]] = [
    ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
    ("LFM2.5-350M", "LiquidAI/LFM2.5-350M"),
    ("LFM2.5-1.2B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct"),
]

# A short but non-trivial prompt.  Two sentences is enough to encode
# but short enough that prompt encoding isn't the dominant term.
DEFAULT_PROMPT = "Explain in two sentences how recurrent neural networks differ from attention-based transformers."

# RSS sampled at this frequency during the timed generate() call.
SAMPLE_INTERVAL_S = 0.25


@dataclass
class RunResult:
    label: str
    repo_id: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    param_count_M: float
    weight_disk_mb: float
    prompt_tokens: int
    encode_ms: float
    ttft_ms: float
    per_token_ms: float
    total_ms: float
    tokens_per_sec: float
    peak_rss_gb: float
    load_ms: float
    runs: int
    errors: list[str] = field(default_factory=list)
    sample_text: str = ""


def human_count(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def model_shape(model) -> dict[str, Any]:
    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "num_layers", 0)) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "vocab_size": vocab,
        "hidden_dim": hidden,
        "num_layers": n_layers,
        "param_count": n_params,
    }


def measure_rss_thread() -> tuple[list[float], threading.Thread, threading.Event]:
    """Spawn a thread that samples current-process RSS at fixed interval.

    Returns a list (mutable) the thread writes to, the thread object, and
    a stop event.  Caller is responsible for setting the stop event and
    joining.
    """
    rss_samples: list[float] = []
    stop = threading.Event()
    proc = psutil.Process()

    def _loop():
        while not stop.is_set():
            try:
                rss_samples.append(proc.memory_info().rss / (1024 ** 3))
            except Exception:
                pass
            stop.wait(SAMPLE_INTERVAL_S)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return rss_samples, t, stop


def time_one_generate(model, tokenizer, input_ids, attention_mask, gen_tokens: int) -> tuple[float, int, str]:
    """Run one model.generate pass.  Returns (wall_sec, n_out_tokens, decoded)."""
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0
    new_tokens = out.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    return elapsed, new_tokens, text


def bench_one(label: str, repo_id: str, prompt: str, gen_tokens: int, runs: int) -> RunResult:
    import time as _time

    print(f"\n{'=' * 70}\n[{label}] {repo_id}\n{'=' * 70}")
    errors: list[str] = []

    # --- Download + load --------------------------------------------------
    t_load_start = _time.perf_counter()
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.eval()
    except Exception as e:
        msg = f"load failed: {type(e).__name__}: {str(e)[:200]}"
        print(f"  LOAD FAILED: {msg}")
        errors.append(msg)
        return RunResult(
            label=label, repo_id=repo_id, vocab_size=0, hidden_dim=0,
            num_layers=0, param_count_M=0, weight_disk_mb=0,
            prompt_tokens=0, encode_ms=0, ttft_ms=0, per_token_ms=0,
            total_ms=0, tokens_per_sec=0, peak_rss_gb=0, load_ms=0,
            runs=0, errors=errors,
        )
    load_ms = (_time.perf_counter() - t_load_start) * 1000

    shape = model_shape(model)
    print(f"  loaded: {human_count(shape['param_count'])} params, "
          f"hidden={shape['hidden_dim']}, layers={shape['num_layers']}, "
          f"vocab={shape['vocab_size']}  ({load_ms/1000:.1f}s incl download)")

    # Weight-file size on disk: walk the safetensors cache.  We don't
    # have a clean API for this so we look it up by hash in the HF cache.
    weight_mb = 0.0
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
        for sibling in info.siblings:
            if sibling.rfilename.endswith(".safetensors"):
                weight_mb += (sibling.size or 0) / (1024 * 1024)
        if weight_mb == 0:
            # Fallback: model params * 4 bytes (fp32) approx
            weight_mb = shape["param_count"] * 4 / (1024 * 1024)
        print(f"  weight files on disk: {weight_mb:.0f} MB")
    except Exception as e:
        print(f"  (could not get file sizes: {e})")
        weight_mb = shape["param_count"] * 4 / (1024 * 1024)

    # --- Prompt encode timing ---------------------------------------------
    t0 = _time.perf_counter()
    inputs = tok(prompt, return_tensors="pt")
    encode_ms = (_time.perf_counter() - t0) * 1000
    n_prompt = int(inputs.input_ids.shape[1])
    print(f"  prompt: {n_prompt} tokens ({prompt[:60]!r}...)  encode={encode_ms:.1f}ms")

    # --- Warmup (model + torch caches, JIT-tune MKL) ----------------------
    print(f"  warmup...")
    try:
        _, nt, _ = time_one_generate(model, tok, inputs.input_ids, inputs.attention_mask, 8)
        print(f"    warmup ok ({nt} tokens)")
    except Exception as e:
        msg = f"warmup failed: {type(e).__name__}: {str(e)[:200]}"
        print(f"    {msg}")
        errors.append(msg)
        # We still proceed with the timed runs to capture partial data.

    # --- Timed runs -------------------------------------------------------
    latencies_s: list[float] = []
    rates_tokps: list[float] = []
    per_token_ms_runs: list[float] = []
    ttft_ms_runs: list[float] = []
    peak_rss_samples: list[float] = []

    for r in range(runs):
        rss, t_rss, stop = measure_rss_thread()
        try:
            elapsed_s, n_out, decoded = time_one_generate(
                model, tok, inputs.input_ids, inputs.attention_mask, gen_tokens
            )
        except Exception as e:
            msg = f"run {r} failed: {type(e).__name__}: {str(e)[:200]}"
            print(f"    {msg}")
            errors.append(msg)
            stop.set(); t_rss.join(timeout=0.5)
            continue
        stop.set(); t_rss.join(timeout=1.0)

        # Try to extract first-token latency via a separate measurement.
        # Easiest proxy: re-run a single-token generation.
        t0 = _time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        ttft_ms = (_time.perf_counter() - t0) * 1000

        latencies_s.append(elapsed_s)
        if n_out > 0:
            rates_tokps.append(n_out / elapsed_s)
            per_token_ms_runs.append(1000 * elapsed_s / n_out)
        ttft_ms_runs.append(ttft_ms)
        peak_rss_samples.append(max(rss) if rss else 0.0)
        print(f"    run {r+1}: {elapsed_s:.2f}s, {n_out} tok, "
              f"{n_out/elapsed_s:.1f} tok/s, "
              f"peak RSS {max(rss) if rss else 0:.2f} GB")

    if not rates_tokps:
        return RunResult(
            label=label, repo_id=repo_id, vocab_size=shape["vocab_size"],
            hidden_dim=shape["hidden_dim"], num_layers=shape["num_layers"],
            param_count_M=shape["param_count"] / 1e6, weight_disk_mb=weight_mb,
            prompt_tokens=n_prompt, encode_ms=encode_ms,
            ttft_ms=0, per_token_ms=0, total_ms=0, tokens_per_sec=0,
            peak_rss_gb=0, load_ms=load_ms, runs=0, errors=errors,
            sample_text=decoded if 'decoded' in locals() else "",
        )

    # Use median across runs as the representative.
    per_token_ms = statistics.median(per_token_ms_runs)
    ttft_ms = statistics.median(ttft_ms_runs)
    total_ms = statistics.median(latencies_s) * 1000
    tokens_per_sec = statistics.median(rates_tokps)
    peak_rss = max(peak_rss_samples)

    res = RunResult(
        label=label, repo_id=repo_id, vocab_size=shape["vocab_size"],
        hidden_dim=shape["hidden_dim"], num_layers=shape["num_layers"],
        param_count_M=shape["param_count"] / 1e6, weight_disk_mb=weight_mb,
        prompt_tokens=n_prompt, encode_ms=encode_ms,
        ttft_ms=ttft_ms, per_token_ms=per_token_ms,
        total_ms=total_ms, tokens_per_sec=tokens_per_sec,
        peak_rss_gb=peak_rss, load_ms=load_ms, runs=len(rates_tokps),
        errors=errors, sample_text=decoded[:120],
    )

    # --- Release ----------------------------------------------------------
    del model, tok, inputs
    gc.collect()
    try:
        import torch as _torch
        _torch.cuda.empty_cache() if _torch.cuda.is_available() else None
    except Exception:
        pass

    return res


def print_summary(results: list[RunResult], gen_tokens: int) -> None:
    print(f"\n{'=' * 70}\nSummary (median over N runs, greedy decode, "
          f"{gen_tokens} new tokens)\n{'=' * 70}\n")

    cols = [
        ("Model",        24),
        ("Params",        9),
        ("Wt MB",         8),
        ("Prompt tok",   10),
        ("Enc ms",        8),
        ("TTFT ms",       9),
        ("tok/s",         8),
        ("ms/tok",        8),
        ("Total s",       9),
        ("Peak RSS GB",  12),
    ]
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row = "  ".join(
            [
                r.label[:24].ljust(24),
                f"{r.param_count_M:.0f}M".ljust(9),
                f"{r.weight_disk_mb:.0f}".ljust(8),
                str(r.prompt_tokens).ljust(10),
                f"{r.encode_ms:.1f}".ljust(8),
                f"{r.ttft_ms:.1f}".ljust(9),
                f"{r.tokens_per_sec:.2f}".ljust(8),
                f"{r.per_token_ms:.2f}".ljust(8),
                f"{r.total_ms/1000:.2f}".ljust(9),
                f"{r.peak_rss_gb:.2f}".ljust(12),
            ]
        )
        print(row)
        for e in r.errors:
            print(f"  ! {r.label}: {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--gen-tokens", type=int, default=100)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--models", nargs="*", default=None,
                   help="Optional subset of model labels to run.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    print(f"threads={torch.get_num_threads()}, device=cpu, "
          f"gen_tokens={args.gen_tokens}, runs={args.runs}")
    print(f"prompt: {args.prompt[:80]!r}...")

    models_to_run = MODELS
    if args.models:
        models_to_run = [(l, r) for l, r in MODELS if l in args.models]
        if not models_to_run:
            print(f"No matching models for --models {args.models}; "
                  f"available labels: {[l for l, _ in MODELS]}")
            return 1

    results: list[RunResult] = []
    for label, repo in models_to_run:
        r = bench_one(label, repo, args.prompt, args.gen_tokens, args.runs)
        results.append(r)
        # Tidy between models.
        gc.collect()

    print_summary(results, args.gen_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())