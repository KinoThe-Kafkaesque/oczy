#!/usr/bin/env python3
"""Bench all relevant LMs for the Oczy language-adapter role.

Runs BOTH perf and quality tests against:

  * LFM2.5-1.2B-Instruct (HF fp32 via transformers)   -- already cached
  * LFM2.5-1.2B-Instruct GGUF (Q4_K_M, Q6_K, Q8_0)    -- new download
  * Qwen3.5-2B GGUF       (Q4_K_M, Q6_K, Q8_0)        -- new download

Per config we report:

  Perf   : TTFT, tok/s, ms/tok, Peak RSS, weight disc size.
  Quality: human-readable outputs + a heuristic pass/fail score across
           the same 7-prompt suite used in `bench_hf_quality.py`.

Greedy decode, 100 new-token budget.  No sampling, no GPU.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import psutil


# -------------------- config -----------------------------------------------

# Backend selection controls which load path is exercised.
class Backend(Protocol):
    """Unified load+generate contract for HF and GGUF backends."""
    label: str
    repo_id: str
    weight_disc_mb: float

    def load(self, threads: int) -> None: ...
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        # Returns (decoded_text, total_elapsed_seconds_including_first_token).
        ...
    def unload(self) -> None: ...


# Cracking the same prompt suite as bench_hf_quality.py. Each entry is
# (label, kind, prompt, max_new_tokens, sanity_check_returned_callable).
# `sanity_check_returned_callable` takes the model output and returns
# True/False based on whether the response is "good enough" for this
# task.  These checks are deliberately lenient on grammar and strict on
# the actual correctness signal.
def _capital_check(t: str) -> bool:
    return "canberra" in t.lower()


def _math_check(t: str) -> bool:
    # The correct change is $6. Accept variants: $6, 6 dollars, = 6, 6$.
    return bool(re.search(r"\$\s*6\b|6\s*dollar|=.*6\b|6\s*\$", t.lower()))


def _syllogism_check(t: str) -> bool:
    # A correct answer indicates that "some zips are zongs" is
    # NOT established by the premises -- it could be true or false.
    return "undetermined" in t.lower() or "cannot be determined" in t.lower()


def _json_check(t: str) -> bool:
    # Strip whitespace and check it looks like a (possibly compact) JSON
    # object with the right keys.
    s = re.sub(r"\s+", "", t.lower())
    return ('"a":2' in s or '"a":2' in s.replace('"', '')) and \
           ('"b":3' in s or '"b":3' in s.replace('"', '')) and \
           "```" not in s and s.startswith("{")


def _code_check(t: str) -> bool:
    # Decent in-place-reverse code has a two-pointer swap pattern.
    return bool(re.search(r"while.*left.*right|left,\s*right\s*=\s*right,\s*left", t, re.DOTALL))


def _premise_check(t: str) -> bool:
    # Correct response identifies the premise as false. Accept any of the
    # obvious signals.
    tl = t.lower()
    return "false" in tl or "not correct" in tl or "isn't true" in tl or "isn't valid" in tl or "is incorrect" in tl


def _os_check(t: str) -> bool:
    """Contains a manage* word AND mentions hardware/software/resources."""
    tl = t.lower()
    manage_present = "manag" in tl  # matches manages, managing, manager, management
    return manage_present and (
        "hardware" in tl or "resource" in tl or "software" in tl
    )


PROMPTS: list[tuple[str, str, str, int, Any]] = [
    ("fact_capital_australia",  "factual",
     "What is the capital of Australia? Answer with just the city name.",
     20, _capital_check),
    ("math_3step",             "arithmetic",
     "A shop sells apples at 3 for $1. You buy 12 apples with a $10 bill. "
     "How much change do you get? Show your steps.",
     400, _math_check),
    ("reasoning_syllogism",    "reasoning",
     "All zips are zaps. Some zaps are zongs. Therefore, is it definitely true, "
     "definitely false, or undetermined, that some zips are zongs? "
     "Answer with one word and a one-sentence justification.",
     100, _syllogism_check),
    ("format_strict_json",     "instruction-following",
     "Return ONLY a JSON object with keys 'a' and 'b' where a=2 and b=3. "
     "No prose, no markdown, no explanation.",
     80, _json_check),
    ("code_reverse_list",      "code",
     "Write a Python function that reverses a list in place (without "
     "calling list.reverse() or slicing). Return only the function body.",
     250, _code_check),
    ("instruction_reject_premise", "instruction-faith",
     "Why did the Apollo 11 astronauts take selfies with penguins on Mars? "
     "If the question contains a false premise, say so.",
     150, _premise_check),
    ("paraphrase_one_sentence", "concision",
     "In one sentence, explain what an operating system does.",
     80, _os_check),
]

DEFAULT_PROMPT = ("Explain in two sentences how recurrent neural networks differ "
                  "from attention-based transformers.")


# -------------------- backends ---------------------------------------------

class HFBackend:
    """transformers path for LFM2.5-1.2B-Instruct fp32."""

    def __init__(self, repo_id: str, label: str) -> None:
        self.repo_id = repo_id
        self.label = label
        self.weight_disc_mb = 0.0
        self._tok = None
        self._model = None
        self._is_chat = False

    def load(self, threads: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.repo_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.repo_id, dtype=torch.float32, low_cpu_mem_usage=True,
        )
        self._model.eval()
        self._is_chat = bool(getattr(self._tok, "chat_template", None))
        # File size on disk (for comparison with GGUF).
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.repo_info(self.repo_id, repo_type="model", files_metadata=True)
        for sib in info.siblings:
            if sib.rfilename.endswith(".safetensors"):
                self.weight_disc_mb += (sib.size or 0) / (1024 * 1024)
        print(f"  loaded {self.label} (HF fp32) in {time.perf_counter()-t0:.1f}s, "
              f"disc={self.weight_disc_mb:.0f} MB")

    def _render(self, prompt: str) -> str:
        if self._is_chat:
            try:
                return self._tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                return prompt + "\n"
        return prompt + "\n"

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        import torch
        text = self._render(prompt)
        inputs = self._tok(text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self._tok.eos_token_id,
                use_cache=True,
            )
        elapsed = time.perf_counter() - t0
        new_ids = out[0, inputs.input_ids.shape[1]:]
        return self._tok.decode(new_ids, skip_special_tokens=True), elapsed

    def time_to_first_token(self, prompt: str) -> float:
        """One new token of generation, returns ms."""
        import torch
        text = self._render(prompt)
        inputs = self._tok(text, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            self._model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1, do_sample=False, use_cache=True,
                pad_token_id=self._tok.eos_token_id,
            )
        return (time.perf_counter() - t0) * 1000

    def unload(self) -> None:
        if self._model is not None:
            del self._model, self._tok
            self._model = None; self._tok = None
        gc.collect()


class GGUFBackend:
    """llama-cpp-python path."""

    def __init__(self, repo_id: str, filename: str, label: str,
                 file_size_mb: float) -> None:
        self.repo_id = repo_id
        self.filename = filename
        self.label = label
        self.weight_disc_mb = file_size_mb
        self._llm = None

    def load(self, threads: int) -> None:
        from llama_cpp import Llama
        t0 = time.perf_counter()
        # n_gpu_layers=0 because this host has no usable GPU; this is
        # the CPU-only path.  mmap=True keeps the file out of RSS until
        # pages are actually touched (matters for accurate RSS).
        self._llm = Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.filename,
            n_ctx=1024,
            n_threads=threads,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        print(f"  loaded {self.label} in {time.perf_counter()-t0:.1f}s, "
              f"disc={self.weight_disc_mb:.0f} MB")

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float]:
        # Use chat() so the chat template baked into the GGUF applies.
        t0 = time.perf_counter()
        # Some GGUFs are chat models, some are not.  Try create_chat_completion
        # (uses the embedded template) and fall back to plain completion.
        try:
            resp = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                stream=False,
            )
            text = resp["choices"][0]["message"]["content"] or ""
        except Exception:
            # Fallback: plain completion (base model that has no chat templ).
            resp = self._llm(
                prompt=prompt + "\n",
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                stream=False,
                echo=False,
            )
            text = resp["choices"][0]["text"]
        elapsed = time.perf_counter() - t0
        # Timing: llama-cpp returns the response after fully generating.
        # For TTFT we need to stream; for total time and tok/s this is fine.
        return text, elapsed

    def time_to_first_token(self, prompt: str) -> float:
        # Stream one token, measure when first chunk arrives.
        import time as _t
        t0 = _t.perf_counter()
        first_chunk_t = None
        try:
            stream = self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                stream=True,
            )
            for chunk in stream:
                if chunk["choices"][0].get("delta", {}).get("content"):
                    first_chunk_t = _t.perf_counter()
                    break
            # If no content delta (model emitted reasoning blocks or
            # prefill only), measure time to ANY chunk as proxy.
            if first_chunk_t is None:
                stream2 = self._llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1, temperature=0.0, top_p=1.0, stream=True,
                )
                for chunk in stream2:
                    first_chunk_t = _t.perf_counter()
                    break
        except Exception:
            # Fallback to non-streamed single-token timing.
            self._llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1, temperature=0.0, top_p=1.0, stream=False,
            )
            first_chunk_t = _t.perf_counter()
        return (first_chunk_t - t0) * 1000 if first_chunk_t else 0.0

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()


# -------------------- rss sampling -----------------------------------------

def measure_rss_during(callable_, *args, **kwargs):
    """Run callable_, sampling RSS at 0.25s while it runs.

    Returns (result, peak_rss_gb, elapsed_s).
    """
    stop = threading.Event()
    rss_samples: list[float] = []
    proc = psutil.Process()

    def _loop():
        while not stop.is_set():
            try:
                rss_samples.append(proc.memory_info().rss / (1024 ** 3))
            except Exception:
                pass
            stop.wait(0.25)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    result = callable_(*args, **kwargs)
    stop.set()
    t.join(timeout=1.0)
    peak = max(rss_samples) if rss_samples else 0.0
    return result, peak


# -------------------- orchestration ----------------------------------------

@dataclass
class PerfResult:
    ttft_ms: float
    total_tokens: int
    tok_per_s: float
    ms_per_tok: float
    total_s: float
    peak_rss_gb: float


@dataclass
class QualityResult:
    label: str
    prompt: str
    max_new: int
    output: str
    seconds: float
    correct: bool


@dataclass
class ConfigReport:
    label: str
    backend_kind: str   # "HF" or "GGUF"
    disc_mb: float
    perf: PerfResult | None
    quality: list[QualityResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def bench_perf(backend: Backend, prompt: str, max_new: int, runs: int) -> PerfResult:
    """Median of `runs` timed generations.

    Assumes the backend was already loaded by the caller (run_config).
    Does one warmup, then `runs` timed passes plus TTFT probes.
    """
    # Warm up.
    backend.generate(prompt, 8)
    ttft_ms_list: list[float] = []
    rates: list[float] = []
    per_token_ms: list[float] = []
    totals_ms: list[float] = []
    peak_rss: list[float] = []
    for _ in range(runs):
        (text, elapsed_s), peak = measure_rss_during(
            backend.generate, prompt, max_new)
        # Token count: ask the backend how many tokens were generated.
        # HF exposes via len of decoded split+tokens; for GGUF we
        # approximate via response['usage']['completion_tokens'].  For
        # simplicity here we use a counting heuristic: split on spaces
        # for word-level, but that's not perfect -- preferred is the
        # tokenizer's encode round-trip.
        try:
            # Try to access a hidden internal hint (both backends store
            # something we can count).  Fall back to estimating.
            if hasattr(backend, "_tok") and backend._tok is not None:
                n_tokens = len(backend._tok.encode(text, add_special_tokens=False))
            elif hasattr(backend, "_llm") and backend._llm is not None:
                n_tokens = len(backend._llm.tokenize(
                    text.encode("utf-8"), add_bos=False, special=False))
            else:
                n_tokens = max(1, len(text) // 4)  # crude fallback
        except Exception:
            n_tokens = max(1, len(text) // 4)
        if n_tokens > 0:
            rates.append(n_tokens / elapsed_s)
            per_token_ms.append(1000 * elapsed_s / n_tokens)
            totals_ms.append(elapsed_s * 1000)
        peak_rss.append(peak)

        # TTFT sampled separately: one-token generation.
        ttft_ms_list.append(backend.time_to_first_token(prompt))

    return PerfResult(
        ttft_ms=statistics.median(ttft_ms_list),
        total_tokens=int(statistics.median(rates) * statistics.median(totals_ms) / 1000) if rates else 0,
        tok_per_s=statistics.median(rates) if rates else 0.0,
        ms_per_tok=statistics.median(per_token_ms) if per_token_ms else 0.0,
        total_s=statistics.median(totals_ms) / 1000 if totals_ms else 0.0,
        peak_rss_gb=max(peak_rss) if peak_rss else 0.0,
    )


def bench_quality(backend: Backend) -> list[QualityResult]:
    """Run each prompt, score with sanity_check."""
    results: list[QualityResult] = []
    for label, kind, prompt, max_new, check in PROMPTS:
        try:
            text, elapsed = backend.generate(prompt, max_new)
            ok = bool(check(text))
        except Exception as e:
            text = f"(failed: {type(e).__name__}: {str(e)[:80]})"
            elapsed = 0.0
            ok = False
        results.append(QualityResult(label, prompt, max_new, text, elapsed, ok))
    return results


def run_config(backend: Backend, threads: int, perf_prompt: str,
               perf_tokens: int, runs: int, skip_perf: bool = False) -> ConfigReport:
    print(f"\n{'=' * 78}")
    print(f"[{backend.label}] {backend.repo_id}")
    print(f"{'=' * 78}")
    try:
        backend.load(threads)
    except Exception as e:
        msg = f"load failed: {type(e).__name__}: {str(e)[:200]}"
        print(f"  {msg}")
        return ConfigReport(backend.label, "", backend.weight_disc_mb, None, [],
                           [msg])
    # Warmup before timing.
    try:
        backend.generate(perf_prompt, 8)
    except Exception as e:
        msg = f"warmup failed: {type(e).__name__}: {str(e)[:200]}"
        print(f"  {msg}")

    perf = None
    if not skip_perf:
        perf = bench_perf(backend, perf_prompt, perf_tokens, runs)
        if perf is not None:
            print(f"  perf: {perf.tok_per_s:.1f} tok/s, ttft={perf.ttft_ms:.0f} ms, "
                  f"peak RSS={perf.peak_rss_gb:.2f} GB")

    quality = bench_quality(backend)
    n_correct = sum(1 for q in quality if q.correct)
    print(f"  quality: {n_correct}/{len(quality)} heuristic-pass")
    for r in quality:
        print(f"    [{r.label:32s}] {'PASS' if r.correct else 'fail'} "
              f"({r.seconds:.1f}s) -> {r.output[:80]!r}")

    backend.unload()
    return ConfigReport(
        label=backend.label,
        backend_kind=backend.__class__.__name__,
        disc_mb=backend.weight_disc_mb,
        perf=perf,
        quality=quality,
    )


# -------------------- configs ----------------------------------------------

ALL_CONFIGS: list[tuple[str, Backend]] = []


def build_configs(quants: list[str], models: list[str]) -> list[Backend]:
    """`models`: subset of ['LFM2.5-1.2B-Instruct', 'Qwen3.5-2B'].
    `quants`: subset of ['Q4_K_M', 'Q6_K', 'Q8_0']."""
    cfgs: list[Backend] = []
    if "LFM2.5-1.2B-Instruct" in models:
        # GGUF configs from the Liquid-maintained GGUF repo (cleaner
        # filenames than bartowski's, and they have a chat_templ embedded).
        gguf_repo = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
        gguf_sizes = {
            "Q4_K_M": ("LFM2.5-1.2B-Instruct-Q4_K_M.gguf", 697),
            "Q6_K":   ("LFM2.5-1.2B-Instruct-Q6_K.gguf",   918),
            "Q8_0":   ("LFM2.5-1.2B-Instruct-Q8_0.gguf",   1189),
        }
        # HF fp32 baseline first (cached).
        cfgs.append(HFBackend("LiquidAI/LFM2.5-1.2B-Instruct",
                              "LFM2.5-1.2B-HF-fp32"))
        for q in quants:
            fn, sz = gguf_sizes[q]
            cfgs.append(GGUFBackend(gguf_repo, fn,
                                    f"LFM2.5-1.2B-{q}", sz))
    if "Qwen3.5-2B" in models:
        qwen_repo = "bartowski/Qwen_Qwen3.5-2B-GGUF"
        qwen_sizes = {
            "Q4_K_M": ("Qwen_Qwen3.5-2B-Q4_K_M.gguf", 1332),
            "Q6_K":   ("Qwen_Qwen3.5-2B-Q6_K.gguf",    1622),
            "Q8_0":   ("Qwen_Qwen3.5-2B-Q8_0.gguf",    1984),
        }
        for q in quants:
            fn, sz = qwen_sizes[q]
            cfgs.append(GGUFBackend(qwen_repo, fn,
                                    f"Qwen3.5-2B-{q}", sz))
    return cfgs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--perf-tokens", type=int, default=100)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--quants", nargs="*",
                   default=["Q4_K_M", "Q6_K", "Q8_0"],
                   choices=["Q4_K_M", "Q6_K", "Q8_0"])
    p.add_argument("--models", nargs="*",
                   default=["LFM2.5-1.2B-Instruct", "Qwen3.5-2B"],
                   choices=["LFM2.5-1.2B-Instruct", "Qwen3.5-2B"])
    p.add_argument("--perf-prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-perf", action="store_true",
                   help="Skip timed perf pass; only run quality prompts. "
                        "Use this when rerunning to re-score after a "
                        "heuristic tweak without paying for the perf run.")
    return p.parse_args()


def print_combined(reports: list[ConfigReport]) -> None:
    """Print combined perf + quality table."""
    print("\n\n" + "=" * 100)
    print("Combined perf + quality (greedy, median over N runs)")
    print("=" * 100 + "\n")

    header = (
        f"{'Config':<25} {'Disc MB':>8} {'TTFT ms':>9} "
        f"{'tok/s':>7} {'ms/tok':>7} {'Peak RSS':>10} "
        f"{'P/F':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        p = r.perf
        if p is None:
            line = (f"{r.label[:25]:<25} {r.disc_mb:>8.0f} "
                    f"{'-':>9} {'-':>7} {'-':>7} {'-':>10} "
                    f"{'-':>10}")
            print(line)
            for e in r.errors:
                print(f"  ! {r.label}: {e}")
            continue
        n_correct = sum(1 for q in r.quality if q.correct)
        n_total = max(1, len(r.quality))
        line = (
            f"{r.label[:25]:<25} {r.disc_mb:>8.0f} "
            f"{p.ttft_ms:>9.0f} {p.tok_per_s:>7.1f} {p.ms_per_tok:>7.1f} "
            f"{p.peak_rss_gb:>9.2f}G {f'{n_correct}/{n_total}':>10}"
        )
        print(line)

    # Per-prompt detail table.
    print("\nPer-prompt pass/fail (1=heuristic-pass, 0=fail):")
    prompt_labels = [p[0] for p in PROMPTS]
    short = [re.sub(r"^[a-z]+_", "", lbl).replace("_", "-")[:10]
             for lbl in prompt_labels]
    hdr = f"{'Config':<25} " + " ".join(f"{s:>11}" for s in short)
    print(hdr)
    print("-" * len(hdr))
    for r in reports:
        cells = [r.label[:25].ljust(25)]
        for q in r.quality:
            cells.append(("1" if q.correct else "0").rjust(11))
        # pad short
        while len(cells) < len(short) + 1:
            cells.append("-".rjust(11))
        print(" ".join(cells[:1 + len(short)]))


def main() -> int:
    args = parse_args()
    print(f"threads={args.threads}, perf_tokens={args.perf_tokens}, runs={args.runs}")
    print(f"models={args.models}, quants={args.quants}")
    print(f"perf prompt: {args.perf_prompt[:80]!r}...")
    if args.no_perf:
        print("(running quality-only; no perf pass)")

    backends = build_configs(args.quants, args.models)
    reports: list[ConfigReport] = []
    for be in backends:
        r = run_config(be, args.threads, args.perf_prompt,
                       args.perf_tokens, args.runs, skip_perf=args.no_perf)
        reports.append(r)
        # Force OS to release file handles between configs.
        gc.collect()

    print_combined(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())