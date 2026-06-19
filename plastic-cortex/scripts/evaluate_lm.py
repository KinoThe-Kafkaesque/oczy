"""Quickly evaluate a saved LMPlasticCortex checkpoint.

Usage:
    uv run python plastic-cortex/scripts/evaluate_lm.py \
        --model plastic-cortex/checkpoints/lm/model.pkl \
        --corpus plastic-cortex/data/codex_corpus_2k.txt \
        --prompts "hello" "what is" "the"
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plastic_cortex.char_tokenizer import CharTokenizer
from plastic_cortex.lm_cortex import LMPlasticCortex


def load_corpus(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line and not line.startswith("# ")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PlasticCortex LM checkpoint.")
    parser.add_argument("--model", type=Path, required=True, help="Path to model.pkl")
    parser.add_argument("--corpus", type=Path, default=None, help="Corpus for loss evaluation")
    parser.add_argument("--prompts", nargs="+", default=["hello", "what", "the"], help="Generation prompts")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    model = LMPlasticCortex.load(args.model)
    print(f"Loaded model from {args.model}")
    print(f"Status: {model.status()}")

    if args.corpus and args.corpus.exists():
        print("\nEvaluating per-character cross-entropy on corpus...")
        lines = load_corpus(args.corpus)
        random.shuffle(lines)
        total_loss = 0.0
        total_chars = 0
        for line in lines:
            model.reset_state()
            loss = model.train_step(line, lr=0.0)
            total_loss += loss * len(line)
            total_chars += len(line)
        avg_loss = total_loss / max(1, total_chars)
        print(f"Corpus avg_loss: {avg_loss:.4f} (perplexity: {2.718 ** avg_loss:.2f})")

    print("\nGeneration samples:")
    for prompt in args.prompts:
        model.reset_state()
        response = model.answer(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        print(f"  {prompt!r} -> {response!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
