"""Offline training script for the tiny NumPy LM PlasticCortex.

This script fits a CharTokenizer on a plain-text corpus and trains an
LMPlasticCortex with SGD, printing per-epoch average loss. After training it
saves the tokenizer and the model and runs a small smoke-test generation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from plastic_cortex.char_tokenizer import CharTokenizer
from plastic_cortex.lm_cortex import LMPlasticCortex


def _resolve_default(path_str: str) -> Path:
    """Return a usable path whether running from the repo root or package root."""
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    # Running from inside the package directory (e.g. cd plastic-cortex): strip
    # the leading "plastic-cortex" segment from repo-root relative defaults.
    package_root = Path(__file__).resolve().parent.parent
    parts = candidate.parts
    if parts and parts[0] == package_root.name:
        return package_root.joinpath(*parts[1:])
    return candidate


def load_corpus(path: Path) -> list[str]:
    """Load non-empty lines from the corpus text file."""
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    return [line for line in lines if line]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tiny NumPy LM on a character-level corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("plastic-cortex/data/default_corpus.txt"),
        help="Path to the plain-text corpus (default: plastic-cortex/data/default_corpus.txt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="SGD learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Recurrent hidden dimension (default: 128).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plastic-cortex/checkpoints/lm"),
        help="Directory to write tokenizer.json and model.pkl (default: plastic-cortex/checkpoints/lm).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    corpus_path = _resolve_default(str(args.corpus))
    lines = load_corpus(corpus_path)
    if not lines:
        print(f"Corpus is empty: {corpus_path}", file=sys.stderr)
        return 1

    outdir = _resolve_default(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Fit tokenizer on the full corpus and persist it.
    tokenizer = CharTokenizer()
    tokenizer.fit(lines)
    tokenizer.save(outdir / "tokenizer.json")
    print(f"Corpus: {corpus_path} ({len(lines)} lines, vocab_size={tokenizer.vocab_size})")

    # Initialize the model with the tokenizer's vocabulary size.
    config = {
        "hidden_dim": args.hidden_dim,
        "vocab_size": tokenizer.vocab_size,
        "seed": 42,
    }
    model = LMPlasticCortex(config)
    # Wire the fitted tokenizer into the model so generation uses the same
    # vocabulary the model was trained on.
    model.tokenizer = tokenizer

    best_loss = float("inf")
    patience = 0
    best_path = outdir / "model_best.pkl"

    for epoch in range(1, args.epochs + 1):
        random.shuffle(lines)
        epoch_loss = 0.0
        for line in lines:
            model.reset_state()
            epoch_loss += model.train_step(line, lr=args.lr)
        avg_loss = epoch_loss / len(lines)
        print(f"Epoch {epoch:03d}/{args.epochs}: avg_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            model.save(best_path)
        else:
            patience += 1
            if patience >= 5:
                print(f"Early stopping at epoch {epoch} (no improvement for 5 epochs).")
                break

    final_path = outdir / "model.pkl"
    if best_path.exists():
        final_path.write_bytes(best_path.read_bytes())
    else:
        model.save(final_path)
    print(f"Saved model to {final_path}")

    # Smoke test: generate a response for "hello".
    model.reset_state()
    prompt = "hello"
    response = model.answer(prompt, max_tokens=100, temperature=0.8)
    print(f"\nSmoke test -- prompt: {prompt!r}")
    print(f"Generated: {response!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
