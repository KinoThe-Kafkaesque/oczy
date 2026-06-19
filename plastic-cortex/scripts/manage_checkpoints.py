#!/usr/bin/env python3
"""Checkpoint manager for the tiny NumPy LM training workflow.

Lists, promotes, and deletes checkpoint runs under ``plastic-cortex/checkpoints/``.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_DEFAULT_TARGET_DIR = _CHECKPOINTS_DIR / "lm"


@dataclass
class RunInfo:
    name: str
    path: Path
    loss: float | None
    epoch: int | None
    hidden_dim: int | None
    vocab_size: int | None
    tokenizer: str | None


def _find_files(root: Path, pattern: str) -> list[Path]:
    """Return all files under *root* matching the glob *pattern*."""
    results: list[Path] = []
    if root.exists():
        results = [p for p in root.rglob(pattern) if p.is_file()]
    return sorted(results)


def _discover_model(run_dir: Path) -> Path | None:
    """Return the best-annotated model.pkl, or the first one found under *run_dir*."""
    log_path = run_dir / "train.log"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Saved best model \(loss=[\d.]+\) to\s+(\S+)", text)
        if m:
            best = Path(m.group(1))
            if best.is_absolute() and best.exists():
                return best
            candidate = run_dir / best.name if best.parts else run_dir / best
            if candidate.exists():
                return candidate
    candidates = _find_files(run_dir, "model.pkl")
    return candidates[0] if candidates else None


def _discover_tokenizer(run_dir: Path, model_path: Path | None = None) -> Path | None:
    """Return the tokenizer JSON for *run_dir*, preferring one near *model_path*."""
    candidates = _find_files(run_dir, "*.json")
    if model_path:
        same_dir = sorted([p for p in candidates if p.parent == model_path.parent])
        if same_dir:
            return same_dir[0]
    for name in ("tokenizer.json", "tokenizer_*.json"):
        matches = _find_files(run_dir, name)
        if matches:
            return matches[0]
    return candidates[0] if candidates else None


def _parse_log(log_path: Path) -> dict[str, object]:
    """Parse train.log for loss, epoch, hidden_dim, vocab_size and tokenizer."""
    result: dict[str, object] = {
        "loss": None,
        "epoch": None,
        "hidden_dim": None,
        "vocab_size": None,
        "tokenizer": None,
    }

    text = log_path.read_text(encoding="utf-8", errors="replace")

    vocab_match = re.search(r"vocab_size=(?P<vocab>\d+)", text)
    if vocab_match:
        result["vocab_size"] = int(vocab_match.group("vocab"))

    tok_match = re.search(r"tokenizer=(?P<tok>[^,\s]+)", text)
    if tok_match:
        result["tokenizer"] = tok_match.group("tok")

    best_match = re.search(r"Saved best model \(loss=(?P<loss>[\d.]+)\)", text)
    if best_match:
        result["loss"] = float(best_match.group("loss"))

    last_epoch: int | None = None
    last_loss: float | None = None
    first_hidden: int | None = None
    for line in text.splitlines():
        m = re.search(
            r"Epoch\s+(?P<epoch>\d+):\s+hidden=(?P<hidden>\d+)\s+avg_loss=(?P<loss>[\d.]+)",
            line,
        )
        if m:
            last_epoch = int(m.group("epoch"))
            last_loss = float(m.group("loss"))
            if first_hidden is None:
                first_hidden = int(m.group("hidden"))

    result["hidden_dim"] = first_hidden
    if result["loss"] is None:
        result["loss"] = last_loss
    result["epoch"] = last_epoch

    return result


def find_checkpoints(base_dir: Path) -> list[RunInfo]:
    """Return a list of RunInfo objects for each checkpoint under *base_dir*."""
    runs: list[RunInfo] = []
    if not base_dir.exists():
        return runs

    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        train_log = entry / "train.log"
        if not train_log.is_file():
            continue
        if not _find_files(entry, "model.pkl"):
            continue

        parsed = _parse_log(train_log)
        runs.append(
            RunInfo(
                name=entry.name,
                path=entry,
                loss=parsed.get("loss") if parsed.get("loss") is not None else None,
                epoch=parsed.get("epoch"),
                hidden_dim=parsed.get("hidden_dim"),
                vocab_size=parsed.get("vocab_size"),
                tokenizer=parsed.get("tokenizer"),
            )
        )

    return sorted(runs, key=lambda r: (r.loss if r.loss is not None else float("inf")))


def _fmt(value: object | None, missing: str = "-") -> str:
    return missing if value is None else str(value)


def _fmt_loss(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def print_table(runs: Iterable[RunInfo]) -> None:
    """Print a formatted table of checkpoint runs sorted by loss."""
    rows = list(runs)
    headers = ["Run", "Loss", "Epoch", "Hidden", "Vocab", "Tokenizer"]
    data = [
        [
            r.name,
            _fmt_loss(r.loss),
            _fmt(r.epoch),
            _fmt(r.hidden_dim),
            _fmt(r.vocab_size),
            _fmt(r.tokenizer),
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in data:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for row in data:
        print(fmt_row(row))


def cmd_promote(run_name: str, base_dir: Path, target_dir: Path) -> int:
    """Copy a run's model.pkl and tokenizer.json to the default teach.py path."""
    run_dir = base_dir / run_name
    if not run_dir.is_dir():
        print(f"Run not found: {run_dir}", file=sys.stderr)
        return 1

    model_src = _discover_model(run_dir)
    if model_src is None:
        print(f"No model.pkl found for {run_name}", file=sys.stderr)
        return 1

    tokenizer_src = _discover_tokenizer(run_dir, model_src)
    if tokenizer_src is None:
        print(f"No tokenizer JSON found for {run_name}", file=sys.stderr)
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)
    model_dst = target_dir / "model.pkl"
    tokenizer_dst = target_dir / "tokenizer.json"

    shutil.copyfile(model_src, model_dst)
    shutil.copyfile(tokenizer_src, tokenizer_dst)

    print(f"Promoted {run_name}")
    print(f"  model:     {model_src} -> {model_dst}")
    print(f"  tokenizer: {tokenizer_src} -> {tokenizer_dst}")
    return 0


def cmd_delete(run_name: str, base_dir: Path) -> int:
    """Remove a checkpoint directory."""
    run_dir = base_dir / run_name
    if not run_dir.is_dir():
        print(f"Run not found: {run_dir}", file=sys.stderr)
        return 1

    print(f"Deleting checkpoint directory: {run_dir}")
    shutil.rmtree(run_dir)
    print(f"Deleted {run_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage plastic-cortex training checkpoints."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--promote",
        metavar="RUN_NAME",
        help="Copy RUN_NAME's model.pkl and tokenizer.json to checkpoints/lm/",
    )
    group.add_argument(
        "--delete",
        metavar="RUN_NAME",
        help="Remove RUN_NAME's checkpoint directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.promote:
        return cmd_promote(args.promote, _CHECKPOINTS_DIR, _DEFAULT_TARGET_DIR)
    if args.delete:
        return cmd_delete(args.delete, _CHECKPOINTS_DIR)

    runs = find_checkpoints(_CHECKPOINTS_DIR)
    if not runs:
        print(f"No checkpoints found in {_CHECKPOINTS_DIR}")
        return 0

    print_table(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
