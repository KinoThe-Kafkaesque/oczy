#!/usr/bin/env python3
"""OpenAI-compatible proxy for LFM2.5-1.2B with optional Oczy augmentation.

Serves two model IDs:
  - lfm-vanilla     : raw LFM2.5-1.2B
  - lfm-oczy        : LFM2.5-1.2B + Oczy knowledge store fact prepend

Usage:
    python proxy_server.py --port 8080 --model-path /path/to/model.gguf

Then register in Pi via ~/.pi/agent/models.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

class LFMEngine:
    """Thin wrapper around llama.cpp for local inference."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                verbose=False,
            )
        return self._llm

    def generate(self, messages: list[dict], max_tokens: int = 1024) -> str:
        """Convert messages to LFM2.5 instruct format and generate."""
        prompt = self._build_prompt(messages)
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|user|>", "<|assistant|>"],
        )
        return output["choices"][0]["text"].strip()

    @staticmethod
    def _build_prompt(messages: list[dict]) -> str:
        """Build LFM2.5 instruct prompt from chat messages."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Knowledge augmentation (Oczy)
# ---------------------------------------------------------------------------

class OczyAugmenter:
    """Prepends recalled Oczy codebase facts to the system message."""

    def __init__(self) -> None:
        self._store = None

    @property
    def store(self):
        if self._store is None:
            sys.path.insert(0, str(
                Path(__file__).resolve().parent.parent.parent / "src"
            ))
            from oczy.experiments.codebase_qa.knowledge_store import (
                KnowledgeStore,
            )
            self._store = KnowledgeStore(embed_fn=None)
            facts_path = (
                Path(__file__).resolve().parent.parent.parent
                / "src" / "oczy" / "experiments" / "codebase_qa" / "facts.json"
            )
            if facts_path.exists():
                with facts_path.open() as f:
                    for fact in json.load(f):
                        self._store.add_fact(
                            fact["key"], fact["value"], fact.get("metadata", {})
                        )
        return self._store

    def augment(self, messages: list[dict]) -> list[dict]:
        """Prepend recalled facts to the last user message."""
        # Find the last user message to use as recall query.
        user_texts = [
            m["content"] for m in messages if m.get("role") == "user"
        ]
        if not user_texts:
            return messages

        query = user_texts[-1]
        facts_block = self.store.format_context(query, k=3, min_score=0.0)
        if facts_block.strip() == "Retrieved repository facts:":
            return messages  # no facts found

        # Prepend facts as a system message.
        augmented = list(messages)
        augmented.insert(0, {"role": "system", "content": facts_block})
        return augmented


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(engine: LFMEngine, augmenter: OczyAugmenter | None = None) -> FastAPI:
    app = FastAPI(title="LFM2.5 Proxy")

    @app.get("/v1/models")
    async def list_models():
        models = [{"id": "lfm-vanilla", "object": "model"}]
        if augmenter is not None:
            models.append({"id": "lfm-oczy", "object": "model"})
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model", "lfm-vanilla")
        messages: list[dict] = body.get("messages", [])
        max_tokens: int = body.get("max_tokens", 1024)
        stream: bool = body.get("stream", False)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Apply Oczy augmentation if requested.
        if "oczy" in model and augmenter is not None:
            messages = augmenter.augment(messages)

        generated = engine.generate(messages, max_tokens=max_tokens)

        if stream:
            async def _stream():
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": generated},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(), media_type="text/event-stream")

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generated},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="LFM2.5 OpenAI-compatible proxy")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--model-path", required=True, help="Path to GGUF model")
    p.add_argument("--no-oczy", action="store_true", help="Disable Oczy augmentation")
    p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = p.parse_args()

    engine = LFMEngine(args.model_path)
    augmenter = None if args.no_oczy else OczyAugmenter()

    app = create_app(engine, augmenter)
    print(f"LFM2.5 proxy starting on http://{args.host}:{args.port}")
    print(f"  Models: lfm-vanilla" + (" lfm-oczy" if augmenter else ""))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
