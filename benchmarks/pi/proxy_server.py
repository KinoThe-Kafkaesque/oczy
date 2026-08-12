#!/usr/bin/env python3
"""OpenAI-compatible proxy for LFM2.5-1.2B with optional Oczy augmentation.

Serves two model IDs:
  - lfm-vanilla     : raw LFM2.5-1.2B
  - lfm-oczy        : LFM2.5-1.2B + Oczy knowledge store fact prepend

Supports OpenAI-style function calling (tools) via prompt injection:
tool definitions are appended to the system prompt and the model's
JSON output is parsed into tool_calls.

Usage:
    python proxy_server.py --port 8080 --model-path /path/to/model.gguf

Then register in Pi via ~/.pi/agent/models.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

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
                n_ctx=16384,
                n_threads=8,
                verbose=False,
            )
        return self._llm

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        tools: list[dict] | None = None,
    ) -> dict:
        """Generate a response. Returns dict with 'content' and/or 'tool_calls'.

        When *tools* are provided, tool definitions are injected into the
        system prompt and the model output is parsed for JSON tool calls.
        """
        prompt = self._build_prompt(messages, tools=tools)
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|im_start|>", "<|im_end|>"],
        )
        text = output["choices"][0]["text"].strip()

        if tools:
            tool_calls = self._parse_tool_calls(text)
            if tool_calls:
                return {"content": None, "tool_calls": tool_calls}

        return {"content": text, "tool_calls": None}

    # -- content normalization ---------------------------------------------

    @staticmethod
    def _coerce_content(content) -> str:
        """Normalize OpenAI message content (str or list-of-parts) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
            return " ".join(texts)
        return str(content)

    # -- tool injection & parsing -------------------------------------------

    @staticmethod
    def _format_tools(tools: list[dict]) -> str:
        """Format OpenAI tool definitions into a concise prompt instruction."""
        lines = [
            "\n\n--- TOOL USE ---",
            "You have access to the following tools. To call a tool, respond "
            "with ONLY a JSON object in this exact format and nothing else:",
            '{"name": "<tool_name>", "arguments": {<parameters>}}',
            "",
            "Example — to read a file:",
            '{"name": "read", "arguments": {"path": "pyproject.toml"}}',
            "",
            "Available tools:",
        ]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "").split(".")[0]
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            param_strs = []
            for pname, pinfo in params.items():
                ptype = pinfo.get("type", "any")
                req = "required" if pname in required else "optional"
                param_strs.append(f"  {pname} ({ptype}, {req})")
            lines.append(f"- {name}: {desc}")
            if param_strs:
                lines.append("  Parameters:")
                lines.extend(param_strs)
        lines.append("")
        lines.append(
            "After receiving a tool result, use it to answer the user's "
            "question. If no tool is needed, respond normally.",
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict] | None:
        """Extract tool calls from model output text.

        Supports two formats:
        1. JSON:  {"name": "read", "arguments": {"path": "foo.py"}}
        2. Bracket: [read(path="foo.py")]  (model's natural format)

        Returns a list of OpenAI-format tool_call dicts, or None.
        """
        # --- Strategy 1: JSON objects -------------------------------------
        candidates: list[str] = []
        # Pattern 1a: fenced code block
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            candidates.append(m.group(1))
        # Pattern 1b: raw JSON object (balanced brace matching)
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
                    start = -1

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            args = obj.get("arguments") or obj.get("args") or obj.get("parameters")
            if name and isinstance(args, dict):
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                return [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }]

        # --- Strategy 2: bracket format [name(arg="val", ...)] ------------
        bracket_match = re.match(
            r'\s*\[(\w+)\s*\((.*)\)\]', text, re.DOTALL,
        )
        if bracket_match:
            name = bracket_match.group(1)
            args_str = bracket_match.group(2)
            args: dict = {}
            # Parse key="value" or key=value pairs.
            for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                args[m.group(1)] = m.group(2)
            for m in re.finditer(r"(\w+)\s*=\s*'([^']*)'", args_str):
                args[m.group(1)] = m.group(2)
            # Also handle unquoted values (numbers, booleans).
            for m in re.finditer(r"(\w+)\s*=\s*([^,'\s]+)", args_str):
                key, val = m.group(1), m.group(2)
                if key not in args:
                    args[key] = val
            if name:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                return [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }]

        return None

    # -- prompt building ----------------------------------------------------

    @classmethod
    def _build_prompt(
        cls,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> str:
        """Build LFM2.5 instruct prompt from chat messages.

        Handles system, user, assistant (including tool_calls), and tool
        (result) roles.  When *tools* are provided, a tool-calling
        instruction is appended to the first system message.
        """
        parts: list[str] = []
        tool_instruction = cls._format_tools(tools) if tools else None
        tool_names = [t["function"]["name"] for t in tools] if tools else []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = cls._coerce_content(msg.get("content", ""))

            if role == "system":
                if tool_instruction and idx == 0:
                    content = content + tool_instruction
                parts.append(f"<|im_start|>system\n{content}")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}")
            elif role == "assistant":
                # If the assistant message has tool_calls, format them as
                # the JSON the model would have produced.
                tc = msg.get("tool_calls")
                if tc:
                    tc_text_parts = []
                    for call in tc:
                        fn = call.get("function", {})
                        tc_text_parts.append(
                            json.dumps({
                                "name": fn.get("name"),
                                "arguments": json.loads(fn.get("arguments", "{}")),
                            })
                        )
                    parts.append(f"<|im_start|>assistant\n{chr(10).join(tc_text_parts)}")
                else:
                    parts.append(f"<|im_start|>assistant\n{content}")
            elif role == "tool":
                # Tool result — format as a user message with a clear prefix.
                parts.append(f"<|im_start|>user\n[Tool Result]\n{content}")

        # When tools are available, add a final reminder right before the
        # assistant turn.  Small models follow instructions better when they
        # appear immediately before the generation point.
        if tool_names:
            # Build concrete examples for each tool — no abstract placeholders
            # because the 1.2B model copies them literally.
            examples: list[str] = []
            for t in tools or []:
                fn = t.get("function", {})
                name = fn.get("name", "")
                params = fn.get("parameters", {}).get("properties", {})
                required = fn.get("parameters", {}).get("required", [])
                # Build an example call with the first required param.
                if required:
                    pname = required[0]
                    ptype = params.get(pname, {}).get("type", "string")
                    if ptype == "string":
                        examples.append(f'[{name}({pname}="value")]')
                    else:
                        examples.append(f'[{name}({pname}=value)]')
                else:
                    examples.append(f"[{name}()]")
            ex_str = "  ".join(examples[:4])
            parts.append(
                f"<|im_start|>system\n"
                f"Use a tool now. Examples:\n"
                f"{ex_str}"
            )

        parts.append("<|im_start|>assistant\n")
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
            LFMEngine._coerce_content(m["content"])
            for m in messages
            if m.get("role") == "user"
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
        # Pi sends max_completion_tokens; fall back to max_tokens.
        max_tokens: int = (
            body.get("max_completion_tokens")
            or body.get("max_tokens")
            or 1024
        )
        stream: bool = body.get("stream", False)
        tools: list[dict] | None = body.get("tools")
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Apply Oczy augmentation if requested.
        if "oczy" in model and augmenter is not None:
            messages = augmenter.augment(messages)

        result = engine.generate(messages, max_tokens=max_tokens, tools=tools)
        content = result["content"]
        tool_calls = result["tool_calls"]
        finish_reason = "tool_calls" if tool_calls else "stop"

        if stream:
            async def _stream():
                if tool_calls:
                    # Send tool_calls as a single chunk.
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": tool_calls,
                            },
                            "finish_reason": finish_reason,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": finish_reason,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(_stream(), media_type="text/event-stream")

        message: dict = {"role": "assistant"}
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
        else:
            message["content"] = content

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
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
