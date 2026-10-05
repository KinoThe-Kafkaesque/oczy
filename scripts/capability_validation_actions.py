"""Strict JSON actions with real temporary-file outcomes, not substring scoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

ACTION_SYSTEM = '''You operate a small filesystem. Respond with exactly one JSON object per turn.
To call a tool use {"tool":"NAME","arguments":{...}}. Tool results arrive in the next user message.
Available tools and exact string argument names:
read_file(path): read an existing UTF-8 file.
write_file(path, content): create or overwrite a UTF-8 file with the exact content.
replace_text(path, old, new): replace the one exact occurrence of old with new in an existing file.
Paths are relative and case-sensitive. Preserve all other files and content.
After completing the requested actions respond with {"final":"REQUESTED FINAL STRING"}.
Do not use markdown fences, explanations, extra keys, or imagined tool results.'''

SCHEMAS = {"read_file": {"path"}, "write_file": {"path", "content"}, "replace_text": {"path", "old", "new"}}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate JSON key")
        obj[key] = value
    return obj


def parse_action(text):
    obj = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    if set(obj) == {"final"} and isinstance(obj["final"], str):
        return obj
    if set(obj) != {"tool", "arguments"} or not isinstance(obj["tool"], str):
        raise ValueError("Invalid action fields")
    name, args = obj["tool"], obj["arguments"]
    if name not in SCHEMAS or not isinstance(args, dict) or set(args) != SCHEMAS[name]:
        raise ValueError("Unknown tool or wrong argument fields")
    if any(not isinstance(v, str) or len(v.encode()) > 1024 for v in args.values()):
        raise ValueError("Arguments must be bounded strings")
    return obj


def execute(root, call):
    call = parse_action(json.dumps(call))
    if "final" in call:
        raise ValueError("Final is not an executable tool")
    name, args = call["tool"], call["arguments"]
    relative = Path(args["path"])
    if relative.is_absolute() or not args["path"] or ".." in relative.parts or relative == Path("."):
        raise ValueError("Invalid relative path")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents]) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes workspace or is a symlink")
    if name == "read_file":
        return {"content": path.read_text()}
    if name == "write_file":
        path.write_text(args["content"])
    else:
        text = path.read_text()
        if not args["old"] or text.count(args["old"]) != 1:
            raise ValueError("Replacement requires one nonempty exact match")
        path.write_text(text.replace(args["old"], args["new"], 1))
    return {"ok": True}


def snapshot(root):
    return {str(p.relative_to(root)): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def run_case(case, generate, *, max_calls=4):
    """Only task text and actual tool outputs enter the actor, never gold actions."""
    with tempfile.TemporaryDirectory(prefix="oczy-actions-") as directory:
        root = Path(directory)
        for name, content in case["initial_files"].items():
            (root / name).write_text(content)
        prompt = [{"role": "system", "content": ACTION_SYSTEM}, {"role": "user", "content": case["request"]}]
        calls, transcript, final, error = [], [], None, None
        for _ in range(max_calls + 1):
            output = generate(prompt)
            transcript.append({"generated": output})
            try:
                action = parse_action(output)
                if "final" in action:
                    final = action["final"]
                    break
                if len(calls) >= max_calls:
                    raise ValueError("Tool-call budget exceeded")
                calls.append(action)
                result = execute(root, action)
                transcript[-1]["actual_tool_result"] = result
                prompt.extend([{"role": "assistant", "content": output},
                               {"role": "user", "content": "Tool result: " + json.dumps(result)}])
            except (ValueError, OSError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
        actual = snapshot(root)
    fields = {"tool_sequence_exact": calls == case["expected_calls"],
              "filesystem_exact": actual == case["expected_files"],
              "final_exact": final == case["expected_final"], "execution_error_free": error is None}
    return {"case": case["name"], "transcript": transcript, "calls": calls, "final": final,
            "error": error, "actual_files": actual, **fields, "correct": all(fields.values())}
