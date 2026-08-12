"""Frozen-by-code dataset for the six-stage Pi tool-use curriculum."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolEpisode:
    id: str
    request: str
    expected_tools: tuple[str, ...] = ()
    expected_params: tuple[tuple[str, str], ...] = ()
    expected_answer_terms: tuple[str, ...] = ()
    tool_results: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolStage:
    id: str
    name: str
    skill: str
    episodes: tuple[ToolEpisode, ...]


def _ep(
    id: str,
    request: str,
    *tools: str,
    params: dict[str, str] | None = None,
    terms: tuple[str, ...] = (),
    results: tuple[str, ...] = (),
) -> ToolEpisode:
    return ToolEpisode(
        id=id,
        request=request,
        expected_tools=tuple(tools),
        expected_params=tuple(sorted((params or {}).items())),
        expected_answer_terms=terms,
        tool_results=results,
    )


def build_tool_curriculum() -> tuple[ToolStage, ...]:
    """Return the complete 8/12/8/6/8/3 stage substrate from Experiment 08."""
    return (
        ToolStage(
            "stage_0_format",
            "Tool-format grounding",
            "Emit one parseable tool call instead of prose.",
            (
                _ep("s0_read", "Read the file config.toml", "read", params={"path": "config.toml"}),
                _ep("s0_bash", "List files in src/", "bash", params={"command": "ls src/"}),
                _ep("s0_write", "Create hello.py with print('hi')", "write", params={"path": "hello.py"}),
                _ep("s0_read2", "Show me main.py", "read", params={"path": "main.py"}),
                _ep("s0_bash2", "Run the tests", "bash", params={"command": "pytest"}),
                _ep("s0_write2", "Save TODO to notes.txt", "write", params={"path": "notes.txt"}),
                _ep("s0_read3", "Open utils.py", "read", params={"path": "utils.py"}),
                _ep("s0_bash3", "Check git status", "bash", params={"command": "git status"}),
            ),
        ),
        ToolStage(
            "stage_1_selection",
            "Tool selection",
            "Choose the correct tool and copy request parameters.",
            (
                _ep("s1_read_1", "Read pyproject.toml", "read", params={"path": "pyproject.toml"}),
                _ep("s1_read_2", "Show src/main.py", "read", params={"path": "src/main.py"}),
                _ep("s1_bash_1", "List all Python files", "bash", params={"command": "find"}),
                _ep("s1_bash_2", "Run pytest", "bash", params={"command": "pytest"}),
                _ep("s1_write_1", "Create test.py", "write", params={"path": "test.py"}),
                _ep("s1_write_2", "Save hello to out.txt", "write", params={"path": "out.txt"}),
                _ep("s1_edit_1", "Change foo to bar in config.py", "edit", params={"path": "config.py"}),
                _ep("s1_edit_2", "Replace old with new in app.py", "edit", params={"path": "app.py"}),
                _ep("s1_read_3", "What's in README.md?", "read", params={"path": "README.md"}),
                _ep("s1_bash_3", "Show git log", "bash", params={"command": "git log"}),
                _ep("s1_write_3", "Write done to complete.txt", "write", params={"path": "complete.txt"}),
                _ep("s1_read_4", "Open the Dockerfile", "read", params={"path": "Dockerfile"}),
            ),
        ),
        ToolStage(
            "stage_2_result",
            "Tool-result integration",
            "Use a tool result to answer the original request.",
            (
                _ep("s2_name", "Read pyproject.toml and tell me the project name", "read", terms=("oczy",), results=('name = "oczy"',)),
                _ep("s2_ver", "Read pyproject.toml and tell me the version", "read", terms=("0.1.0",), results=('version = "0.1.0"',)),
                _ep("s2_count", "Count Python files in src", "bash", terms=("42",), results=("42",)),
                _ep("s2_branch", "Show current git branch", "bash", terms=("main",), results=("main",)),
                _ep("s2_deps", "List dependencies", "read", terms=("numpy", "pytest"), results=("numpy, pytest",)),
                _ep("s2_test", "Run pytest and report status", "bash", terms=("pass",), results=("5 passed",)),
                _ep("s2_file", "Read utils.py and name its function", "read", terms=("helper",), results=("def helper():",)),
                _ep("s2_error", "Run the script and report the error", "bash", terms=("syntaxerror",), results=("SyntaxError: invalid syntax",)),
            ),
        ),
        ToolStage(
            "stage_3_chain",
            "Multi-turn tool chains",
            "Complete an ordered tool sequence and ground the final answer.",
            (
                _ep("s3_read_edit", "Read config.py then enable DEBUG", "read", "edit", terms=("done",)),
                _ep("s3_bash_read", "Find the file containing main then read it", "bash", "read"),
                _ep("s3_read_bash", "Read the test then run it", "read", "bash", terms=("pass",)),
                _ep("s3_write_bash", "Create a hello script then run it", "write", "bash", terms=("hello",)),
                _ep("s3_bash_bash", "Check git status then commit if clean", "bash", "bash"),
                _ep("s3_read_read", "Read pyproject.toml and README.md and compare", "read", "read"),
            ),
        ),
        ToolStage(
            "stage_4_ambiguity",
            "Tool selection under ambiguity",
            "Resolve intent where a keyword suggests the wrong tool.",
            (
                _ep("s4_grep", "Find files containing import numpy", "bash"),
                _ep("s4_find", "Find files named config.py", "bash"),
                _ep("s4_edit_vs_write", "Fix a typo in README.md", "edit"),
                _ep("s4_read_vs_bash", "What does the Makefile do?", "read"),
                _ep("s4_bash_vs_read", "Show the directory structure", "bash"),
                _ep("s4_write_vs_edit", "Add a test to test_api.py", "edit"),
                _ep("s4_bash_vs_edit", "Rename all txt files to md", "bash"),
                _ep("s4_read_vs_grep", "Where is def main defined?", "bash"),
            ),
        ),
        ToolStage(
            "stage_5_pi",
            "Pi integration",
            "Run the three external Pi tasks after the direct curriculum.",
            (
                _ep("read-file", "Read pyproject.toml and tell me the project name", "read", terms=("oczy",)),
                _ep("find-file", "Find Python files containing CortexAgent", "bash", terms=("cortex_agent.py",)),
                _ep("edit-file", "Create /tmp/oczy_bench_marker.py", "write", params={"path": "/tmp/oczy_bench_marker.py"}),
            ),
        ),
    )
