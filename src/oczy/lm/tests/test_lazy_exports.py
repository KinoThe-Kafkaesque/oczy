"""Regression tests for optional llama-cpp imports at the LM package boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3]
_BLOCK_LLAMA_CPP = r"""
import importlib.abc
import sys

class BlockLlamaCpp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "llama_cpp" or fullname.startswith("llama_cpp."):
            raise ModuleNotFoundError("No module named 'llama_cpp'", name="llama_cpp")
        return None

sys.modules.pop("llama_cpp", None)
sys.meta_path.insert(0, BlockLlamaCpp())
"""


def _run_python(code: str, *, pythonpath: list[Path] | None = None) -> subprocess.CompletedProcess[str]:
    paths = [str(path) for path in (pythonpath or [])]
    paths.append(str(_SRC_ROOT))
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        os.pathsep.join(paths + [env["PYTHONPATH"]])
        if env.get("PYTHONPATH")
        else os.pathsep.join(paths)
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stderr + result.stdout


def test_hf_driver_import_does_not_require_llama_cpp() -> None:
    """HF-only driver import remains usable when the optional llama-cpp package is absent."""
    result = _run_python(
        _BLOCK_LLAMA_CPP
        + r'''
import sys
from oczy.lm.hf_driver import HFDriver, KVHandle

assert HFDriver.__name__ == "HFDriver"
assert KVHandle.__name__ == "KVHandle"
assert "llama_cpp" not in sys.modules
'''
    )

    _assert_ok(result)


def test_cvec_exports_without_llama_cpp_resolve_or_raise_by_dependency_need() -> None:
    """Dependency-free exports resolve; cvec driver exports raise the missing dependency."""
    result = _run_python(
        _BLOCK_LLAMA_CPP
        + r'''
import oczy.lm as lm

assert "ReservedPosition" in lm.__all__, lm.__all__
reserved_position = lm.ReservedPosition
assert reserved_position.__name__ == "ReservedPosition"
assert "llama_cpp" not in sys.modules

for name in {"CVecDriverConfig", "LlamaCVecDriver"}:
    assert name in lm.__all__, lm.__all__
    try:
        getattr(lm, name)
    except ModuleNotFoundError as exc:
        assert exc.name == "llama_cpp", (name, exc.name, str(exc))
    else:
        raise AssertionError(f"{name} resolved even though llama_cpp was blocked")
'''
    )

    _assert_ok(result)


def test_lazy_cvec_exports_resolve_with_available_llama_cpp_stub(tmp_path: Path) -> None:
    """With llama-cpp importable, public cvec names resolve from ``oczy.lm`` and stay exported."""
    (tmp_path / "llama_cpp.py").write_text(
        '"""Minimal import-time llama_cpp stub for package-boundary tests."""\n'
        '__version__ = "0.3.31"\n'
        "class Llama:\n"
        "    pass\n",
        encoding="utf-8",
    )

    result = _run_python(
        r'''
import oczy.lm as lm

expected = {"CVecDriverConfig", "LlamaCVecDriver", "ReservedPosition"}
assert expected <= set(lm.__all__), lm.__all__

resolved = {name: getattr(lm, name) for name in expected}
assert resolved["CVecDriverConfig"].__module__ == "oczy.lm.cvec_driver"
assert resolved["LlamaCVecDriver"].__module__ == "oczy.lm.cvec_driver"
assert resolved["ReservedPosition"].__name__ == "ReservedPosition"
assert all(getattr(lm, name) is value for name, value in resolved.items())
assert expected <= set(lm.__all__), lm.__all__
''',
        pythonpath=[tmp_path],
    )

    _assert_ok(result)
