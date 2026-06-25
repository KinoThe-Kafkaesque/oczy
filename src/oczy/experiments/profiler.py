"""Lightweight per-component profiling instrumentation for Oczy agents.

Uses only the Python standard library: ``time`` for wall-clock elapsed time,
``tracemalloc`` for peak allocated-memory deltas when tracing, ``sys`` for a
fall-back size estimate, and ``pickle`` for the size helper.
"""

from __future__ import annotations

import pickle
import sys
import time
import tracemalloc
from typing import Any


class ComponentProfiler:
    """Profiles one named component (a region of code, module call, etc.).

    ``peak_memory_bytes`` keeps the maximum delta observed across all calls.
    Time is cumulative.  This class is also a context manager so the
    ``AgentProfiler`` can return it directly from ``profile()``.
    """

    __slots__ = ("name", "call_count", "total_time_ms", "peak_memory_bytes", "_start_time_ns", "_baseline_peak")

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count: int = 0
        self.total_time_ms: float = 0.0
        self.peak_memory_bytes: int = 0

    def __enter__(self) -> "ComponentProfiler":
        self._start_time_ns = time.perf_counter_ns()
        if tracemalloc.is_tracing():
            tracemalloc.reset_peak()
            self._baseline_peak = tracemalloc.get_traced_memory()[1]
        else:
            # Keep a stable per-call baseline without starting tracing.
            self._baseline_peak = sys.getsizeof({})
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed_ns = time.perf_counter_ns() - self._start_time_ns
        self.total_time_ms += elapsed_ns / 1_000_000
        self.call_count += 1

        if tracemalloc.is_tracing():
            peak = tracemalloc.get_traced_memory()[1]
            delta = peak - self._baseline_peak
        else:
            # Non-tracing mode: we cannot meaningfully attribute allocation to
            # the component, so report 0.
            delta = 0

        if delta > self.peak_memory_bytes:
            self.peak_memory_bytes = delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "total_time_ms": self.total_time_ms,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


class AgentProfiler:
    """Collection of ``ComponentProfiler`` instances keyed by component name."""

    def __init__(self, components: list[str]) -> None:
        self._profilers: dict[str, ComponentProfiler] = {
            name: ComponentProfiler(name) for name in components
        }

    def profile(self, component: str) -> ComponentProfiler:
        """Return the profiler for *component*, creating it if necessary."""
        if component not in self._profilers:
            self._profilers[component] = ComponentProfiler(component)
        return self._profilers[component]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return a dict mapping each component to its stats."""
        return {name: profiler.to_dict() for name, profiler in self._profilers.items()}

    def reset(self) -> None:
        """Reset every component profiler to its initial state."""
        for profiler in self._profilers.values():
            profiler.call_count = 0
            profiler.total_time_ms = 0.0
            profiler.peak_memory_bytes = 0


def measure_size(obj: Any) -> int:
    """Approximate byte size of *obj* using ``pickle.dumps`` when possible.

    Falls back to ``sys.getsizeof`` if ``pickle`` fails.
    """
    try:
        return len(pickle.dumps(obj))
    except Exception:
        return sys.getsizeof(obj)
