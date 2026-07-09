"""A single pickle-based memory metrics helper.

Every organ used to report its memory footprint using a different
convention: ``trace_bytes`` for raw buffers, ``sys.getsizeof`` on a
JSON-serialised wrapper (Python object overhead, not payload bytes),
``len(json.dumps(...))`` (UTF-8 string length, ignoring the
object's own Python-internal state), or no ``status()`` at all
(:class:`WorldModelCritic`).

This module is the single source of truth for what
``status()["serialized_bytes"]`` means across every organ: the length of
``pickle.dumps`` of the organ's persistent state in its highest
protocol.  Pickle is the same serializer used by every ``save()`` path
in the project, so the number reported here matches the number that
ends up on disk.
"""

from __future__ import annotations

import pickle
from typing import Any


def mem_bytes(obj: Any) -> int:
    """Return the size of ``obj`` when pickled at the highest protocol.

    Falls back to ``json.dumps(...).encode("utf-8")`` length when pickle
    raises (e.g. for objects holding file handles).  Returns ``0`` when
    even JSON serialisation fails.
    """
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        import json

        try:
            return len(json.dumps(obj, default=str).encode("utf-8"))
        except Exception:
            return 0