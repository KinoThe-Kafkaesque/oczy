"""Dependency-injectable Google Colab CLI 0.6.0 provider for the parallel scheduler.

Wraps the ``colab`` CLI (v0.6.0) as a provider-neutral remote execution backend
alongside the existing Kaggle client.  The scheduler drives Colab jobs through
the :class:`ColabClient` protocol; :class:`ColabCliClient` is the real
subprocess-backed implementation.

Design constraints (enforced here):
  * **CPU-only**: ``colab run`` is invoked WITHOUT ``--gpu``/``--tpu``.  The
    CLI defaults to ``Variant.DEFAULT`` + ``Accelerator.NONE`` when neither
    flag is present.  We never construct an argv containing accelerator flags.
  * **No shell=True**: every subprocess call uses an explicit argv list.
  * **Stable session identity**: the caller-supplied session ``name`` is the
    durable job identity (analogous to a Kaggle kernel slug).  It appears in
    ``colab sessions`` output and survives scheduler restarts via the CLI's
    own ``~/.config/colab-cli/sessions.json`` state file.
  * **Exact capacity detection**: HTTP 412 from the Colab backend raises
    ``TooManyAssignmentsError`` inside the CLI, which prints a full Python
    traceback to **stderr** and exits non-zero.  :func:`classify_colab_output`
    requires **both** a non-zero exit code **and** the marker in stderr
    (not stdout) so that a successful script merely printing ``precondition
    failed`` is classified as ``COLAB_COMPLETE``, not a capacity rejection.
  * **Deadlock-free output collection**: :meth:`ColabCliClient.collect` reads
    stdout/stderr from scheduler-owned temp files (not OS pipes) and writes
    ``stdout.log``, ``stderr.log``, and ``result.json`` into the output
    directory.  Temp-file redirection ensures arbitrary output volume cannot
    block the process on a full pipe buffer.  It is safe to call multiple
    times — output is read once and cached; file writes overwrite.  Temp
    files are closed and unlinked after collection.  ``collect`` does NOT
    stop the session; the scheduler always calls :meth:`stop` explicitly
    after collection (and on failure/interrupt) so every allocated session
    is torn down.
  * **Every session stopped**: :meth:`stop` is best-effort and safe to call on
    a non-existent or already-stopped session (the CLI prints a message and
    returns exit 0).  The scheduler calls ``stop`` after collect, on error,
    and during restart recovery for orphaned handles.
  * **Injectable for tests**: the process factory (``popen_factory``) and
    command runner (``runner``) are constructor-injected.  Tests pass fakes
    without subclassing.

CLI semantics (source-verified against googlecolab/google-colab-cli 0.6.0):

  ``colab sessions``
      Human-readable, one line per session::
          [<name>] <endpoint> | Hardware: <X> | Variant: <Y>
      Orphaned server-side assignments show ``[?]``.  No sessions yields
      ``[colab] No active sessions found on server.``  There is no ``--json``
      flag; we parse the text format (pinned by upstream tests).

  ``colab run --keep --session <name> --timeout <seconds> -- SCRIPT ARGS...``
      Blocks until script completion.  ``--timeout`` is the Jupyter kernel
      per-execution quiet-period ceiling (default 30s), not a total wall-clock
      job timeout.  The ``--`` separator ensures SCRIPT and ARGS are treated
      as positionals, not CLI flags (e.g. ``--gpu`` in ARGS is forwarded to
      the script).  Script ``print()`` streams to stdout; progress/tracebacks
      go to stderr.  Exit code propagates ``sys.exit(N)``.  ``--keep`` keeps
      the session alive after completion for output retrieval and later
      ``colab stop``.

  ``colab stop --session <name>``
      Kills keep-alive daemon, shuts down kernel, unassigns VM, removes local
      state.  Returns exit 0 even if the session is not found.

Module is importable both as a package member (``infrastructure.kaggle.colab_provider``)
and directly as a script (``from colab_provider import ...``) — no relative
imports, guarded ``__main__`` block.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Status constants (imported by the scheduler)
# ---------------------------------------------------------------------------

COLAB_RUNNING = "running"
COLAB_COMPLETE = "complete"
COLAB_ERROR = "error"
COLAB_CAPACITY_REJECTED = "capacity_rejected"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COLAB_SESSIONS_TIMEOUT = 30.0
DEFAULT_COLAB_STOP_TIMEOUT = 60.0
DEFAULT_COLAB_RUN_TIMEOUT = 21600  # 6h, matches Kaggle DEFAULT_JOB_TIMEOUT

# CLI progress prefix; lines starting with this on stderr are infrastructure,
# not script output.
_COLAB_PREFIX = "[colab]"

# Regex for parsing ``colab sessions`` lines: ``[<name>] <endpoint> | ...``
_SESSION_LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(.+)$")
_NO_SESSIONS_RE = re.compile(r"no active sessions", re.IGNORECASE)

# Capacity-rejection markers (HTTP 412 traceback).
_CAPACITY_MARKERS = ("toomanyassignmentserror", "precondition failed")


# ---------------------------------------------------------------------------
# Capacity detection helpers
# ---------------------------------------------------------------------------


def is_capacity_rejected(output: str) -> bool:
    """Return True iff *output* indicates an HTTP 412 capacity rejection.

    The Colab CLI does not catch ``TooManyAssignmentsError``; it propagates as
    an unhandled exception, printing a full Python traceback to stderr.  We
    match case-insensitively against the exception type name
    (``TooManyAssignmentsError``) or the HTTP reason phrase
    (``Precondition Failed``).
    """
    lowered = output.lower()
    return any(marker in lowered for marker in _CAPACITY_MARKERS)


def classify_colab_output(proc: subprocess.Popen) -> str:
    """Classify a finished ``colab run`` Popen into a status constant.

    Reads the process output from scheduler-owned temp files (stashed as
    ``_colab_stdout_path`` / ``_colab_stderr_path`` on the proc by
    :func:`_default_popen_factory`) or, if those attrs are absent (e.g. an
    injected fake Popen), from ``proc.stdout`` / ``proc.stderr`` text pipes.
    The output is drained exactly once; repeated calls return the same
    classification from cached content stored on the process object.

    Exit-code harvesting:
      ``proc.returncode`` is None until the process has been polled.  If it
      is None, ``poll()`` is called (nonblocking) to obtain the real exit
      code.  If poll() also returns None the process is still running →
      ``COLAB_RUNNING`` is returned without caching or draining pipes, so a
      later call can re-classify once the process exits.

    Classification order (once an exit code is available):
      1. If the process exited non-zero **and** the **stderr** output
         contains a capacity-rejection marker → ``COLAB_CAPACITY_REJECTED``.
         Both conditions are required: a successful script (exit 0) that
         merely prints ``precondition failed`` to stdout is NOT a capacity
         rejection.  The 412 traceback always appears on stderr with a
         non-zero exit.
      2. If exit code is 0 → ``COLAB_COMPLETE``.
      3. Otherwise → ``COLAB_ERROR``.

    The Popen must have been started either with temp-file redirection (real
    factory) or with ``stdout=PIPE, stderr=PIPE, text=True`` (fake) so that
    the output is readable.  If the pipes/files were already consumed by the
    caller, this function falls back to empty strings (and classification
    proceeds on exit code alone).
    """
    cached = getattr(proc, "_colab_classified_output", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    # Obtain the exit code robustly.  A Popen whose ``returncode`` is None
    # may have already terminated but not yet been polled (an injected
    # fake sets returncode only on poll(); a real Popen is polled by the
    # scheduler before calling us, but direct callers may not).  Call
    # ``poll()`` — which is nonblocking — to harvest the real exit code.
    # If poll() also returns None the process is still running: do NOT
    # misclassify it as complete/error, and do not cache or drain pipes
    # so a later call can re-classify once it exits.
    rc = proc.returncode
    if rc is None:
        rc = proc.poll()
    if rc is None:
        return COLAB_RUNNING

    stdout_text = _read_proc_output(proc, "stdout")
    stderr_text = _read_proc_output(proc, "stderr")

    # F6: capacity rejection requires BOTH a non-zero exit code AND the
    # marker in stderr (where the Python traceback appears).  A successful
    # script that prints "precondition failed" to stdout is not a rejection.
    if rc != 0 and is_capacity_rejected(stderr_text):
        result = COLAB_CAPACITY_REJECTED
    elif rc == 0:
        result = COLAB_COMPLETE
    else:
        result = COLAB_ERROR

    # Cache so repeated calls don't re-read exhausted pipes/files.
    proc._colab_classified_output = result  # type: ignore[attr-defined]
    proc._colab_stdout = stdout_text  # type: ignore[attr-defined]
    proc._colab_stderr = stderr_text  # type: ignore[attr-defined]
    return result


def _drain_pipe(stream: Any) -> str:
    """Read a text pipe to EOF, returning empty string if already consumed."""
    if stream is None:
        return ""
    try:
        content = stream.read()
    except (ValueError, OSError):
        # Pipe already closed or consumed.
        return ""
    return content if content is not None else ""


def _read_proc_output(proc: subprocess.Popen, stream: str) -> str:
    """Read *stream* (``"stdout"`` or ``"stderr"``) from a Popen.

    Real Popens created by :func:`_default_popen_factory` stash temp file
    paths as ``_colab_stdout_path`` / ``_colab_stderr_path``.  If present,
    read from the file.  Otherwise (injected fake Popen), fall back to
    ``proc.stdout`` / ``proc.stderr`` pipe draining via :func:`_drain_pipe`.
    """
    path = getattr(proc, f"_colab_{stream}_path", None)
    if path is not None:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except OSError:
            return ""
    return _drain_pipe(getattr(proc, stream, None))


def _cleanup_proc_tempfiles(proc: subprocess.Popen) -> None:
    """Close and unlink scheduler-owned temp files stashed on *proc*.

    Safe to call on a Popen without temp-file attrs (e.g. an injected fake)
    and idempotent (safe to call multiple times — attrs are cleared after
    cleanup).
    """
    for attr in ("_colab_stdout_file", "_colab_stderr_file"):
        fh = getattr(proc, attr, None)
        if fh is not None:
            try:
                fh.close()
            except (ValueError, OSError):
                pass
            setattr(proc, attr, None)
    for attr in ("_colab_stdout_path", "_colab_stderr_path"):
        path = getattr(proc, attr, None)
        if path is not None:
            try:
                os.unlink(path)
            except OSError:
                pass
            setattr(proc, attr, None)


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------


def parse_sessions(output: str) -> list[dict[str, str]]:
    """Parse ``colab sessions`` stdout into a list of session dicts.

    Each dict is ``{"name": str, "state": str}``.  The ``state`` field is the
    raw session state as reported by the CLI (typically ``"running"`` for
    active sessions; orphaned server-side assignments get name ``"?"``).

    The scheduler counts every returned entry as one active slot occupying an
    account-wide assignment, including sessions started by other processes
    (external sessions).
    """
    sessions: list[dict[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if _NO_SESSIONS_RE.search(line):
            continue
        match = _SESSION_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        # Every line in `colab sessions` represents an active server-side
        # assignment.  The CLI does not print idle/inactive sessions — only
        # live assignments appear.  So "state" is uniformly "running".
        sessions.append({"name": name, "state": COLAB_RUNNING})
    return sessions


# ---------------------------------------------------------------------------
# Protocol (consumed by the scheduler)
# ---------------------------------------------------------------------------


@runtime_checkable
class ColabClient(Protocol):
    """Dependency-injectable Colab CLI adapter.

    Any object with matching ``sessions``, ``run``, ``collect``, ``stop``,
    and ``remember`` methods satisfies this protocol — tests can pass a fake
    without subclassing.
    """
    def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
        """List account-wide active Colab sessions.

        Returns a list of ``{"name": str, "state": str}`` dicts.  Every entry
        represents a live server-side assignment occupying one slot,
        including sessions started externally.  The scheduler uses
        ``len(result)`` for external-session capacity accounting.
        """
        ...

    def run(
        self,
        name: str,
        script: str,
        *,
        arguments: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.Popen:
        """Launch a kept CPU-only Colab session executing *script*.

        Returns the ``subprocess.Popen`` immediately for the scheduler to
        poll.  The argv is::

            colab run --keep --session <name> --timeout <seconds> -- SCRIPT ARGS...

        The ``--`` separator ensures *script* and *arguments* are treated as
        positionals, not CLI flags (e.g. a script arg ``--gpu`` is forwarded
        to the script, not consumed by the CLI).  No ``--gpu`` or ``--tpu``
        flags are ever passed (CPU-only).  stdout/stderr are redirected to
        scheduler-owned temp files (real factory) or captured pipes (fake)
        for later classification and collection.
        """
        ...

    def collect(self, name: str, output_dir: str) -> dict[str, Any]:
        """Collect stdout/stderr/result metadata into *output_dir*.

        Writes ``stdout.log``, ``stderr.log``, and ``result.json``.  Safe to
        call multiple times (idempotent file writes).  Does NOT stop the
        session — the scheduler calls :meth:`stop` explicitly afterward.

        Returns ``{"ok": bool, "error": str | None}``.
        """
        ...

    def stop(self, name: str, *, timeout: float | None = None) -> None:
        """Stop a named Colab session (best-effort).

        Safe to call on a non-existent or already-stopped session.  Kills the
        keep-alive daemon, shuts down the kernel, unassigns the VM, and
        removes local CLI state.
        """
        ...

    def remember(self, name: str, proc: subprocess.Popen) -> None:
        """Associate a Popen with a session name for later :meth:`collect`.

        Called by the scheduler after :meth:`run` so that :meth:`collect`
        can retrieve the process handle by session name.  The scheduler may
        also call :func:`classify_colab_output` on the proc before collect;
        the cached output is then available to collect without re-reading
        exhausted pipes/files.
        """
        ...
# ---------------------------------------------------------------------------
# Real CLI implementation
# ---------------------------------------------------------------------------

# Type aliases for injected factories.
Runner = Callable[[list[str], float | None], subprocess.CompletedProcess]
PopenFactory = Callable[[list[str]], subprocess.Popen]


def _default_runner(argv: list[str], timeout: float | None) -> subprocess.CompletedProcess:
    """Run a colab CLI command synchronously, returning stdout.  No shell=True."""
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"colab {' '.join(argv[:2])} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _default_popen_factory(argv: list[str]) -> subprocess.Popen:
    """Launch a colab CLI command with stdout/stderr redirected to temp files.

    Using scheduler-owned temp files instead of ``subprocess.PIPE`` ensures
    that arbitrary output volume cannot deadlock the process on a full 64KB
    OS pipe buffer.  The temp file paths and handles are stashed on the
    returned Popen as ``_colab_stdout_path`` / ``_colab_stderr_path`` and
    ``_colab_stdout_file`` / ``_colab_stderr_file`` for later reading by
    :func:`classify_colab_output` and :meth:`ColabCliClient.collect`, and
    cleanup by :func:`_cleanup_proc_tempfiles`.
    """
    stdout_fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".out", delete=False, encoding="utf-8"
    )
    stderr_fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".err", delete=False, encoding="utf-8"
    )
    proc = subprocess.Popen(
        argv,
        stdout=stdout_fh,
        stderr=stderr_fh,
        text=True,
    )
    # Stash temp file paths and handles on the proc for later reading/cleanup.
    proc._colab_stdout_path = stdout_fh.name  # type: ignore[attr-defined]
    proc._colab_stderr_path = stderr_fh.name  # type: ignore[attr-defined]
    proc._colab_stdout_file = stdout_fh  # type: ignore[attr-defined]
    proc._colab_stderr_file = stderr_fh  # type: ignore[attr-defined]
    return proc


class ColabCliClient:
    """Real Colab CLI 0.6.0 adapter using subprocess (no shell=True).

    Constructor injects the process factory (``popen_factory``) and command
    runner (``runner``) for deterministic testing.  Defaults use real
    ``subprocess.Popen`` / ``subprocess.run`` with explicit argv lists.
    """

    def __init__(
        self,
        *,
        runner: Runner | None = None,
        popen_factory: PopenFactory | None = None,
        sessions_timeout: float = DEFAULT_COLAB_SESSIONS_TIMEOUT,
        stop_timeout: float = DEFAULT_COLAB_STOP_TIMEOUT,
        run_timeout: float = DEFAULT_COLAB_RUN_TIMEOUT,
    ) -> None:
        self._runner = runner if runner is not None else _default_runner
        self._popen_factory = popen_factory if popen_factory is not None else _default_popen_factory
        self.sessions_timeout = sessions_timeout
        self.stop_timeout = stop_timeout
        self.run_timeout = run_timeout

    # -- sessions ---------------------------------------------------------

    def sessions(self, *, timeout: float | None = None) -> list[dict[str, str]]:
        """List account-wide active Colab sessions via ``colab sessions``."""
        to = timeout if timeout is not None else self.sessions_timeout
        result = self._runner(["colab", "sessions"], to)
        return parse_sessions(result.stdout)

    # -- run --------------------------------------------------------------

    def run(
        self,
        name: str,
        script: str,
        *,
        arguments: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.Popen:
        """Launch a kept CPU-only session executing *script*.

        Returns the Popen immediately for the scheduler to poll.  The argv
        is::

            colab run --keep --session <name> --timeout <seconds> -- SCRIPT ARGS...

        The ``--`` separator (F3) ensures *script* and *arguments* are
        treated as positionals by the CLI, not as flags — a script or
        argument starting with ``--`` (e.g. ``--gpu``) is forwarded to the
        script, not consumed by the CLI.  No ``--gpu`` or ``--tpu`` flags
        are ever passed (CPU-only).  The timeout is clamped to at least 1
        second (F11) so sub-second values don't truncate to ``--timeout 0``.
        stdout/stderr are redirected to temp files (real factory) or pipes
        (fake) for :func:`classify_colab_output` and :meth:`collect`.
        """
        to = timeout if timeout is not None else self.run_timeout
        # F11: clamp to >= 1 so int() truncation never yields 0.
        timeout_str = str(int(max(1, to)))
        argv: list[str] = [
            "colab", "run",
            "--keep",
            "--session", name,
            "--timeout", timeout_str,
            "--",  # F3: separator — positionals after this are not CLI flags
            script,
        ]
        if arguments:
            argv.extend(arguments)
        proc = self._popen_factory(argv)
        # Stash the session name for collect()/stop() correlation.
        proc._colab_session_name = name  # type: ignore[attr-defined]
        return proc

    # -- collect ----------------------------------------------------------

    def collect(self, name: str, output_dir: str) -> dict[str, Any]:
        """Collect stdout/stderr/result metadata into *output_dir*.

        Writes ``stdout.log``, ``stderr.log``, and ``result.json``.  Idempotent:
        safe to call multiple times (overwrites files).  Does NOT stop the
        session — the scheduler calls :meth:`stop` explicitly afterward.
        Scheduler-owned temp files (F1) are cleaned up here.

        ``name`` is the session name; the scheduler passes the same stable
        name used for :meth:`run`.  The Popen handle is retrieved from the
        internal ``_procs`` map populated by :meth:`remember`.  If the
        Popen's output was already read by :func:`classify_colab_output`,
        the cached content is used; this makes collect idempotent with
        respect to classification.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        proc = self._procs.get(name)
        stdout_text = ""
        stderr_text = ""
        exit_code: int | None = None
        status = COLAB_ERROR

        try:
            if proc is not None:
                # Use cached output from classify_colab_output if available.
                cached_stdout = getattr(proc, "_colab_stdout", None)
                cached_stderr = getattr(proc, "_colab_stderr", None)
                cached_status = getattr(proc, "_colab_classified_output", None)
                if cached_stdout is not None:
                    stdout_text = cached_stdout
                else:
                    stdout_text = _read_proc_output(proc, "stdout")
                if cached_stderr is not None:
                    stderr_text = cached_stderr
                else:
                    stderr_text = _read_proc_output(proc, "stderr")
                exit_code = proc.returncode
                if cached_status is not None:
                    status = cached_status  # type: ignore[assignment]
                else:
                    status = classify_colab_output(proc)

            (out / "stdout.log").write_text(stdout_text, encoding="utf-8")
            (out / "stderr.log").write_text(stderr_text, encoding="utf-8")

            result_meta: dict[str, Any] = {
                "ok": status == COLAB_COMPLETE,
                "error": None if status == COLAB_COMPLETE else status,
                "exit_code": exit_code,
                "status": status,
                "session": name,
            }
            (out / "result.json").write_text(
                json.dumps(result_meta, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            return {"ok": result_meta["ok"], "error": result_meta["error"]}
        finally:
            # F1: clean up scheduler-owned temp files after collection.
            if proc is not None:
                _cleanup_proc_tempfiles(proc)

    # -- stop -------------------------------------------------------------

    def stop(self, name: str, *, timeout: float | None = None) -> None:
        """Stop a named Colab session (best-effort, no error on missing).

        Also cleans up scheduler-owned temp files (F1) via :meth:`forget`.
        """
        to = timeout if timeout is not None else self.stop_timeout
        try:
            self._runner(["colab", "stop", "--session", name], to)
        except (RuntimeError, subprocess.TimeoutExpired):
            # Best-effort: the session may already be gone, or the CLI may
            # time out during teardown.  Either way, the scheduler must not
            # fail because stop() raised.
            pass
        finally:
            # Clean up internal proc tracking and temp files.
            self.forget(name)

    # -- internal proc tracking -------------------------------------------
    # The scheduler calls run() -> Popen, then classify_colab_output(proc),
    # then collect(name, output_dir).  We need to associate the Popen with
    # the session name for collect().  The scheduler could pass the proc to
    # collect, but the protocol signature is collect(name, output_dir) to
    # mirror the Kaggle client's output(kernel_id, output_dir).  So we track
    # procs internally by session name.
    #
    # This map is populated by run() and depopulated by stop() or
    # forget().  It is NOT durable across restarts — on restart, the
    # scheduler detects orphaned handles (running Colab job with no local
    # proc) and calls stop(name) directly.

    @property
    def _procs(self) -> dict[str, subprocess.Popen]:
        # Lazy-init on the instance (not the class) to avoid shared state
        # across instances in tests.
        procs = self.__dict__.get("_colab_procs")
        if procs is None:
            procs = {}
            self.__dict__["_colab_procs"] = procs
        return procs

    def remember(self, name: str, proc: subprocess.Popen) -> None:
        """Associate a Popen with a session name for later collect().

        Called by the scheduler after :meth:`run` so that :meth:`collect`
        can retrieve the process handle by session name.  The scheduler may
        also call :func:`classify_colab_output` on the proc before collect;
        the cached output is then available to collect without re-reading
        exhausted pipes.
        """
        self._procs[name] = proc

    def forget(self, name: str) -> subprocess.Popen | None:
        """Remove a session's Popen from internal tracking and clean temp files.

        Returns the removed Popen, or None if not tracked.  Called by
        :meth:`stop` and :meth:`collect` (via cleanup) when a job is retired.
        Cleans up scheduler-owned temp files (F1) on the removed Popen.
        """
        proc = self._procs.pop(name, None)
        if proc is not None:
            _cleanup_proc_tempfiles(proc)
        return proc


# ---------------------------------------------------------------------------
# Orphaned handle detection (restart recovery)
# ---------------------------------------------------------------------------


def detect_orphaned_sessions(
    client: ColabClient,
    known_names: set[str],
    *,
    timeout: float | None = None,
) -> list[str]:
    """Detect Colab sessions that exist on the backend but have no local Popen.

    After a scheduler restart, the scheduler has durable state recording which
    Colab jobs were ``running`` (with ``remote_id`` = session name) but no
    local process handle.  This function cross-references the backend's active
    sessions (via ``client.sessions()``) against the set of session names the
    scheduler still considers live (``known_names``).

    Returns a list of session names that are active on the backend and in
    ``known_names`` but for which the scheduler has no local Popen (i.e.,
    every name in ``known_names`` that also appears in the backend session
    list — the scheduler calls this only when it has no local handle for
    those jobs).

    The scheduler should:
      1. For each orphaned name, mark the job as ``failed`` (interrupted).
      2. Best-effort call ``client.stop(name)`` to tear down the session.
      3. Never silently resubmit or claim success for an orphaned job.
    """
    active = client.sessions(timeout=timeout)
    active_names = {s["name"] for s in active}
    # Orphaned: known to scheduler but the scheduler has no local proc.
    # The scheduler passes only the names it cannot reconcile with a local
    # Popen; we return those that are still live on the backend.
    return sorted(known_names & active_names)


# ---------------------------------------------------------------------------
# CLI (smoke-test entry point)
# ---------------------------------------------------------------------------


def _main() -> int:
    """Minimal smoke-test entry: list sessions and print count."""
    client = ColabCliClient()
    sessions = client.sessions()
    print(f"Active Colab sessions: {len(sessions)}")
    for s in sessions:
        print(f"  {s['name']}: {s['state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
