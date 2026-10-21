"""Event hooks for the Oczy Kaggle parallel scheduler (new module; no frozen sources touched).

``attach_hooks(scheduler, ...)`` wraps one ``ParallelScheduler`` instance so that real
state transitions emit events to subscribers:

    scheduled, submitted, collecting, succeeded, failed, collected,
    lease_acquired, lease_released, batch_finished, scheduler_error

Note: the scheduler's job state has no separate "queued/running on Kaggle" step, so
``submitted`` means the push succeeded and the job entered ``running``; there is no
distinct ``running`` event.

Guarantees
* Hooks never raise into, or block, the scheduler: observation is a local append plus an
  enqueue; delivery runs on a daemon thread; every hook-side exception is swallowed.
* At-least-once: every (event, subscription) delivery is persisted in ``outbox.json`` before
  it is attempted, retried with backoff, then dead-lettered. Event ids are deterministic
  (state file + job + attempt + type), so consumers can dedupe, and re-observation after a
  restart never creates a second event.
* Subscriptions live in ``<state>.hooks/subscriptions.json`` and survive restarts.

Sinks: ``jsonl`` (append-only file), ``command`` (argv, event JSON on stdin), ``webhook``
(HTTPS POST; URL/key from an env-style config file, never from code), ``inbox`` (one JSON
file per event in a directory a local notifier can poll).

CLI: ``python scheduler_hooks.py flush <state.json.hooks dir>``  redeliver pending;
     ``python scheduler_hooks.py inbox <dir> [--ack]``           list/ack inbox events.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

EVENT_SCHEMA = "oczy/scheduler-event/v1"
SUBS_SCHEMA = "oczy/scheduler-hook-subscriptions/v1"
EVENT_TYPES = ("scheduled", "submitted", "collecting", "succeeded", "failed", "collected",
               "lease_acquired", "lease_released", "batch_finished", "scheduler_error")
SINK_TYPES = ("jsonl", "command", "webhook", "inbox")
DEFAULT_BACKOFF = (1.0, 5.0, 30.0, 120.0, 600.0, 1800.0, 3600.0)
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_WEBHOOK_ENV = "~/.config/worldsim-webhook.env"
DEFAULT_INBOX = "~/.local/state/oczy/notify-inbox"
_SECRET_KEYS = ("url", "key", "token", "authorization")


class HookConfigError(ValueError):
    pass


# ----------------------------------------------------------------- helpers
def _canon(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=str)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_line(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[7:].strip()
        out[k] = v.strip().strip("'\"")
    return out


def validate_subscription(sub: Mapping[str, Any]) -> dict[str, Any]:
    sink = sub.get("sink")
    if not isinstance(sink, Mapping) or sink.get("type") not in SINK_TYPES:
        raise HookConfigError(f"sink.type must be one of {SINK_TYPES}")
    events = sub.get("events", ["*"])
    if not isinstance(events, list) or not events:
        raise HookConfigError("events must be a non-empty list")
    for e in events:
        if e != "*" and e not in EVENT_TYPES:
            raise HookConfigError(f"unknown event type {e!r}")
    jobs = sub.get("jobs", ["*"])
    if not isinstance(jobs, list) or not all(isinstance(j, str) for j in jobs):
        raise HookConfigError("jobs must be a list of name globs")
    t = sink["type"]
    if t in ("jsonl", "inbox") and not isinstance(sink.get("path", DEFAULT_INBOX if t == "inbox" else None), str):
        raise HookConfigError(f"{t} sink needs a string path")
    if t == "command":
        c = sink.get("argv")
        if not (isinstance(c, list) and c and all(isinstance(a, str) for a in c)):
            raise HookConfigError("command sink needs a non-empty argv list (no shell)")
    if t == "webhook":
        if "url" in sink or "key" in sink:
            raise HookConfigError("webhook url/key must come from config_file, never inline")
    out = {"events": list(events), "jobs": list(jobs), "sink": dict(sink)}
    out["id"] = hashlib.sha256(_canon(out).encode()).hexdigest()[:12]
    return out


def parse_hook_arg(spec: str) -> dict[str, Any]:
    """``--hook`` value: a JSON object, or shorthand ``type:arg`` (arg = path/argv/env-file)."""
    spec = spec.strip()
    if spec.startswith("{"):
        return validate_subscription(json.loads(spec))
    t, _, arg = spec.partition(":")
    if t == "command":
        return validate_subscription({"sink": {"type": "command", "argv": shlex.split(arg)}})
    if t == "webhook":
        return validate_subscription({"sink": {"type": "webhook", "config_file": arg or DEFAULT_WEBHOOK_ENV}})
    if t in ("jsonl", "inbox"):
        return validate_subscription({"sink": {"type": t, "path": arg or DEFAULT_INBOX}})
    raise HookConfigError(f"bad --hook {spec!r}; use JSON or one of {SINK_TYPES} as type:arg")


def default_subscriptions() -> list[dict[str, Any]]:
    """Inbox + optional webhook; used by callers (e.g. masisim) that want notification by default."""
    return [
        {"events": ["succeeded", "failed", "batch_finished", "scheduler_error"],
         "sink": {"type": "inbox", "path": DEFAULT_INBOX}},
        {"events": ["succeeded", "failed", "batch_finished", "scheduler_error"],
         "sink": {"type": "webhook", "config_file": DEFAULT_WEBHOOK_ENV, "optional": True}},
    ]


# ------------------------------------------------------------------- sinks
def _redact(s: str, secrets: list[str]) -> str:
    for x in secrets:
        if x:
            s = s.replace(x, "***")
    return s


class DeliveryError(Exception):
    pass


def deliver(sink: Mapping[str, Any], event: Mapping[str, Any], *, timeout: float = 15.0) -> str:
    """Deliver one event; return 'sent' or 'skipped'; raise DeliveryError on failure."""
    t = sink["type"]
    body = json.dumps(event, sort_keys=True, default=str)
    if t == "jsonl":
        _append_line(Path(sink["path"]).expanduser(), event)
        return "sent"
    if t == "inbox":
        d = Path(sink.get("path", DEFAULT_INBOX)).expanduser()
        name = f"{int(event['ts_epoch']):012d}-{event['event_id']}.json"
        if not (d / name).exists() and not (d / "processed" / name).exists():
            _atomic_write(d / name, json.dumps(event, indent=2, sort_keys=True, default=str) + "\n")
        return "sent"
    if t == "command":
        env = dict(os.environ, OCZY_HOOK_EVENT=event["type"], OCZY_HOOK_EVENT_ID=event["event_id"],
                   OCZY_HOOK_JOB=str((event.get("job") or {}).get("name", "")))
        try:
            r = subprocess.run(sink["argv"], input=body, text=True, capture_output=True,
                               timeout=float(sink.get("timeout", 30)), env=env, shell=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeliveryError(f"command failed to run: {type(exc).__name__}") from exc
        if r.returncode != 0:
            raise DeliveryError(f"command exit {r.returncode}: {r.stderr[-300:]}")
        return "sent"
    if t == "webhook":
        cfg_path = Path(sink["config_file"]).expanduser()
        if not cfg_path.is_file():
            if sink.get("optional"):
                return "skipped"
            raise DeliveryError("webhook config file missing")
        if cfg_path.stat().st_mode & 0o077:
            raise DeliveryError("webhook config file must be chmod 600")
        cfg = parse_env_file(cfg_path)
        url = cfg.get(sink.get("url_var", "WORLDSIM_WEBHOOK_URL"), "")
        key = cfg.get(sink.get("key_var", "WORLDSIM_WEBHOOK_KEY"), "")
        if not url:  # template/unfilled file: not configured, not an error
            return "skipped"
        if not (url.startswith("https://") or url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
            raise DeliveryError("webhook url must be https (or localhost)")
        headers = {"Content-Type": "application/json", "Idempotency-Key": event["event_id"],
                   "X-Oczy-Event": event["type"]}
        if key:
            hdr = sink.get("key_header") or cfg.get("WORLDSIM_WEBHOOK_HEADER") or "Authorization"
            scheme = sink.get("key_scheme") or cfg.get("WORLDSIM_WEBHOOK_SCHEME") or ("Bearer" if hdr == "Authorization" else "")
            headers[hdr] = f"{scheme} {key}".strip()
        req = urllib.request.Request(url, data=body.encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - https enforced above
                if not 200 <= resp.status < 300:
                    raise DeliveryError(f"webhook HTTP {resp.status}")
        except urllib.error.HTTPError as exc:
            raise DeliveryError(f"webhook HTTP {exc.code}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise DeliveryError(_redact(f"webhook transport error: {type(exc).__name__}", [url, key])) from None
        return "sent"
    raise DeliveryError(f"unknown sink {t}")


# -------------------------------------------------------------- dispatcher
class HookDispatcher:
    """Durable outbox + background delivery for one scheduler state file."""

    def __init__(self, hooks_dir: Path, *, backoff=DEFAULT_BACKOFF, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 clock: Callable[[], float] = time.time, deliver_fn=deliver, start_thread: bool = True):
        self.dir = Path(hooks_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backoff, self.max_attempts, self.clock, self.deliver_fn = tuple(backoff), max_attempts, clock, deliver_fn
        self._lock = threading.RLock()
        self._dlock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._subs: dict[str, dict[str, Any]] = {}
        self._outbox: dict[str, dict[str, Any]] = {}
        self._emitted: set[str] = set()
        self._load()
        self._thread = None
        if start_thread:
            self._thread = threading.Thread(target=self._loop, name="oczy-hooks", daemon=True)
            self._thread.start()

    # persistence
    def _load(self) -> None:
        try:
            d = json.loads((self.dir / "subscriptions.json").read_text())
            self._subs = {s["id"]: s for s in d.get("subscriptions", [])}
        except (OSError, ValueError, KeyError):
            self._subs = {}
        try:
            self._outbox = json.loads((self.dir / "outbox.json").read_text()).get("pending", {})
        except (OSError, ValueError):
            self._outbox = {}
        try:
            for line in (self.dir / "events.jsonl").read_text().splitlines():
                try:
                    self._emitted.add(json.loads(line)["event_id"])
                except (ValueError, KeyError):
                    pass
        except OSError:
            pass

    def _save_outbox(self) -> None:
        _atomic_write(self.dir / "outbox.json", json.dumps({"pending": self._outbox}, sort_keys=True))

    def subscribe(self, subs: list[Mapping[str, Any]]) -> None:
        with self._lock:
            for s in subs:
                v = validate_subscription(s)
                self._subs[v["id"]] = v
            _atomic_write(self.dir / "subscriptions.json", json.dumps(
                {"schema_version": SUBS_SCHEMA, "subscriptions": list(self._subs.values())}, indent=2, sort_keys=True))

    # emit (called on the scheduler thread; must be cheap and safe)
    def emit(self, event: dict[str, Any]) -> bool:
        with self._lock:
            if event["event_id"] in self._emitted:
                return False
            self._emitted.add(event["event_id"])
            _append_line(self.dir / "events.jsonl", event)
            name = str((event.get("job") or {}).get("name", ""))
            for s in self._subs.values():
                if not any(e in ("*", event["type"]) for e in s["events"]):
                    continue
                if event.get("job") and not any(fnmatch.fnmatchcase(name, g) for g in s["jobs"]):
                    continue
                self._outbox[f"{event['event_id']}:{s['id']}"] = {
                    "event": event, "sub": s["id"], "attempts": 0, "next_at": 0.0, "last_error": None}
            self._save_outbox()
        self._wake.set()
        return True

    # delivery
    def _due(self, now: float):
        return [k for k, v in self._outbox.items() if v["next_at"] <= now]

    def deliver_due(self) -> int:
        """Serialized: only one delivery pass at a time, so a pass never duplicates another's work."""
        with self._dlock:
            return self._deliver_due()

    def _deliver_due(self) -> int:
        n = 0
        with self._lock:
            keys = self._due(self.clock())
        for k in keys:
            with self._lock:
                ent = self._outbox.get(k)
                sub = self._subs.get(ent["sub"]) if ent else None
            if ent is None:
                continue
            try:
                if sub is None:
                    raise DeliveryError("subscription removed")
                self.deliver_fn(sub["sink"], ent["event"])
                with self._lock:
                    self._outbox.pop(k, None)
                n += 1
            except Exception as exc:  # delivery must never escape
                msg = _redact(f"{type(exc).__name__}: {exc}", [])[:400]
                with self._lock:
                    ent["attempts"] += 1
                    ent["last_error"] = msg
                    if ent["attempts"] >= self.max_attempts:
                        self._outbox.pop(k, None)
                        _append_line(self.dir / "dead_letter.jsonl", {
                            "dead_at": self.clock(), "event_id": ent["event"]["event_id"], "sub": ent["sub"],
                            "sink_type": sub["sink"]["type"] if sub else None, "attempts": ent["attempts"],
                            "last_error": msg, "event": ent["event"]})
                    else:
                        ent["next_at"] = self.clock() + self.backoff[min(ent["attempts"] - 1, len(self.backoff) - 1)]
            with self._lock:
                self._save_outbox()
        return n

    def _loop(self) -> None:
        while not self._stop:
            try:
                self.deliver_due()
            except Exception:
                pass
            self._wake.wait(timeout=1.0)
            self._wake.clear()

    def drain(self, timeout: float = 20.0) -> int:
        """Best-effort flush at exit: retry only what is due within ``timeout``; rest stays persisted."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                self.deliver_due()
            except Exception:
                break
            with self._lock:
                soon = [v for v in self._outbox.values() if v["next_at"] <= self.clock() + 2.0]
            if not soon:
                break
            time.sleep(0.25)
        with self._lock:
            return len(self._outbox)

    def close(self) -> None:
        self._stop = True
        self._wake.set()


# ------------------------------------------------------------ event building
def _iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(etype: str, *, state_path: str, batch_path: str, job: Mapping[str, Any] | None,
               now: float, extra: Mapping[str, Any] | None = None, key_extra: str = "") -> dict[str, Any]:
    j = None
    if job is not None:
        j = {k: job.get(k) for k in ("name", "kernel_id", "provider", "account_id", "attempts", "state", "error",
                                     "output_dir", "submitted_at", "completed_at", "collected_at",
                                     "runtime_manifest_verified")}
    ident = _canon([state_path, (j or {}).get("name"), (j or {}).get("attempts"), etype, key_extra])
    ev = {"schema_version": EVENT_SCHEMA, "event_id": hashlib.sha256(ident.encode()).hexdigest()[:32],
          "type": etype, "ts": _iso(now), "ts_epoch": now, "state_path": state_path, "batch_path": batch_path,
          "job": j}
    if extra:
        ev["detail"] = dict(extra)
    return ev


_STATE_EVENTS = {"running": "submitted", "collecting": "collecting", "succeeded": "succeeded", "failed": "failed"}


class SchedulerObserver:
    """Turns persisted-state transitions and lease calls into events; never raises."""

    def __init__(self, dispatcher: HookDispatcher, *, clock: Callable[[], float] = time.time):
        self.d, self.clock = dispatcher, clock
        self.state_path = ""
        self.batch_path = ""
        self._seen: dict[str, tuple] = {}

    def _emit(self, *a, **k) -> None:
        try:
            self.d.emit(make_event(*a, state_path=self.state_path, batch_path=self.batch_path, now=self.clock(), **k))
        except Exception:
            pass

    def on_save(self, jobs: Mapping[str, Any]) -> None:
        try:
            for name, job in jobs.items():
                jd = job.to_dict() if hasattr(job, "to_dict") else dict(job)
                jd["attempts"] = jd.get("attempts", 0)
                sig = (jd["state"], jd["attempts"], bool(jd.get("collected_at")))
                prev = self._seen.get(name)
                if prev is None:
                    self._emit("scheduled", job=jd)  # first sight; id dedupes across restarts
                if prev is None or prev[:2] != sig[:2]:
                    ev = _STATE_EVENTS.get(jd["state"])
                    if ev and not (prev is None and jd["state"] == "pending"):
                        self._emit(ev, job=jd)
                if jd.get("collected_at") and (prev is None or not prev[2]):
                    self._emit("collected", job=jd)
                self._seen[name] = sig
        except Exception:
            pass

    def on_lease(self, kind: str, job: Any, ok: bool = True) -> None:
        try:
            if not ok:
                return
            jd = job.to_dict() if hasattr(job, "to_dict") else dict(job)
            self._emit(kind, job=jd)
        except Exception:
            pass

    def on_finish(self, summary: Mapping[str, Any] | None, error: BaseException | None) -> None:
        try:
            if error is not None:
                self._emit("scheduler_error", job=None, extra={"error": f"{type(error).__name__}: {error}"[:400]},
                           key_extra=str(int(self.clock())))
            elif summary is not None:
                self._emit("batch_finished", job=None, extra={k: summary.get(k) for k in
                           ("total", "succeeded", "failed", "all_succeeded") if k in summary},
                           key_extra=_canon(sorted((n, v.get("state"), v.get("attempts")) for n, v in
                                                   (summary.get("jobs") or {}).items())) if isinstance(
                               summary.get("jobs"), Mapping) else "")
        except Exception:
            pass


# ------------------------------------------------------------------ attach
def hooks_dir_for(state_path: str | Path) -> Path:
    p = Path(state_path).expanduser().resolve()
    return p.with_name(p.name + ".hooks")


def batch_subscriptions(batch_path: str | Path) -> list[dict[str, Any]]:
    """Subscriptions declared in the batch: top-level ``hooks`` plus per-job ``hooks`` (scoped to that job)."""
    out: list[dict[str, Any]] = []
    try:
        data = json.loads(Path(batch_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for s in data.get("hooks") or []:
        out.append(validate_subscription(s))
    for entry in data.get("jobs") or []:
        for s in (entry.get("hooks") or []) if isinstance(entry, dict) else []:
            out.append(validate_subscription({**s, "jobs": [entry["name"]]}))
    return out


def attach_hooks(scheduler: Any, *, cli_hooks: list[Mapping[str, Any]] | None = None,
                 drain_timeout: float = 20.0, dispatcher_factory=HookDispatcher) -> Any:
    """Wrap ``scheduler.run`` and the lease methods of this instance. Safe if anything here fails."""
    orig_run, orig_save = scheduler.run, scheduler._save_state
    orig_acq, orig_rel = scheduler._acquire_job_lease, scheduler._release_job_lease
    ctx: dict[str, Any] = {"obs": None}

    def run(batch_path, state_path, **kw):
        obs = disp = None
        try:
            disp = dispatcher_factory(hooks_dir_for(state_path))
            disp.subscribe(list(batch_subscriptions(batch_path)) + list(cli_hooks or []))
            obs = SchedulerObserver(disp, clock=getattr(scheduler, "clock", time.time))
            obs.state_path, obs.batch_path = str(Path(state_path).expanduser().resolve()), str(
                Path(batch_path).expanduser().resolve())
            ctx["obs"] = obs
        except Exception as exc:  # misconfigured hooks must not stop the run
            print(f"[hooks] disabled: {type(exc).__name__}: {exc}", file=sys.stderr)
            ctx["obs"] = obs = None
        summary, err = None, None
        try:
            summary = orig_run(batch_path, state_path, **kw)
            return summary
        except BaseException as exc:
            err = exc
            raise
        finally:
            if obs is not None:
                obs.on_finish(summary, err)
                try:
                    left = disp.drain(drain_timeout)
                    if left:
                        print(f"[hooks] {left} deliveries pending; retry with: python scheduler_hooks.py flush "
                              f"{disp.dir}", file=sys.stderr)
                    disp.close()
                except Exception:
                    pass
            ctx["obs"] = None

    def save(state_path, batch_path, jobs):
        orig_save(state_path, batch_path, jobs)
        if ctx["obs"] is not None:
            ctx["obs"].on_save(jobs)

    def acquire(job):
        ok = orig_acq(job)
        if ctx["obs"] is not None and getattr(scheduler, "lease_store", None) is not None and job.account_id:
            ctx["obs"].on_lease("lease_acquired", job, bool(ok))
        return ok

    def release(job):
        orig_rel(job)
        if ctx["obs"] is not None and getattr(scheduler, "lease_store", None) is not None and job.account_id:
            ctx["obs"].on_lease("lease_released", job)

    scheduler.run, scheduler._save_state = run, save
    scheduler._acquire_job_lease, scheduler._release_job_lease = acquire, release
    return scheduler


# --------------------------------------------------------------------- CLI
def list_inbox(d: Path, ack: bool = False) -> list[dict[str, Any]]:
    out = []
    for p in sorted(d.expanduser().glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
        if ack:
            (d.expanduser() / "processed").mkdir(exist_ok=True)
            os.replace(p, d.expanduser() / "processed" / p.name)
    return out


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) >= 2 and a[0] == "flush":
        d = HookDispatcher(Path(a[1]), start_thread=False)
        n = d.drain(30.0)
        print(json.dumps({"still_pending": n}))
        return 0 if n == 0 else 1
    if len(a) >= 2 and a[0] == "inbox":
        for ev in list_inbox(Path(a[1]), ack="--ack" in a):
            print(json.dumps(ev, sort_keys=True))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
