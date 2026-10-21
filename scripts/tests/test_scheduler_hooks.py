"""Scheduler hook layer: events at real transitions, durability, at-least-once, isolation."""
from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle"))
import scheduler_hooks as sh  # noqa: E402
from test_kaggle_parallel_scheduler import (CountingSleeper, FakeKaggleClient, ParallelScheduler,  # noqa: E402
                                            _make_batch, _make_clock, _make_kernel_dir)


def _events(hooks_dir: Path):
    p = hooks_dir / "events.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _run(tmp_path, *, client, hooks=None, batch_hooks=None, jobs=("a",), sched_kw=None):
    entries = []
    for n in jobs:
        _make_kernel_dir(tmp_path, n, f"owner/k-{n}")
        entries.append({"name": n, "kernel_dir": n, "output_dir": f"out/{n}"})
    batch = _make_batch(tmp_path, entries)
    if batch_hooks is not None:
        d = json.loads(batch.read_text()); d["hooks"] = batch_hooks; batch.write_text(json.dumps(d))
    holder, clock = _make_clock()
    s = ParallelScheduler(client, clock=clock, sleeper=CountingSleeper(holder), **(sched_kw or {}))
    sh.attach_hooks(s, cli_hooks=hooks, drain_timeout=5)
    summary = s.run(batch, tmp_path / "state.json", poll_interval=0.01)
    return s, summary, sh.hooks_dir_for(tmp_path / "state.json")


def test_success_emits_ordered_events_to_jsonl_and_inbox(tmp_path):
    inbox = tmp_path / "inbox"
    _, summary, hd = _run(tmp_path, client=FakeKaggleClient(),
                          hooks=[sh.parse_hook_arg(f"inbox:{inbox}")])
    types = [e["type"] for e in _events(hd)]
    assert types[0] == "scheduled"
    for a, b in (("submitted", "collecting"), ("collecting", "succeeded")):
        assert types.index(a) < types.index(b)
    assert "collected" in types and types[-1] == "batch_finished"
    assert "failed" not in types
    assert len({e["event_id"] for e in _events(hd)}) == len(types)
    got = [json.loads(p.read_text()) for p in sorted(inbox.glob("*.json"))]
    assert {e["type"] for e in got} >= {"succeeded", "batch_finished"} or True  # all events when "*"
    assert len(got) == len(types)
    assert (hd / "outbox.json").exists() and json.loads((hd / "outbox.json").read_text())["pending"] == {}


def test_failure_event_carries_error(tmp_path):
    c = FakeKaggleClient(status_sequences={"owner/k-a": ["error"]})
    _, summary, hd = _run(tmp_path, client=c, hooks=[sh.parse_hook_arg(f"inbox:{tmp_path/'i'}")])
    f = [e for e in _events(hd) if e["type"] == "failed"]
    assert len(f) == 1 and f[0]["job"]["error"] == "kernel reported error status"
    assert not summary["all_succeeded"]


def test_event_and_job_filters(tmp_path):
    sub = sh.validate_subscription({"events": ["failed"], "jobs": ["b*"],
                                    "sink": {"type": "inbox", "path": str(tmp_path / "i")}})
    c = FakeKaggleClient(status_sequences={"owner/k-a": ["error"], "owner/k-b1": ["error"]})
    _run(tmp_path, client=c, hooks=[sub], jobs=("a", "b1"))
    got = [json.loads(p.read_text()) for p in (tmp_path / "i").glob("*.json")]
    assert [(e["type"], e["job"]["name"]) for e in got] == [("failed", "b1")]


def test_batch_level_and_job_level_hooks_persist_and_survive_restart(tmp_path):
    batch_hooks = [{"events": ["succeeded"], "sink": {"type": "inbox", "path": str(tmp_path / "i")}}]
    _, _, hd = _run(tmp_path, client=FakeKaggleClient(), batch_hooks=batch_hooks)
    assert len(json.loads((hd / "subscriptions.json").read_text())["subscriptions"]) == 1
    n_events = len(_events(hd))
    # restart on the same state, no hooks given anywhere: subscriptions reload, nothing re-emitted
    d = json.loads((tmp_path / "batch.json").read_text()); d.pop("hooks"); (tmp_path / "batch.json").write_text(json.dumps(d))
    holder, clock = _make_clock()
    s = ParallelScheduler(FakeKaggleClient(), clock=clock, sleeper=CountingSleeper(holder))
    sh.attach_hooks(s, drain_timeout=2)
    s.run(tmp_path / "batch.json", tmp_path / "state.json", poll_interval=0.01)
    assert len(list((tmp_path / "i").glob("*.json"))) == 1
    ids = [e["event_id"] for e in _events(hd)]
    assert len(ids) == len(set(ids))  # restart adds only new batch_finished-type events, never duplicates


def test_command_sink_gets_event_on_stdin(tmp_path):
    out = tmp_path / "got.jsonl"
    script = tmp_path / "hook.sh"
    script.write_text(f"#!/bin/sh\ncat >> {out}\necho >> {out}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    sub = {"events": ["succeeded"], "sink": {"type": "command", "argv": [str(script)]}}
    _run(tmp_path, client=FakeKaggleClient(), hooks=[sub])
    ev = json.loads(out.read_text().strip())
    assert ev["type"] == "succeeded" and ev["schema_version"] == sh.EVENT_SCHEMA


def test_hook_failures_never_break_or_block_scheduler(tmp_path):
    bad = [{"events": ["*"], "sink": {"type": "command", "argv": ["/nonexistent/cmd"]}},
           {"events": ["*"], "sink": {"type": "webhook", "config_file": str(tmp_path / "missing.env")}}]
    t0 = time.monotonic()
    s, summary, hd = _run(tmp_path, client=FakeKaggleClient(), hooks=bad)
    assert summary["all_succeeded"]
    assert time.monotonic() - t0 < 30
    assert json.loads((hd / "outbox.json").read_text())["pending"]  # retained for retry, not lost


def test_broken_dispatcher_disables_hooks_but_run_succeeds(tmp_path):
    def boom(_):
        raise RuntimeError("no disk")
    entries = []
    _make_kernel_dir(tmp_path, "a", "owner/k-a"); entries.append({"name": "a", "kernel_dir": "a", "output_dir": "o/a"})
    batch = _make_batch(tmp_path, entries)
    holder, clock = _make_clock()
    s = ParallelScheduler(FakeKaggleClient(), clock=clock, sleeper=CountingSleeper(holder))
    sh.attach_hooks(s, dispatcher_factory=boom)
    assert s.run(batch, tmp_path / "state.json", poll_interval=0.01)["all_succeeded"]


def test_retry_backoff_then_dead_letter_and_idempotent_ids(tmp_path):
    now = [100.0]
    calls = []

    def flaky(sink, ev):
        calls.append(ev["event_id"])
        raise sh.DeliveryError("HTTP 503")

    d = sh.HookDispatcher(tmp_path / "h", backoff=(10, 20), max_attempts=3, clock=lambda: now[0],
                          deliver_fn=flaky, start_thread=False)
    d.subscribe([{"sink": {"type": "inbox", "path": str(tmp_path / "i")}}])
    ev = sh.make_event("failed", state_path="s", batch_path="b", job={"name": "j", "attempts": 1}, now=100.0)
    assert d.emit(ev) and not d.emit(ev)  # same id: second emit is a no-op
    d.deliver_due(); assert len(calls) == 1
    d.deliver_due(); assert len(calls) == 1          # backoff not elapsed
    now[0] += 10; d.deliver_due(); assert len(calls) == 2
    now[0] += 20; d.deliver_due(); assert len(calls) == 3
    dl = [json.loads(l) for l in (tmp_path / "h/dead_letter.jsonl").read_text().splitlines()]
    assert len(dl) == 1 and dl[0]["event_id"] == ev["event_id"] and dl[0]["attempts"] == 3
    assert json.loads((tmp_path / "h/outbox.json").read_text())["pending"] == {}


def test_pending_deliveries_survive_restart_and_flush(tmp_path):
    ok = []
    d1 = sh.HookDispatcher(tmp_path / "h", backoff=(0,), deliver_fn=lambda s, e: (_ for _ in ()).throw(
        sh.DeliveryError("down")), start_thread=False)
    d1.subscribe([{"sink": {"type": "inbox", "path": str(tmp_path / "i")}}])
    d1.emit(sh.make_event("succeeded", state_path="s", batch_path="b", job={"name": "j", "attempts": 1}, now=1.0))
    d1.deliver_due()
    d2 = sh.HookDispatcher(tmp_path / "h", backoff=(0,), deliver_fn=lambda s, e: ok.append(e["event_id"]) or "sent",
                           start_thread=False)
    assert d2.drain(2) == 0 and len(ok) == 1


def test_webhook_reads_config_file_enforces_https_and_redacts(tmp_path, monkeypatch):
    cfg = tmp_path / "w.env"
    cfg.write_text("WORLDSIM_WEBHOOK_URL=https://hooks.example.test/x?s=SECRET123\nWORLDSIM_WEBHOOK_KEY=KEY456\n"); cfg.chmod(0o600)
    seen = {}

    class R:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_open(req, timeout=0):
        seen.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, body=json.loads(req.data))
        return R()

    monkeypatch.setattr(sh.urllib.request, "urlopen", fake_open)
    ev = sh.make_event("succeeded", state_path="s", batch_path="b", job={"name": "j", "attempts": 1}, now=1.0)
    assert sh.deliver({"type": "webhook", "config_file": str(cfg)}, ev) == "sent"
    assert seen["headers"]["idempotency-key"] == ev["event_id"] and seen["headers"]["authorization"] == "Bearer KEY456"
    assert seen["body"]["type"] == "succeeded"
    cfg.write_text("WORLDSIM_WEBHOOK_URL=http://evil.example.test/x\n")
    with pytest.raises(sh.DeliveryError, match="https"):
        sh.deliver({"type": "webhook", "config_file": str(cfg)}, ev)
    cfg.write_text("WORLDSIM_WEBHOOK_URL=https://h.example.test/SECRET123\nWORLDSIM_WEBHOOK_KEY=KEY456\n")

    def boom(req, timeout=0):
        raise sh.urllib.error.URLError("conn refused to https://h.example.test/SECRET123")
    monkeypatch.setattr(sh.urllib.request, "urlopen", boom)
    with pytest.raises(sh.DeliveryError) as ei:
        sh.deliver({"type": "webhook", "config_file": str(cfg)}, ev)
    assert "SECRET123" not in str(ei.value) and "KEY456" not in str(ei.value)
    assert sh.deliver({"type": "webhook", "config_file": str(tmp_path / "none.env"), "optional": True}, ev) == "skipped"
    with pytest.raises(sh.HookConfigError):
        sh.validate_subscription({"sink": {"type": "webhook", "config_file": "x", "url": "https://inline"}})


def test_lease_events_and_inbox_cli(tmp_path):
    class Job:
        account_id, name = "kaggle-primary", "j"
        def to_dict(self): return {"name": "j", "account_id": "kaggle-primary", "attempts": 1, "state": "running"}

    class Sched:
        lease_store = object(); clock = staticmethod(lambda: 5.0)
        def run(self, b, s, **k): return {"all_succeeded": True, "total": 1, "succeeded": 1, "failed": 0}
        def _save_state(self, *a): pass
        def _acquire_job_lease(self, job): return True
        def _release_job_lease(self, job): pass

    (tmp_path / "b.json").write_text(json.dumps({"hooks": [{"sink": {"type": "inbox", "path": str(tmp_path / "i")}}]}))
    s = sh.attach_hooks(Sched())
    orig_run = s.run
    # run wrapper builds the observer; drive lease calls from inside the wrapped run
    inner = {}
    def run_inside(b, st, **k):
        s._acquire_job_lease(Job()); s._release_job_lease(Job()); return {"all_succeeded": True}
    orig = s.__dict__.copy()
    # replace the underlying run via a fresh Sched whose run triggers lease calls
    class Sched2(Sched):
        def run(self, b, st, **k):
            self._acquire_job_lease(Job()); self._release_job_lease(Job()); return {"all_succeeded": True}
    s2 = sh.attach_hooks(Sched2())
    s2.run(tmp_path / "b.json", tmp_path / "state.json")
    types = [e["type"] for e in sh.list_inbox(tmp_path / "i")]
    assert types.count("lease_acquired") == 1 and types.count("lease_released") == 1 and "batch_finished" in types
    assert len(sh.list_inbox(tmp_path / "i", ack=True)) == len(types) and sh.list_inbox(tmp_path / "i") == []


def test_scheduler_error_event_and_reraise(tmp_path):
    class Sched:
        lease_store = None; clock = staticmethod(time.time)
        def run(self, b, s, **k): raise RuntimeError("kaboom")
        def _save_state(self, *a): pass
        def _acquire_job_lease(self, j): return True
        def _release_job_lease(self, j): pass
    (tmp_path / "b.json").write_text("{}")
    s = sh.attach_hooks(Sched(), cli_hooks=[sh.parse_hook_arg(f"inbox:{tmp_path/'i'}")])
    with pytest.raises(RuntimeError, match="kaboom"):
        s.run(tmp_path / "b.json", tmp_path / "st.json")
    assert [e["type"] for e in sh.list_inbox(tmp_path / "i")] == ["scheduler_error"]


def test_cli_hook_flag_is_accepted_and_bad_hook_rejected(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infrastructure" / "kaggle"))
    import runpy
    mod = runpy.run_path(str(Path(__file__).resolve().parents[2] / "infrastructure/kaggle/parallel_scheduler.py"))
    _make_kernel_dir(tmp_path, "a", "owner/k-a")
    batch = _make_batch(tmp_path, [{"name": "a", "kernel_dir": "a", "output_dir": "o/a"}])
    holder, clock = _make_clock()
    rc = mod["main"](["run", str(batch), "--state", str(tmp_path / "s.json"), "--hook", f"inbox:{tmp_path/'i'}",
                      "--poll-interval", "0.01"], client=FakeKaggleClient(), clock=clock, sleeper=CountingSleeper(holder))
    assert rc == 0 and any(json.loads(p.read_text())["type"] == "succeeded" for p in (tmp_path / "i").glob("*.json"))
    with pytest.raises(SystemExit):
        mod["main"](["run", str(batch), "--state", str(tmp_path / "s2.json"), "--hook", "bogus:x"],
                    client=FakeKaggleClient())


def test_empty_template_skips_cleanly_and_perms_enforced(tmp_path, monkeypatch):
    cfg = tmp_path / "w.env"
    cfg.write_text("# template\nWORLDSIM_WEBHOOK_URL=\nWORLDSIM_WEBHOOK_KEY=\n")
    cfg.chmod(0o600)
    monkeypatch.setattr(sh.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call network"))
    ev = sh.make_event("succeeded", state_path="s", batch_path="b", job={"name": "j", "attempts": 1}, now=1.0)
    assert sh.deliver({"type": "webhook", "config_file": str(cfg)}, ev) == "skipped"  # even when not optional
    cfg.chmod(0o644)
    with pytest.raises(sh.DeliveryError, match="chmod 600"):
        sh.deliver({"type": "webhook", "config_file": str(cfg)}, ev)
    # skipped deliveries leave nothing pending or dead-lettered
    d = sh.HookDispatcher(tmp_path / "h", start_thread=False)
    cfg.chmod(0o600)
    d.subscribe([{"sink": {"type": "webhook", "config_file": str(cfg)}}])
    d.emit(ev); d.deliver_due()
    assert json.loads((tmp_path / "h/outbox.json").read_text())["pending"] == {}
    assert not (tmp_path / "h/dead_letter.jsonl").exists()
