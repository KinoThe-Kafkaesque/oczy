"""Exercise live-input boundaries, local HTTP checks and package tamper detection."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from workbench.catalog import digest
from workbench.live import validate_trial
from workbench.pack import verify_package
from workbench.server import make_server


@pytest.mark.parametrize("value", [{"client": "amber", "input": "a b", "seed": 0},
                                  {"client": "amber", "input": "x", "seed": True},
                                  {"client": "../", "input": "x", "seed": 0},
                                  {"client": "amber", "input": "x", "seed": 3},
                                  {"client": "amber", "input": "x", "seed": 0, "path": "/etc/passwd"}, []])
def test_rejects_unbounded_or_ambiguous_live_inputs(value):
    with pytest.raises(ValueError):
        validate_trial(value)


def test_package_tamper_is_detected(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_text("original")
    (tmp_path / "PACKAGE_MANIFEST.json").write_text(json.dumps({"files": {file.name: digest(file)}}))
    assert verify_package(tmp_path)["packaged_files"] == 1
    file.write_text("changed")
    with pytest.raises(ValueError, match="Packaged file changed"):
        verify_package(tmp_path)


def test_http_security_and_valid_trial():
    class Engine:
        calls = []

        def available(self):
            return True

        def query(self, value):
            self.calls.append(validate_trial(value))
            return {"mode": "test fixture"}

    engine = Engine()
    server = make_server(0, engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/evidence") as response:
            token = json.load(response)["csrf"]
        payload = json.dumps({"client": "amber", "input": "berry", "seed": 0}).encode()
        for path, headers, status in [('/../CURRENT_STATE.md', {}, 404), ('/api/evidence', {'Host': 'attacker.invalid'}, 403)]:
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url + path, headers=headers))
            assert error.value.code == status
        for extra in ({}, {"X-Oczy-Token": token, "Origin": "https://attacker.invalid"}):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url + "/api/trial", data=payload, headers={"Content-Type": "application/json", **extra}))
            assert error.value.code == 403
        assert not engine.calls
        with urlopen(Request(url + "/api/trial", data=payload, headers={"Content-Type": "application/json", "X-Oczy-Token": token, "Origin": url})) as response:
            assert json.load(response)["mode"] == "test fixture"
        assert len(engine.calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
