"""Loopback-only evidence server with bounded exploratory inference."""

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from workbench.catalog import DATA, verify_data
from workbench.live import LiveEngine

STATIC = Path(__file__).resolve().parent / "static"


def make_server(port=8765, engine=None):
    verify_data()
    token = secrets.token_urlsafe(32)
    engine = engine or LiveEngine()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, body, mime="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def trusted_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.trusted_host():
                return self.reply(403, {"error": "Use the loopback address printed by the server"})
            path = urlsplit(self.path).path
            if path == "/api/evidence":
                payload = json.loads((DATA / "evidence.json").read_text())
                return self.reply(200, {**payload, "csrf": token, "live_available": engine.available()})
            assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path not in assets:
                return self.reply(404, {"error": "Not found"})
            name, mime = assets[path]
            self.reply(200, (STATIC / name).read_bytes(), mime)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.trusted_host() or self.headers.get("X-Oczy-Token") != token or (origin is not None and origin != "http://" + self.headers.get("Host", "")):
                return self.reply(403, {"error": "Reload this local page before starting a trial"})
            if self.path != "/api/trial":
                return self.reply(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected a small JSON trial request")
                value = json.loads(self.rfile.read(length))
                self.reply(200, engine.query(value))
            except (ValueError, UnicodeError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.engine = engine
    return server


def serve(port=8765, runtime=None, model=None):
    server = make_server(port, LiveEngine(runtime, model))
    print(f"Oczy workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.engine.close()
        server.server_close()
