"""Экран: локальный HTTP-сервер для страницы киоска. Отдаёт display/index.html, шрифты,
/state.json (текущее состояние службы) и /archive.json (призрачные цитаты последних циклов)."""

from __future__ import annotations

import json
import logging
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ROOT, path_dir

log = logging.getLogger("apophenia.display")


def archive_quotes(cfg: Dict[str, Any], limit: int = 30) -> Dict[str, Any]:
    d = path_dir(cfg, "archive")
    files = sorted(d.glob("cycle_*.json"), reverse=True)[:limit]
    cycles = []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cycles.append({
            "cycle": rec.get("cycle"), "finished_at": rec.get("finished_at"),
            "outcome": rec.get("outcome"), "trajectory": rec.get("trajectory", []),
            "confidences": rec.get("confidences", []),
            "quotes": [{"span": q["span"], "title": q.get("title", ""), "class": q.get("class", "")}
                       for q in rec.get("ghosts", {}).get("quotes", [])],
        })
    return {"cycles": cycles}


def last_cycle(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Полная запись последнего завершённого цикла (для протокола на экране в тестовом режиме)."""
    d = path_dir(cfg, "archive")
    files = sorted(d.glob("cycle_*.json"), reverse=True)
    for f in files:
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return {}


class DisplayServer:
    def __init__(self, cfg: Dict[str, Any]):
        d = cfg.get("display", {})
        self.enabled = bool(d.get("enabled", True))
        self.host = d.get("host", "127.0.0.1")
        self.port = int(d.get("port", 8765))
        self.cfg = cfg
        self.state_path = path_dir(cfg, "state") / "display.json"
        self.web_root = ROOT / "display"
        self.fonts_root = ROOT / "fonts"
        self._server: Optional[ThreadingHTTPServer] = None
        self.quote_seconds = int(d.get("quote_seconds", 12))

    def start(self) -> None:
        if not self.enabled:
            return
        outer = self

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, fmt, *args):  # тихий сервер
                pass

            def _json(self, obj: Any) -> None:
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _file(self, path: Path, ctype: str) -> None:
                if not path.exists():
                    self.send_error(404)
                    return
                data = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/", "/index.html"):
                    self._file(outer.web_root / "index.html", "text/html; charset=utf-8")
                elif p == "/state.json":
                    try:
                        st = json.loads(outer.state_path.read_text(encoding="utf-8")) if outer.state_path.exists() else {}
                    except json.JSONDecodeError:
                        st = {}
                    st["quote_seconds"] = outer.quote_seconds
                    self._json(st)
                elif p == "/archive.json":
                    self._json(archive_quotes(outer.cfg))
                elif p == "/last.json":
                    self._json(last_cycle(outer.cfg))
                elif p.startswith("/fonts/") and ".." not in p:
                    self._file(outer.fonts_root / p[len("/fonts/"):], "font/ttf")
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        t = threading.Thread(target=self._server.serve_forever, name="display", daemon=True)
        t.start()
        log.info("Экран: http://%s:%d/", self.host, self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
