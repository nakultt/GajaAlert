"""Read-only local dashboard: latest incidents with VLM report, Sarvam
summary, per-language translations, and playable TTS audio.

Reads incidents/incidents.jsonl (written by gaja.incidents.IncidentLog) fresh
on every request instead of sharing in-memory state with arm_server.py's
verifier thread -- simplest thing that can't get out of sync, no locking
needed. Serves frame JPEGs and Sarvam-generated .wav files straight out of
incidents/<id>/ alongside the JSON API.
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

log = logging.getLogger("gaja.dashboard")

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaja Alert — Incident Dashboard</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: #0b0c10; color: #f1f3f7;
    font-family: 'Segoe UI', system-ui, sans-serif;
  }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #9ea5b4; font-size: 13px; margin-bottom: 20px; }
  #empty { color: #9ea5b4; padding: 40px; text-align: center; border: 1px dashed #2a2f3a; border-radius: 10px; }
  .incident {
    background: rgba(17,20,28,0.75); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 18px 20px; margin-bottom: 16px;
  }
  .incident.latest { border-color: #ff007f55; box-shadow: 0 0 0 1px #ff007f33; }
  .row { display: flex; gap: 20px; flex-wrap: wrap; }
  .frames img {
    width: 160px; height: 120px; object-fit: cover; border-radius: 8px;
    margin-right: 8px; border: 1px solid rgba(255,255,255,0.1);
  }
  .meta { display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px; flex-wrap: wrap; }
  .meta .id { font-family: monospace; color: #9ea5b4; font-size: 12px; }
  .badge {
    font-size: 11px; padding: 2px 8px; border-radius: 999px; text-transform: uppercase;
    letter-spacing: .03em; font-weight: 600;
  }
  .badge.alerted { background: #10b98122; color: #10b981; }
  .badge.rejected { background: #ef444422; color: #ef4444; }
  .badge.no_video, .badge.llm_down { background: #f59e0b22; color: #f59e0b; }
  .conf { color: #4facfe; font-size: 13px; }
  .section-label { color: #9ea5b4; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; margin: 12px 0 4px; }
  .report, .summary { line-height: 1.5; font-size: 14px; }
  .lang-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-top: 6px; }
  .lang-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px 12px; }
  .lang-card .lang { font-size: 11px; color: #4facfe; text-transform: uppercase; margin-bottom: 4px; }
  .lang-card .text { font-size: 13px; margin-bottom: 8px; }
  audio { width: 100%; height: 32px; }
  details summary { cursor: pointer; color: #9ea5b4; font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
  <h1>Gaja Alert — Incident Dashboard</h1>
  <div class="sub">Auto-refreshes every 3s · latest sighting first</div>
  <div id="incidents"><div id="empty">Waiting for the first verified incident…</div></div>

<script>
function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }

function renderIncident(inc, isLatest) {
  const det = inc.detection || {};
  const alert = (inc.alert || {});
  const sarvam = inc.sarvam || {};
  const translations = sarvam.translations || {};
  const audioFiles = sarvam.audio_files || {};
  const frames = (inc.frames || []).map(f => `/incidents/${inc.id}/${f}`);

  const langCards = Object.keys(translations).map(lang => {
    const audio = audioFiles[lang]
      ? `<audio controls src="/incidents/${audioFiles[lang]}"></audio>`
      : `<div style="color:#666;font-size:12px;">no audio</div>`;
    return `<div class="lang-card">
      <div class="lang">${esc(lang)}</div>
      <div class="text">${esc(translations[lang])}</div>
      ${audio}
    </div>`;
  }).join('');

  const framesHtml = frames.length
    ? `<div class="frames">${frames.map(f => `<img src="${f}" loading="lazy">`).join('')}</div>`
    : '';

  return `<div class="incident ${isLatest ? 'latest' : ''}">
    <div class="meta">
      <span class="badge ${esc(inc.status || '')}">${esc(inc.status || 'unknown')}</span>
      <span>${esc(inc.timestamp || '')}</span>
      ${det.confidence != null ? `<span class="conf">confidence ${Number(det.confidence).toFixed(2)}</span>` : ''}
      <span class="id">${esc(inc.id || '')}</span>
    </div>
    <div class="row">
      ${framesHtml}
      <div style="flex:1; min-width:240px;">
        ${inc.report ? `<div class="section-label">VLM report</div><div class="report">${esc(inc.report)}</div>` : ''}
        ${alert.report ? `<div class="section-label">Public alert (en)</div><div class="report">${esc(alert.report)}</div>` : ''}
        ${sarvam.summary ? `<div class="section-label">Sarvam summary</div><div class="summary">${esc(sarvam.summary)}</div>` : ''}
      </div>
    </div>
    ${langCards ? `<div class="section-label">Translations &amp; audio</div><div class="lang-grid">${langCards}</div>` : ''}
    ${sarvam.final_message ? `<details><summary>Agent's final note</summary><div>${esc(sarvam.final_message)}</div></details>` : ''}
  </div>`;
}

async function refresh() {
  try {
    const res = await fetch('/api/incidents');
    const items = await res.json();
    const container = document.getElementById('incidents');
    if (!items.length) {
      container.innerHTML = '<div id="empty">Waiting for the first verified incident…</div>';
      return;
    }
    container.innerHTML = items.map((inc, i) => renderIncident(inc, i === 0)).join('');
  } catch (e) {
    console.error('refresh failed', e);
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""


def _load_incidents(incidents_dir: str, limit: int = 20) -> list:
    path = os.path.join(incidents_dir, "incidents.jsonl")
    items = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return []
    items.reverse()
    return items[:limit]


_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def make_handler(incidents_dir: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/incidents":
                data = json.dumps(_load_incidents(incidents_dir)).encode("utf-8")
                self._send(200, data, "application/json")
            elif path.startswith("/incidents/"):
                self._serve_file(path[len("/incidents/"):])
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def _serve_file(self, rel_path: str):
            parts = [p for p in rel_path.split("/") if p not in ("", ".", "..")]
            if not parts:
                self._send(404, b"not found", "text/plain")
                return
            full = os.path.abspath(os.path.join(incidents_dir, *parts))
            base = os.path.abspath(incidents_dir)
            if os.path.commonpath([full, base]) != base or not os.path.isfile(full):
                self._send(404, b"not found", "text/plain")
                return
            ext = os.path.splitext(full)[1].lower()
            ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
            with open(full, "rb") as f:
                self._send(200, f.read(), ctype)

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def start_dashboard(cfg) -> ThreadingHTTPServer:
    """Starts the dashboard on its own daemon thread; safe to call once at
    startup alongside the websocket servers and the YOLO display loop."""
    server = ThreadingHTTPServer(("127.0.0.1", cfg.dashboard_port),
                                  make_handler(cfg.incidents_dir))
    threading.Thread(target=server.serve_forever, daemon=True, name="gaja-dashboard").start()
    log.info("Dashboard on http://127.0.0.1:%d", cfg.dashboard_port)
    return server
