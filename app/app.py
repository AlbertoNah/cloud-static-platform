import os
import time
import json
from pathlib import Path

import yaml
from flask import Flask, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))

app = Flask(__name__)

_cache = {
    "events": {"ts": 0, "data": None},
    "news": {"ts": 0, "data": None},
    "faq": {"ts": 0, "data": None},
}

def _now() -> float:
    return time.time()

def _load_json(filename: str) -> dict:
    p = DATA_DIR / filename
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_yaml(filename: str) -> dict:
    p = DATA_DIR / filename
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_items(data) -> dict:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return {"items": data["items"]}
    return {"items": []}

def _get_cached(key: str, loader):
    entry = _cache[key]
    if entry["data"] is not None and (_now() - entry["ts"]) < CACHE_TTL:
        return entry["data"], True

    data = _normalize_items(loader())
    entry["data"] = data
    entry["ts"] = _now()
    return data, False

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

@app.get("/readyz")
def readyz():
    ready = DATA_DIR.exists()
    return jsonify({"status": "ready" if ready else "not_ready"}), (200 if ready else 503)

@app.get("/api/events")
def api_events():
    data, cached = _get_cached("events", lambda: _load_json("events.json"))
    return jsonify({"items": data["items"], "cached": cached}), 200

@app.get("/api/news")
def api_news():
    data, cached = _get_cached("news", lambda: _load_json("news.json"))
    return jsonify({"items": data["items"], "cached": cached}), 200

@app.get("/api/faq")
def api_faq():
    data, cached = _get_cached("faq", lambda: _load_yaml("faq.yaml"))
    return jsonify({"items": data["items"], "cached": cached}), 200

@app.get("/")
def index():
    events, _ = _get_cached("events", lambda: _load_json("events.json"))
    news, _ = _get_cached("news", lambda: _load_json("news.json"))
    faq, _ = _get_cached("faq", lambda: _load_yaml("faq.yaml"))

    html = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>Static Content Platform</title></head>
    <body style="font-family: Arial; margin: 24px;">
      <h1>Static Content Platform</h1>
      <p>TTL Cache: {{ttl}}s (env CACHE_TTL)</p>

      <h2>Events</h2><pre>{{events}}</pre>
      <h2>News</h2><pre>{{news}}</pre>
      <h2>FAQ</h2><pre>{{faq}}</pre>
    </body>
    </html>
    """
    return render_template_string(
        html,
        ttl=CACHE_TTL,
        events=json.dumps(events, indent=2, ensure_ascii=False),
        news=json.dumps(news, indent=2, ensure_ascii=False),
        faq=json.dumps(faq, indent=2, ensure_ascii=False),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
