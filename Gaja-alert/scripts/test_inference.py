"""Smoke test for the local llama-server (Gemma 4 E4B / Qwen3-VL).

Usage:
    uv run python scripts/test_inference.py             # text-only test
    uv run python scripts/test_inference.py frame.jpg   # vision test

Requires the server to be running:  powershell -File scripts\\serve-llm.ps1
Uses only stdlib (urllib) so it works even before deps are installed.
"""

import base64
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"


def chat(messages, **kw):
    # enable_thinking=False is required: Gemma 4's chat template defaults to
    # emitting a chain-of-thought "reasoning_content" pass before the real
    # answer, which burns the token budget on short max_tokens requests and
    # adds ~5-10x latency. Detection/report tasks need fast, direct answers.
    kw.setdefault("chat_template_kwargs", {"enable_thinking": False})
    body = json.dumps({"messages": messages, "temperature": 0.1, **kw}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    dt = time.time() - t0
    usage = out.get("usage", {})
    print(f"[{dt:.1f}s] prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} tokens")
    return out["choices"][0]["message"]["content"]


def main():
    # health check
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        print("health:", r.read().decode().strip())

    if len(sys.argv) > 1:
        img = base64.b64encode(open(sys.argv[1], "rb").read()).decode()
        content = [
            {"type": "text", "text": (
                "Does this image contain an elephant? "
                'Reply with JSON only: {"elephant": true/false, "confidence": 0.0-1.0, "notes": "..."}'
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ]
    else:
        content = "Reply with exactly one short sentence: what model are you?"

    print(chat([{"role": "user", "content": content}]))


if __name__ == "__main__":
    main()
