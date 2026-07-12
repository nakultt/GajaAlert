"""Live smoke test for Qwen QAIRT text generation and tool calling on :8081."""

import json
import urllib.request

BASE = "http://127.0.0.1:8081"


def post(payload):
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


with urllib.request.urlopen(f"{BASE}/health", timeout=5) as response:
    health = json.load(response)
assert health["backend"] == "qairt" and health["device"] == "npu", health

text = post({
    "messages": [{"role": "user", "content": "Reply with exactly: NPU ready"}],
    "max_tokens": 20,
    "temperature": 0.1,
})
assert text["choices"][0]["message"]["content"], text
assert text["timings"]["backend"] == "qairt", text
assert text["timings"]["device"] == "NPU", text

tool = post({
    "messages": [{"role": "user", "content": (
        "Call sarvam_tools_translate to translate the word warning to Hindi. "
        "Use input='warning', source_language_code='en-IN', and "
        "target_language_code='hi-IN'."
    )}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "sarvam_tools_translate",
            "description": "Translate text",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                    "source_language_code": {"type": "string"},
                    "target_language_code": {"type": "string"},
                },
                "required": ["input", "source_language_code", "target_language_code"],
            },
        },
    }],
    "max_tokens": 100,
    "temperature": 0.1,
})
calls = tool["choices"][0]["message"].get("tool_calls") or []
assert calls and calls[0]["function"]["name"] == "sarvam_tools_translate", tool
arguments = json.loads(calls[0]["function"]["arguments"])
assert arguments["target_language_code"] == "hi-IN", arguments

print(json.dumps({
    "health": health,
    "text": text["choices"][0]["message"],
    "tool_call": calls[0],
    "timings": tool["timings"],
}, indent=2, ensure_ascii=True))
