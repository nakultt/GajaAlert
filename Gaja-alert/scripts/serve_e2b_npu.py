"""OpenAI-compatible server for Gemma 4 E2B (QAT Q4_0) on the Hexagon NPU.

Why this exists (see docs/LOCAL_INFERENCE.md, "NPU-prefill / GPU-decode"
section): the requested prefill-on-NPU + decode-on-GPU split is not
implementable in any shipped runtime, and per-backend measurements showed the
NPU wins BOTH phases anyway (prefill 9.8s vs GPU 17.3s on a ~3.5K-token
prompt; decode 19.4 vs 18.9 tok/s). Running everything on the NPU is the
optimal configuration that also leaves the CPU completely free for the
OpenCV/YOLO pipeline.

Run with the native-ARM64 geniex env (NOT the project .venv):
    C:\\Users\\qcwor\\llm\\geniex-env\\Scripts\\python.exe scripts\\serve_e2b_npu.py

Exposes http://127.0.0.1:8081/v1/chat/completions (same client contract as
the E4B llama-server on :8080 — swap by changing the port).
Text-only: the QAT GGUF has no mmproj alongside it; use the E4B server for
image/audio input.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = r"C:\Users\qcwor\llm\models\gemma-4-e2b-qat"
HOST, PORT = "127.0.0.1", 8081
ALIAS = "gemma-4-e2b-npu"

print("[serve_e2b_npu] loading model on NPU (takes ~5-10s)...")
from geniex import AutoModelForCausalLM  # noqa: E402  (import after banner)

# "llama_cpp:npu" = llama.cpp runtime pinned to the Hexagon HTP device.
# A bare "npu" would route to the qairt plugin, which rejects GGUF files.
_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="llama_cpp:npu")
_lock = threading.Lock()  # single NPU session; serialize requests
print(f"[serve_e2b_npu] ready on http://{HOST}:{PORT}/v1/chat/completions")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": ALIAS, "device": "npu"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = req["messages"]
            max_tokens = int(req.get("max_tokens", 256))
            temperature = float(req.get("temperature", 0.7))
            stream_requested = bool(req.get("stream", False))
            # Default thinking OFF (detection prompts need direct answers —
            # same rationale as the E4B server). Clients can override via
            # chat_template_kwargs for llama-server API parity.
            think = bool(
                req.get("chat_template_kwargs", {}).get("enable_thinking", False)
            )
            prompt = _model.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=think
            )
            t0 = time.time()

            if stream_requested:
                # SSE for the web chat UI: yield tokens as llama_cpp produces
                # them instead of buffering the whole reply first.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                chat_id = f"chatcmpl-{int(t0)}"
                with _lock:
                    for token in _model.generate(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    ):
                        chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": int(t0),
                            "model": ALIAS,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()

                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": ALIAS,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "timings": {"wall_s": round(time.time() - t0, 2)},
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                with _lock:
                    text = "".join(
                        _model.generate(
                            prompt,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )
                    )
                self._send(200, {
                    "id": f"chatcmpl-{int(t0)}",
                    "object": "chat.completion",
                    "created": int(t0),
                    "model": ALIAS,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "timings": {"wall_s": round(time.time() - t0, 2)},
                })
        except Exception as e:  # surface errors to the client, keep serving
            if not self.wfile.closed:
                try:
                    self._send(500, {"error": str(e)})
                except Exception:
                    pass

    def log_message(self, fmt, *args):  # quieter default logging
        print(f"[serve_e2b_npu] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
