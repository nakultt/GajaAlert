"""OpenAI-compatible server for Gemma 4 E2B (QAT Q4_0) on hybrid Snapdragon NPU + GPU.

Loaded with pipeline parallelism across:
  - HTP0 (NPU) for layers 0-16
  - GPUOpenCL (GPU) for layers 17-35
  - CPU for fallback operations only (minimal overhead)

Run with the native-ARM64 geniex env:
    C:\\Users\\qcwor\\llm\\geniex-env\\Scripts\\python.exe scripts\\serve_e2b_split.py
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = r"C:\Users\qcwor\llm\models\gemma-4-e2b-qat"
HOST, PORT = "127.0.0.1", 8082
ALIAS = "gemma-4-e2b-split"

print("[serve_e2b_split] loading model on NPU + GPU (takes ~5-15s)...")
from geniex import AutoModelForCausalLM  # noqa: E402

# Initialize the model on both NPU (HTP0) and GPU (GPUOpenCL) using pipeline parallelism.
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    device_map="llama_cpp:HTP0,GPUOpenCL",
    verbose=False
)
_lock = threading.Lock()  # Serialize inference sessions
print(f"[serve_e2b_split] ready on http://{HOST}:{PORT}/v1/chat/completions")


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
            self._send(200, {"status": "ok", "model": ALIAS, "device": "HTP0,GPUOpenCL"})
        elif self.path in ("/", "/index.html"):
            try:
                import os
                html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "index.html")
                with open(html_path, "r", encoding="utf-8") as f:
                    content = f.read().encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._send(500, {"error": f"Failed to load web interface: {str(e)}"})
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
            think = bool(
                req.get("chat_template_kwargs", {}).get("enable_thinking", False)
            )
            prompt = _model.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=think
            )
            t0 = time.time()

            if stream_requested:
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
                                "finish_reason": None
                            }]
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                
                # Send final chunk with finish_reason and timings
                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(t0),
                    "model": ALIAS,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }],
                    "timings": {"wall_s": round(time.time() - t0, 2)}
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
        except Exception as e:
            if not self.wfile.closed:
                try:
                    self._send(500, {"error": str(e)})
                except Exception:
                    pass

    def log_message(self, fmt, *args):
        print(f"[serve_e2b_split] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    try:
        server = ThreadingHTTPServer((HOST, PORT), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("[serve_e2b_split] shutting down...")
    finally:
        _model.close()
