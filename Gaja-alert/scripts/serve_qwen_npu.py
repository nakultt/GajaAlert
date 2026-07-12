"""OpenAI-compatible Qwen3-VL server backed by QAIRT on Hexagon NPU.

The model stays resident for low-latency multi-turn MCP orchestration. Vision
confirmation remains on the llama.cpp multimodal server; this endpoint handles
incident text generation and structured MCP decisions fully on the NPU.
"""

from __future__ import annotations

import json
import os
import base64
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BUNDLE = Path.home() / "Downloads" / (
    "qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite"
) / "qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_x_elite"
DEFAULT_QAIRT = Path.home() / "Downloads" / "v2.48.0.260626" / "qairt" / "2.48.0.260626"

MODEL_DIR = Path(os.environ.get("GAJA_QWEN_NPU_BUNDLE", DEFAULT_BUNDLE)).resolve()
QAIRT_HOME = Path(os.environ.get("QAIRT_HOME", DEFAULT_QAIRT)).resolve()
HOST = os.environ.get("GAJA_QWEN_NPU_HOST", "127.0.0.1")
PORT = int(os.environ.get("GAJA_QWEN_NPU_PORT", "8081"))
ALIAS = "qwen3-vl-4b-qairt-npu"

if not MODEL_DIR.joinpath("metadata.json").is_file():
    raise SystemExit(f"Qwen QAIRT bundle not found: {MODEL_DIR}")
if not QAIRT_HOME.is_dir():
    raise SystemExit(f"QAIRT SDK not found: {QAIRT_HOME}")

arch_bin = QAIRT_HOME / "bin" / "aarch64-windows-msvc"
arch_lib = QAIRT_HOME / "lib" / "aarch64-windows-msvc"
os.environ["QAIRT_HOME"] = str(QAIRT_HOME)
os.environ["PATH"] = os.pathsep.join((str(arch_bin), str(arch_lib), os.environ.get("PATH", "")))

from geniex import AutoModelForCausalLM  # noqa: E402
from gaja.tool_protocol import parse_tool_response, prepare_tool_messages  # noqa: E402

print(f"[qwen-npu] loading {MODEL_DIR} through QAIRT/HTP...")
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), device_map="qairt:npu")
model_lock = threading.Lock()
print(f"[qwen-npu] ready at http://{HOST}:{PORT}/v1/chat/completions")


def _prompt(messages: list[dict], tools: list[dict]) -> str:
    if tools:
        messages = prepare_tool_messages(messages, tools)
    return model.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )


def _prepare_images(messages: list[dict]) -> tuple[list[dict], list[str]]:
    """Convert OpenAI data-URL image blocks to GenieX VLM image blocks."""
    prepared = []
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            prepared.append(message)
            continue
        blocks = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                blocks.append(block)
                continue
            value = block.get("image_url", {})
            url = value.get("url", "") if isinstance(value, dict) else str(value)
            if not url.startswith("data:image/") or ";base64," not in url:
                raise ValueError("Qwen NPU server accepts image_url only as data:image/...;base64 URLs")
            header, encoded = url.split(",", 1)
            subtype = header.split("/", 1)[1].split(";", 1)[0].lower()
            suffix = ".png" if subtype == "png" else ".jpg"
            with tempfile.NamedTemporaryFile(prefix="gaja-vlm-", suffix=suffix, delete=False) as tmp:
                tmp.write(base64.b64decode(encoded, validate=True))
                path = tmp.name
            paths.append(path)
            blocks.append({"type": "image", "image": path})
        prepared.append({**message, "content": blocks})
    return prepared, paths


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok", "model": ALIAS, "backend": "qairt",
                "device": "npu", "bundle": str(MODEL_DIR),
            })
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": ALIAS, "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        image_paths: list[str] = []
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length))
            messages = req["messages"]
            messages, image_paths = _prepare_images(messages)
            tools = req.get("tools") or []
            stream = bool(req.get("stream", False)) and not tools
            max_tokens = max(1, min(int(req.get("max_tokens", 256)), 1024))
            temperature = float(req.get("temperature", 0.1))
            created = int(time.time())
            chat_id = f"chatcmpl-{created}"

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                with model_lock:
                    model.reset()
                    prompt = _prompt(messages, tools)
                    output = model.generate(
                        prompt, max_new_tokens=max_tokens,
                        temperature=temperature, images=image_paths, stream=True,
                    )
                    for token in output:
                        chunk = {
                            "id": chat_id, "object": "chat.completion.chunk",
                            "created": created, "model": ALIAS,
                            "choices": [{"index": 0, "delta": {"content": token},
                                         "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            with model_lock:
                model.reset()
                prompt = _prompt(messages, tools)
                output = model.generate(
                    prompt, max_new_tokens=max_tokens,
                    temperature=temperature, json_mode=bool(tools),
                    images=image_paths,
                )
            message = parse_tool_response(output.text, tools) if tools else {
                "role": "assistant", "content": output.text,
            }
            self._send(200, {
                "id": chat_id, "object": "chat.completion", "created": created,
                "model": ALIAS,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"
                             if message.get("tool_calls") else "stop"}],
                "usage": {
                    "prompt_tokens": output.profile.prompt_tokens,
                    "completion_tokens": output.profile.generated_tokens,
                    "total_tokens": output.profile.prompt_tokens + output.profile.generated_tokens,
                },
                "timings": {
                    "ttft_ms": round(output.profile.ttft / 1000.0, 3),
                    "prefill_tokens_per_second": output.profile.prefill_speed,
                    "decode_tokens_per_second": output.profile.decode_speed,
                    "backend": output.profile.backend,
                    "device": output.profile.device,
                },
            })
        except Exception as exc:
            try:
                self._send(500, {"error": str(exc)})
            except (BrokenPipeError, ConnectionError):
                pass
        finally:
            for path in image_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def log_message(self, fmt, *args):
        print(f"[qwen-npu] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        model.close()
