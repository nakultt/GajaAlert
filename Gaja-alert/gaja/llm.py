"""Clients for the local multimodal and NPU language-model servers.

Vision confirmation, report/alert text, and MCP decisions go to Qwen/QAIRT
on the NPU (:8081), with a hardcoded alert template as the final fallback.

Stdlib urllib only, mirroring scripts/test_inference.py.
enable_thinking=False is required: Gemma 4's template otherwise burns the
token budget on a reasoning pass and adds ~5-10x latency.
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("gaja.llm")

DETECT_PROMPT = (
    "You are an elephant early-warning system analyzing consecutive camera "
    "frames from a forest-edge camera. Does any frame contain an elephant "
    "(full or partial, day or night/IR)? Reply with JSON only: "
    '{"elephant": true/false, "confidence": 0.0-1.0, "notes": "..."}'
)

ALERT_SYSTEM = (
    "You are the alert-writing component of an automated elephant early-warning "
    "system. The detection has already been made by the system's acoustic and "
    "camera sensors; your only job is to turn the detection data into a short "
    "incident report and public alert text. Always answer with the requested JSON."
)

ALERT_USER_TEMPLATE = (
    "Detection data: elephant detected at {location} on {when}. "
    "Vision confidence: {confidence:.2f}. Vision notes: {notes}. "
    "Audio: low-frequency rumble detected (band energy ratio {ratio:.2f}). "
    "Write the report and alerts now. Reply with JSON only: "
    '{{"report": "<2-3 sentence incident report in English>", '
    '"alerts": {{"en": "<1-sentence public alert in English>", '
    '"hi": "<the same alert in Hindi>", "ta": "<the same alert in Tamil>"}}}}'
)

DETAILED_REPORT_PROMPT = (
    "An elephant has just been verified in these camera frames. Write a "
    "detailed incident report for a wildlife safety team: describe the "
    "elephant (appearance, approximate size/position), its apparent "
    "behavior (e.g. grazing, moving, agitated, approaching structures), the "
    "surroundings, and any other relevant observations. "
    'Reply with JSON only: {"description": "..."}'
)


@dataclass
class Detection:
    elephant: bool
    confidence: float
    notes: str
    raw: str = ""


@dataclass
class Alert:
    report: str
    alerts: dict = field(default_factory=dict)
    fallback: bool = False


def parse_json_block(text: str) -> dict | None:
    """Extract the first {...} object from LLM output (tolerates ``` fences)."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class GemmaClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def healthy(self, base: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                return r.status == 200
        except OSError:
            return False

    def _chat_message(self, base: str, messages: list, max_tokens: int,
                       tools: list | None = None) -> dict | None:
        """Full assistant message dict (content + tool_calls, if any)."""
        body = {
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(self.cfg.llm_retries + 1):
            try:
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.cfg.llm_timeout_s) as r:
                    out = json.load(r)
                message = out["choices"][0]["message"]
                log.info("%s answered in %.1fs", base, time.time() - t0)
                return message
            except urllib.error.HTTPError as e:
                # 4xx (e.g. "exceeds the available context size") is a
                # request problem, not a transient failure -- retrying the
                # same oversized payload would just fail 3x for nothing.
                try:
                    detail = e.read().decode(errors="replace")[:300]
                except Exception:
                    detail = str(e)
                log.error("LLM call to %s rejected (HTTP %s): %s", base, e.code, detail)
                return None
            except (OSError, KeyError, json.JSONDecodeError) as e:
                log.error("LLM call to %s failed (attempt %d/%d): %s",
                          base, attempt + 1, self.cfg.llm_retries + 1, e)
                if attempt < self.cfg.llm_retries:
                    time.sleep(5)
        return None

    def _chat(self, base: str, messages: list, max_tokens: int) -> str | None:
        message = self._chat_message(base, messages, max_tokens)
        return message.get("content") if message else None

    def chat_with_tools(self, base: str, messages: list, tools: list,
                         max_tokens: int = 500) -> dict | None:
        """Return an OpenAI-style assistant message for an agentic tool loop.

        The Qwen QAIRT server constrains its NPU output to JSON and validates
        selected tool names before returning this compatible representation.
        """
        return self._chat_message(base, messages, max_tokens, tools=tools)

    def detect_elephant(self, jpegs: list[bytes]) -> Detection | None:
        """Vision confirmation on Qwen QAIRT/NPU. None means it was unreachable."""
        content = [{"type": "text", "text": DETECT_PROMPT}]
        for jpeg in jpegs:
            b64 = base64.b64encode(jpeg).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        text = self._chat(self.cfg.vision_llm_base,
                          [{"role": "user", "content": content}], max_tokens=150)
        if text is None:
            return None
        obj = parse_json_block(text)
        if obj is None:
            log.warning("Unparseable detection reply: %r", text[:200])
            return Detection(False, 0.0, "unparseable reply", raw=text)
        return Detection(
            bool(obj.get("elephant", False)),
            float(obj.get("confidence", 0.0)),
            str(obj.get("notes", "")),
            raw=text,
        )

    def generate_alert(self, det: Detection, ratio: float, when: str) -> Alert:
        """Report + alerts on NPU :8081, then :8080, then a template."""
        messages = [
            {"role": "system", "content": ALERT_SYSTEM},
            {"role": "user", "content": ALERT_USER_TEMPLATE.format(
                location=self.cfg.location_name, when=when,
                confidence=det.confidence, notes=det.notes, ratio=ratio)},
        ]
        for base in (self.cfg.text_llm_base, self.cfg.vision_llm_base):
            text = self._chat(base, messages, max_tokens=400)
            obj = parse_json_block(text) if text else None
            if obj and obj.get("report") and isinstance(obj.get("alerts"), dict):
                return Alert(str(obj["report"]), dict(obj["alerts"]))
            if text is not None:
                log.warning("Unparseable alert reply from %s: %r", base, (text or "")[:200])
        msg = (f"ALERT: Elephant detected near {self.cfg.location_name} at {when}. "
               "Stay indoors and away from the area.")
        return Alert(report=msg, alerts={"en": msg}, fallback=True)

    def generate_detailed_report(self, jpegs: list[bytes]) -> str | None:
        """Rich behavior/surroundings narrative on :8080 (vision), called once
        per sighting right after VLM verification succeeds."""
        content = [{"type": "text", "text": DETAILED_REPORT_PROMPT}]
        for jpeg in jpegs:
            b64 = base64.b64encode(jpeg).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        text = self._chat(self.cfg.vision_llm_base,
                          [{"role": "user", "content": content}], max_tokens=400)
        if text is None:
            return None
        obj = parse_json_block(text)
        if obj and obj.get("description"):
            return str(obj["description"])
        return text
