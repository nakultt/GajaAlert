"""LLM-driven Sarvam MCP tool orchestration for verified elephant incidents.

Unlike sarvam_workflow.py's fixed summarize -> translate -> speak sequence,
this lets the vision LLM decide which of the Sarvam MCP tools to call (and
with what arguments), in an OpenAI-style tool-calling loop against the MCP
server's live tool list. Only the alert-relevant completion, translation, and
TTS tools are exposed so their schemas fit the NPU model's 4096-token context.

Tool decisions run on the hardware-compiled Qwen QAIRT bundle through
scripts/serve_qwen_npu.py. The host MCP client performs the actual tool
operation and feeds each result into the next NPU inference turn.
"""

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gaja.tool_protocol import parse_tool_response
from sarvam_workflow import SARVAM_API_KEY

log = logging.getLogger("gaja.sarvam_agent")

AUDIO_SUBDIR = "audio"
LANGUAGE_CODES = {
    "en": "en-IN", "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN",
    "te": "te-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
    "mr": "mr-IN", "pa": "pa-IN", "od": "od-IN",
}
# The Qwen NPU bundle has a fixed 4096-token context. Tool results echoed into
# the conversation on every turn are what actually blow that budget, not the
# tool schemas -- cap what goes back into history hard, independent of what
# gets stored in SarvamResult for the dashboard (which keeps the full text).
MAX_TOOL_RESULT_HISTORY_CHARS = 400

AGENT_SYSTEM_PROMPT = (
    "You are the notification component of an elephant early-warning system. "
    "You have just been given a verified incident report. Decide which of "
    "the available tools to call, in what order and with what arguments, to "
    "summarize, translate, and deliver the alert to the field team. Call as "
    "many tools as you judge necessary, then reply with a short plain-text "
    "confirmation and no further tool calls."
)


@dataclass
class SarvamResult:
    """Structured record of what the agent actually did, for the dashboard."""
    summary: str = ""
    translations: dict = field(default_factory=dict)   # lang -> translated text
    audio_files: dict = field(default_factory=dict)     # lang -> path relative to incidents_dir
    final_message: str = ""


def _tool_named(tools, suffix: str) -> str | None:
    return next((tool.name for tool in tools if tool.name.endswith(suffix)), None)


def _result_text(call_result) -> str:
    text = call_result.content[0].text if call_result.content else ""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(obj, dict) and isinstance(obj.get("translated_text"), str):
        return obj["translated_text"]
    return text


def _capture_audio(audio_dir: str, lang: str, before: set[str]) -> str | None:
    """Rename the MCP server's generated WAV to the dashboard's stable name."""
    candidates = [
        path for path in os.listdir(audio_dir)
        if path.lower().endswith(".wav") and path not in before
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda name: os.path.getmtime(os.path.join(audio_dir, name)))
    source = os.path.join(audio_dir, newest)
    target = os.path.join(audio_dir, f"{lang}.wav")
    if os.path.abspath(source) != os.path.abspath(target):
        shutil.move(source, target)
    return target


def _mcp_tool_to_schema(tool) -> dict:
    description = tool.description or ""
    if len(description) > 300:
        description = description[:300] + "…"
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


async def _ensure_guaranteed_translations(session, cfg, result: SarvamResult,
                                           incident_text: str, audio_dir: str,
                                           translate_tool: str | None,
                                           tts_tool: str | None):
    """Deterministic top-up, called after the tool-calling agent loop
    finishes (however it finished): guarantees translate + TTS
    have actually run for every cfg.sarvam_languages entry, regardless of
    whether the LLM chose to call them on its own. Uses the same MCP
    session, so this is just extra tool calls, not a second connection."""
    for lang in cfg.sarvam_languages:
        lang = lang.lower()
        language_code = LANGUAGE_CODES.get(lang, lang if "-" in lang else f"{lang}-IN")
        if lang not in result.translations:
            if not translate_tool:
                log.error("Sarvam MCP did not advertise a translate tool")
                continue
            try:
                log.info("Sarvam agent: guaranteeing translation for %r", lang)
                call_result = await session.call_tool(translate_tool, {
                    "input": incident_text,
                    "target_language_code": language_code,
                    "source_language_code": "en-IN",
                })
                result.translations[lang] = _result_text(call_result)
            except Exception:
                log.exception("Guaranteed translate failed for %r", lang)
                continue

        if lang not in result.audio_files:
            if not tts_tool:
                log.error("Sarvam MCP did not advertise a TTS tool")
                continue
            text_to_speak = result.translations.get(lang) or incident_text
            before = set(os.listdir(audio_dir))
            try:
                log.info("Sarvam agent: guaranteeing TTS for %r", lang)
                await session.call_tool(tts_tool, {
                    "text": text_to_speak,
                    "target_language_code": language_code,
                })
                audio_path = _capture_audio(audio_dir, lang, before)
                if audio_path:
                    result.audio_files[lang] = os.path.relpath(
                        audio_path, cfg.incidents_dir).replace(os.sep, "/")
                else:
                    log.warning("Guaranteed TTS for %r returned no WAV", lang)
            except Exception:
                log.exception("Guaranteed TTS failed for %r", lang)


async def run_sarvam_agent(cfg, llm, incident_text: str, incident_id: str) -> SarvamResult:
    """Connect to the Sarvam MCP server, hand its tools to the vision LLM,
    and let the LLM decide which ones to call for this incident. Whatever
    the LLM decides, _ensure_guaranteed_translations then tops up any of
    cfg.sarvam_languages it didn't cover, so translate+TTS always happen.

    The MCP server writes into incidents/<incident_id>/audio and generated WAV
    files are renamed to <language>.wav for predictable dashboard paths."""
    result = SarvamResult()
    audio_dir = os.path.join(cfg.incidents_dir, incident_id, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    server_params = StdioServerParameters(
        command="uvx",
        args=["sarvam-mcp"],
        env={
            **os.environ,
            "SARVAM_API_KEY": SARVAM_API_KEY,
            "SARVAM_MCP_BASE_PATH": os.path.abspath(audio_dir),
            "SARVAM_AUDIO_OUTPUT_MODE": "files",
        },
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            translate_tool = _tool_named(tools_resp.tools, "tools_translate")
            tts_tool = _tool_named(tools_resp.tools, "tools_tts_stream")
            llm_tool = _tool_named(tools_resp.tools, "tools_llm_complete")
            selected_names = {name for name in (translate_tool, tts_tool, llm_tool) if name}
            relevant_tools = [
                tool for tool in tools_resp.tools
                if tool.name in selected_names
            ]
            tool_defs = [_mcp_tool_to_schema(t) for t in relevant_tools]
            if not tool_defs:
                log.warning("Sarvam agent: no MCP tools advertised")
                await _ensure_guaranteed_translations(
                    session, cfg, result, incident_text, audio_dir,
                    translate_tool, tts_tool,
                )
                return result

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": incident_text},
            ]

            for turn in range(cfg.sarvam_agent_max_turns):
                message = llm.chat_with_tools(cfg.tool_llm_base, messages, tool_defs)
                if message is None:
                    log.error("Sarvam agent: LLM call failed on turn %d", turn)
                    break
                if not message.get("tool_calls"):
                    parsed = parse_tool_response(message.get("content") or "", tool_defs)
                    if parsed.get("tool_calls"):
                        message = parsed
                    elif "tool_calls" in (message.get("content") or ""):
                        log.warning("Could not parse proposed tool call: %r", message.get("content"))
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    result.final_message = message.get("content") or ""
                    log.info("Sarvam agent done: %s", result.final_message)
                    break
                messages.append(message)
                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    audio_path = None
                    if name and name.endswith(("tts_speak", "tts_stream")):
                        code = str(args.get("target_language_code", "unknown"))
                        lang = code.split("-", 1)[0].lower()
                        audio_path = os.path.join(audio_dir, f"{lang}.wav")
                        before_audio = set(os.listdir(audio_dir))

                    log.info("Sarvam agent calling %s(%s)", name, args)
                    try:
                        call_result = await session.call_tool(name, args)
                        result_text = _result_text(call_result) or "ok"
                    except Exception as e:
                        log.exception("Sarvam tool %s failed", name)
                        result_text = f"error: {e}"

                    if name and name.endswith("llm_complete") and not result.summary:
                        result.summary = result_text
                    elif name and name.endswith("translate"):
                        code = str(args.get("target_language_code", "unknown"))
                        lang = code.split("-", 1)[0].lower()
                        result.translations[lang] = result_text
                    elif name and name.endswith(("tts_speak", "tts_stream")) and audio_path:
                        lang = str(args.get("target_language_code", "unknown")).split("-", 1)[0].lower()
                        captured = _capture_audio(audio_dir, lang, before_audio)
                        if captured:
                            result.audio_files[lang] = os.path.relpath(
                                captured, cfg.incidents_dir).replace(os.sep, "/")
                        else:
                            log.warning("TTS reported %r but no WAV was created", result_text)

                    # Full result_text goes into SarvamResult (dashboard)
                    # above; only a capped copy goes back into the LLM's own
                    # context so a handful of tool calls can't overflow it.
                    history_text = result_text
                    if len(history_text) > MAX_TOOL_RESULT_HISTORY_CHARS:
                        history_text = history_text[:MAX_TOOL_RESULT_HISTORY_CHARS] + " …[truncated]"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "content": history_text,
                    })
            else:
                log.warning("Sarvam agent hit max turns (%d) without finishing",
                            cfg.sarvam_agent_max_turns)
            await _ensure_guaranteed_translations(
                session, cfg, result, incident_text, audio_dir,
                translate_tool, tts_tool,
            )
    return result


if __name__ == "__main__":
    # Smoke test: requires :8080 running and SARVAM_API_KEY valid.
    from gaja.config import Config
    from gaja.llm import GemmaClient

    logging.basicConfig(level=logging.INFO)
    _cfg = Config.load()
    _llm = GemmaClient(_cfg)
    _result = asyncio.run(run_sarvam_agent(
        _cfg, _llm,
        "Elephant confirmed near Gaja camp perimeter at 06:15 PM "
        "(confidence 0.91). Detailed observation: a single adult elephant "
        "grazing near the tree line, moving slowly parallel to the fence. "
        "Public alert: Elephant sighted near the camp perimeter, stay indoors.",
        incident_id="smoketest",
    ))
    print(_result)
