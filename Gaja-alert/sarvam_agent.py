"""LLM-driven Sarvam MCP tool orchestration for verified elephant incidents.

Unlike sarvam_workflow.py's fixed summarize -> translate -> speak sequence,
this lets the vision LLM decide which of the Sarvam MCP tools to call (and
with what arguments), in an OpenAI-style tool-calling loop against the MCP
server's live tool list. All four existing tools (including the audio
streaming one) are exposed exactly as advertised by the server; none of
sarvam_workflow.py's tool-call logic is touched here.

Only the CPU vision server (:8080, llama.cpp started with --jinja) has
verified tool-calling support in this repo — the NPU-only servers
(scripts/serve_e2b_npu.py, scripts/serve_e2b_split.py) are minimal HTTP
handlers that don't implement the OpenAI `tools` request field, so the
agent loop below talks to :8080 (see gaja/llm.py's chat_with_tools).
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sarvam_workflow import SARVAM_API_KEY

log = logging.getLogger("gaja.sarvam_agent")

AUDIO_SUBDIR = "audio"
# The vision server this agent talks to (:8080) is configured with a 32K
# context window (see docs/LOCAL_INFERENCE.md). Tool results echoed back into
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


async def run_sarvam_agent(cfg, llm, incident_text: str, incident_id: str) -> SarvamResult:
    """Connect to the Sarvam MCP server, hand its tools to the vision LLM,
    and let the LLM decide which ones to call for this incident.

    `sarvam_tts_speak`'s own `output_file` argument is redirected (not its
    implementation) to incidents/<incident_id>/audio/<language>.wav so the
    dashboard has a predictable path to serve, regardless of what filename
    the model itself proposed."""
    result = SarvamResult()
    audio_dir = os.path.join(cfg.incidents_dir, incident_id, AUDIO_SUBDIR)
    os.makedirs(audio_dir, exist_ok=True)

    server_params = StdioServerParameters(
        command="uvx",
        args=["sarvam-mcp"],
        env={**os.environ, "SARVAM_API_KEY": SARVAM_API_KEY},
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_defs = [_mcp_tool_to_schema(t) for t in tools_resp.tools]
            if not tool_defs:
                log.warning("Sarvam agent: no MCP tools advertised")
                return result

            messages = [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": incident_text},
            ]

            for turn in range(cfg.sarvam_agent_max_turns):
                message = llm.chat_with_tools(cfg.vision_llm_base, messages, tool_defs)
                if message is None:
                    log.error("Sarvam agent: LLM call failed on turn %d", turn)
                    return result
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    result.final_message = message.get("content") or ""
                    log.info("Sarvam agent done: %s", result.final_message)
                    return result
                messages.append(message)
                for call in tool_calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    audio_path = None
                    if name == "sarvam_tts_speak":
                        lang = str(args.get("language", "unknown")).lower()
                        audio_path = os.path.join(audio_dir, f"{lang}.wav")
                        args["output_file"] = audio_path

                    log.info("Sarvam agent calling %s(%s)", name, args)
                    try:
                        call_result = await session.call_tool(name, args)
                        result_text = call_result.content[0].text if call_result.content else "ok"
                    except Exception as e:
                        log.exception("Sarvam tool %s failed", name)
                        result_text = f"error: {e}"

                    if name == "sarvam_llm_complete" and not result.summary:
                        result.summary = result_text
                    elif name == "sarvam_translate":
                        lang = str(args.get("target_language", "unknown")).lower()
                        result.translations[lang] = result_text
                    elif name == "sarvam_tts_speak" and audio_path:
                        lang = str(args.get("language", "unknown")).lower()
                        if os.path.isfile(audio_path):
                            result.audio_files[lang] = os.path.relpath(
                                audio_path, cfg.incidents_dir).replace(os.sep, "/")
                        else:
                            log.warning("sarvam_tts_speak reported %r but no file at %s",
                                        result_text, audio_path)

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
            log.warning("Sarvam agent hit max turns (%d) without finishing",
                        cfg.sarvam_agent_max_turns)
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
