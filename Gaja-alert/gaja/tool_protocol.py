"""OpenAI tool-call compatibility for local models without native tool templates.

The Qwen QAIRT bundle understands JSON reliably but its exported chat template
does not currently render the ``tools`` argument.  This module injects the
advertised schemas into the system message and converts the model's constrained
JSON response back into the OpenAI message shape used by :mod:`sarvam_agent`.
"""

from __future__ import annotations

import json
import uuid


TOOL_PROTOCOL = """
You can call the tools listed below.
AVAILABLE_TOOLS:
{tools}

If a tool is needed, respond ONLY with valid JSON in this exact shape:
{{"tool_calls":[{{"name":"tool_name","arguments":{{"key":"value"}}}}]}}
You may include multiple calls in the tool_calls array when appropriate.
When no tool is needed, respond ONLY with: {{"content":"your answer"}}
Never invent a tool name and never claim that you cannot call the listed tools.
""".strip()


def _repair_delimiters(text: str) -> str:
    """Repair only missing JSON object/array closers, never content or names."""
    pairs = {"}": "{", "]": "["}
    closer = {"{": "}", "[": "]"}
    stack: list[str] = []
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
        elif char in closer:
            stack.append(char)
            output.append(char)
        elif char in pairs:
            while stack and stack[-1] != pairs[char]:
                output.append(closer[stack.pop()])
            if stack:
                stack.pop()
            output.append(char)
        else:
            output.append(char)
    while stack:
        output.append(closer[stack.pop()])
    return "".join(output)


def _tool_catalog(tools: list[dict]) -> list[dict]:
    catalog = []
    for item in tools:
        fn = item.get("function", item)
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        catalog.append({
            "name": name,
            "description": str(fn.get("description") or "")[:500],
            "parameters": fn.get("parameters") or {
                "type": "object", "properties": {}
            },
        })
    return catalog


def prepare_tool_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Return text-only chat messages containing a strict tool protocol.

    Prior assistant calls and MCP results are represented as ordinary text so
    the exported QAIRT tokenizer does not need special ``tool_calls`` fields.
    """
    catalog = _tool_catalog(tools)
    protocol = TOOL_PROTOCOL.format(
        tools=json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))
    )
    prepared: list[dict] = []
    injected = False
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if role == "system" and not injected:
            prepared.append({"role": "system", "content": f"{content or ''}\n\n{protocol}"})
            injected = True
            continue
        calls = message.get("tool_calls") or []
        if role == "assistant" and calls:
            compact_calls = []
            for call in calls:
                fn = call.get("function", {})
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                compact_calls.append({"name": fn.get("name"), "arguments": args})
            content = json.dumps({"tool_calls": compact_calls}, ensure_ascii=False)
        if role == "tool":
            role = "user"
            content = f"TOOL_RESULT ({message.get('tool_call_id', 'unknown')}):\n{content or ''}"
        prepared.append({"role": role, "content": str(content or "")})
    if not injected:
        prepared.insert(0, {"role": "system", "content": protocol})
    return prepared


def parse_tool_response(text: str, tools: list[dict]) -> dict:
    """Convert constrained model JSON into an OpenAI assistant message.

    Only advertised tool names are returned. Invalid or ordinary model output
    becomes assistant content, so malformed generations can never dispatch an
    arbitrary MCP method.
    """
    allowed = {item["name"] for item in _tool_catalog(tools)}
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip()
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            obj = json.loads(_repair_delimiters(stripped))
        except (json.JSONDecodeError, TypeError):
            obj = None
    if obj is None:
        # Some Qwen generations wrap otherwise-valid JSON in <tool_call> or
        # explanatory markers. Extract one balanced object, but never execute
        # it unless its name still passes the advertised-tool allowlist below.
        start = stripped.find("{")
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)) if start >= 0 else ():
            char = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(stripped[start:index + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        if obj is None:
            return {"role": "assistant", "content": text or ""}

    calls = []
    for candidate in obj.get("tool_calls", []) if isinstance(obj, dict) else []:
        if not isinstance(candidate, dict) or candidate.get("name") not in allowed:
            continue
        args = candidate.get("arguments", {})
        if not isinstance(args, dict):
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": candidate["name"],
                "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":")),
            },
        })
    if calls:
        return {"role": "assistant", "content": None, "tool_calls": calls}
    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
        content = obj["content"]
        if "tool_calls" in content and content != stripped:
            nested = parse_tool_response(content, tools)
            if nested.get("tool_calls"):
                return nested
        return {"role": "assistant", "content": content}
    return {"role": "assistant", "content": text or ""}
