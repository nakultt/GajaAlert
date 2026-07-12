import json
import unittest

from gaja.tool_protocol import parse_tool_response, prepare_tool_messages


TOOLS = [{
    "type": "function",
    "function": {
        "name": "translate",
        "description": "Translate text",
        "parameters": {"type": "object", "properties": {}},
    },
}]


class ToolProtocolTests(unittest.TestCase):
    def test_injects_schema(self):
        messages = prepare_tool_messages([{"role": "user", "content": "hello"}], TOOLS)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn('"name":"translate"', messages[0]["content"])

    def test_valid_call_becomes_openai_shape(self):
        message = parse_tool_response(
            '{"tool_calls":[{"name":"translate","arguments":{"text":"alert"}}]}',
            TOOLS,
        )
        self.assertIsNone(message["content"])
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "translate")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"text": "alert"})

    def test_unadvertised_call_is_never_dispatched(self):
        message = parse_tool_response(
            '{"tool_calls":[{"name":"shell","arguments":{"cmd":"bad"}}]}',
            TOOLS,
        )
        self.assertNotIn("tool_calls", message)

    def test_wrapped_qwen_call_is_parsed(self):
        message = parse_tool_response(
            '<tool_call>\n{"tool_calls":[{"name":"translate","arguments":{}}]}\n</tool_call>',
            TOOLS,
        )
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "translate")

    def test_tool_call_nested_in_content_is_parsed(self):
        inner = '{"tool_calls":[{"name":"translate","arguments":{}}]}'
        message = parse_tool_response(json.dumps({"content": inner}), TOOLS)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "translate")

    def test_missing_call_closer_is_repaired_then_allowlisted(self):
        malformed = '{"tool_calls":[{"name":"translate","arguments":{"text":"x"}}]}'
        message = parse_tool_response(malformed, TOOLS)
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "translate")

    def test_tool_history_is_text_only(self):
        messages = prepare_tool_messages([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "translate", "arguments": '{"text":"x"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "translated"},
        ], TOOLS)
        self.assertIn("tool_calls", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "user")
        self.assertIn("TOOL_RESULT", messages[2]["content"])


if __name__ == "__main__":
    unittest.main()
