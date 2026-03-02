import unittest
from assistant.tool_calls import parse_tool_calls

class TestToolCalls(unittest.TestCase):
    def test_parse_single_tool_call(self):
        text = '{"tool": "search_web", "args": {"query": "test"}}'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_web")
        self.assertEqual(calls[0]["args"], {"query": "test"})

    def test_parse_multi_tool_calls(self):
        text = '{"tool_calls": [{"name": "tool1", "args": {}}, {"tool": "tool2", "arguments": {"x": 1}}]}'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "tool1")
        self.assertEqual(calls[1]["name"], "tool2")
        self.assertEqual(calls[1]["args"], {"x": 1})

    def test_parse_list_of_calls(self):
        text = '[{"tool": "a"}, {"name": "b", "args": {"v": true}}]'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "a")
        self.assertEqual(calls[1]["name"], "b")
        self.assertEqual(calls[1]["args"], {"v": True})

    def test_invalid_json(self):
        self.assertEqual(parse_tool_calls("not json"), [])

    def test_missing_name(self):
        text = '{"args": {}}'
        self.assertEqual(parse_tool_calls(text), [])

    def test_malformed_args(self):
        text = '{"tool": "test", "args": "not a dict"}'
        self.assertEqual(parse_tool_calls(text), [])

if __name__ == "__main__":
    unittest.main()