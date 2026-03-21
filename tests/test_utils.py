import unittest

from assistant.utils import (parse_json_payload, redact_secrets_obj,
                             redact_secrets_text, slugify)


class TestUtils(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello World"), "hello_world")
        self.assertEqual(slugify("Hello! World@"), "hello_world")
        self.assertEqual(slugify("  spaces  "), "spaces")
        self.assertEqual(slugify("---dash---"), "dash")

    def test_parse_json_payload(self):
        self.assertEqual(parse_json_payload('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_payload('Some text {"a": 1} more text'), {"a": 1})
        self.assertEqual(parse_json_payload('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIsNone(parse_json_payload("not json"))

    def test_redact_secrets_text(self):
        text = "My API_KEY=sk-1234567890abcdef and password=secret"
        redacted = redact_secrets_text(text)
        self.assertIn("API_KEY=***REDACTED***", redacted)
        self.assertIn("password=***REDACTED***", redacted)
        self.assertNotIn("sk-1234567890abcdef", redacted)

    def test_redact_secrets_obj(self):
        data = {
            "api_key": "secret-value",
            "nested": {"password": "12345"},
            "safe": "hello",
        }
        redacted = redact_secrets_obj(data)
        self.assertEqual(redacted["api_key"], "***REDACTED***")
        self.assertEqual(redacted["nested"]["password"], "***REDACTED***")
        self.assertEqual(redacted["safe"], "hello")


if __name__ == "__main__":
    unittest.main()
