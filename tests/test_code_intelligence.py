import shutil
import unittest
from pathlib import Path

from assistant.workspace_tools import WorkspaceTools


class TestCodeIntelligence(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_workspace_tools")
        self.root.mkdir(parents=True, exist_ok=True)
        self.sample = self.root / "sample.py"
        self.sample.write_text(
            "import os\n"
            "\n"
            "class LoginManager:\n"
            "    def login_user(self, username: str) -> bool:\n"
            "        return bool(username)\n"
            "\n"
            "def helper_fn(value: int) -> int:\n"
            "    return value + 1\n",
            encoding="utf-8",
        )
        self.tools = WorkspaceTools(self.root)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_index_symbols(self):
        result = self.tools.index_symbols(path=".")
        self.assertTrue(result["ok"])
        names = {item["name"] for item in result["symbols"]}
        self.assertIn("LoginManager", names)
        self.assertIn("login_user", names)
        self.assertIn("helper_fn", names)

    def test_lookup_symbol(self):
        result = self.tools.lookup_symbol(symbol="login", path=".")
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["count"], 1)
        self.assertTrue(any("login" in item["name"].lower() for item in result["matches"]))

    def test_summarize_file(self):
        result = self.tools.summarize_file(path="sample.py")
        self.assertTrue(result["ok"])
        self.assertEqual(result["language"], "python")
        self.assertIn("symbols=", result["summary"])
        self.assertGreaterEqual(len(result["symbols"]), 2)


if __name__ == "__main__":
    unittest.main()
