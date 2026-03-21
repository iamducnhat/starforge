import shutil
import unittest
from pathlib import Path

from assistant.workspace_tools import WorkspaceTools


class TestWorkspaceAwareness(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_workspace_awareness")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "main.py").write_text(
            "from fastapi import FastAPI\napp = FastAPI()\n",
            encoding="utf-8",
        )
        (self.root / "requirements.txt").write_text("fastapi\npytest\n", encoding="utf-8")
        self.tools = WorkspaceTools(self.root)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_detect_project_context(self):
        ctx = self.tools.detect_project_context(path=".")
        self.assertTrue(ctx["ok"])
        self.assertEqual(ctx["framework"], "FastAPI")
        self.assertIn("main.py", ctx["entry_points"][0])
        self.assertEqual(ctx["test_runner"], "pytest")

    def test_execute_command_structured(self):
        result = self.tools.execute_command("echo hello", path=".")
        self.assertTrue(result["ok"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello", result["stdout"])


if __name__ == "__main__":
    unittest.main()
