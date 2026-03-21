import shutil
import unittest
from pathlib import Path

from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore
from assistant.tools import ToolSystem
from assistant.workspace_tools import WorkspaceTools


class TestToolReliability(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_tool_reliability")
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.root / "memory_blocks"
        self.functions_dir = self.root / "functions"
        self.memory = MemoryStore(self.memory_dir)
        self.registry = FunctionRegistry(self.functions_dir)
        self.workspace_tools = WorkspaceTools(self.root)
        self.tool_system = ToolSystem(self.memory, self.registry, self.workspace_tools)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_safe_tool_call_retries_transient_failures(self):
        calls = {"count": 0}

        def flaky_tool():
            calls["count"] += 1
            if calls["count"] == 1:
                return {"ok": False, "error": "timeout while connecting"}
            return {"ok": True, "value": "recovered"}

        self.tool_system._tools["flaky_tool"] = flaky_tool
        result = self.tool_system.execute("flaky_tool", {})
        self.assertTrue(result["ok"])
        self.assertEqual(result.get("value"), "recovered")
        self.assertEqual(result.get("attempts"), 2)


if __name__ == "__main__":
    unittest.main()
