import shutil
import unittest
from pathlib import Path

from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore
from assistant.tools import ToolSystem
from assistant.workspace_tools import WorkspaceTools


class TestSkillRegistry(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_skills_registry")
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = FunctionRegistry(self.root)
        self.memory = MemoryStore(self.root / "memory")
        self.workspace_tools = WorkspaceTools(self.root)
        self.tool_system = ToolSystem(self.memory, self.registry, self.workspace_tools)

    def tearDown(self):
        if self.root.exists():
            shutil.rmtree(self.root)

    def test_create_and_find_skill(self):
        created = self.registry.create_skill(
            name="Fix Import Error",
            description="Fix Python import path issues",
            keywords=["python", "import", "error"],
            tool_calls=[{"tool": "search_project", "args": {"query": "ImportError"}}],
        )
        self.assertTrue(created["ok"])
        listed = self.registry.find_skills("import")
        self.assertTrue(listed["ok"])
        self.assertGreaterEqual(listed["count"], 1)
        self.assertEqual(listed["skills"][0]["name"], "Fix Import Error")

    def test_record_skill_outcome(self):
        self.registry.create_skill(
            name="Run Tests",
            description="Execute pytest and parse failures",
            keywords=["pytest", "tests"],
            tool_calls=[{"tool": "run_tests", "args": {"path": ".", "runner": "pytest"}}],
        )
        outcome = self.registry.record_skill_outcome(
            name="Run Tests",
            success=True,
            confidence=0.85,
            notes="All tests green",
        )
        self.assertTrue(outcome["ok"])
        skill = outcome["skill"]
        self.assertEqual(skill["success_count"], 1)
        self.assertGreater(skill["confidence_sum"], 0.0)

    def test_create_skill_persists_extended_metadata(self):
        created = self.registry.create_skill(
            name="Auto Fix Auth",
            description="Auto-learned auth repair workflow",
            keywords=["auth", "repair"],
            tool_calls=[{"tool": "run_tests", "args": {"path": "."}}],
            skill="fix_auth",
            inputs=["file_path", "test_runner"],
            steps_template=[{"tool": "edit_file", "args": {"path": "${file_path}"}}],
            match_conditions=["auth", "pytest"],
        )
        self.assertTrue(created["ok"])
        skill = created["skill"]
        self.assertEqual(skill["skill"], "fix_auth")
        self.assertEqual(skill["inputs"], ["file_path", "test_runner"])
        self.assertEqual(
            skill["steps_template"],
            [{"tool": "edit_file", "args": {"path": "${file_path}"}}],
        )
        self.assertEqual(skill["match_conditions"], ["auth", "pytest"])

    def test_tool_system_create_skill_accepts_extended_args(self):
        created = self.tool_system.execute(
            "create_skill",
            {
                "name": "Auto Fix Demo",
                "description": "Autonomous repair workflow",
                "keywords": ["demo", "repair"],
                "tool_calls": [{"tool": "run_tests", "args": {"path": "."}}],
                "skill": "fix_demo",
                "inputs": ["file_path"],
                "steps_template": [
                    {"tool": "edit_file", "args": {"path": "${file_path}"}}
                ],
                "match_conditions": ["demo", "pytest"],
            },
        )
        self.assertTrue(created["ok"])
        skill = created["skill"]
        self.assertEqual(skill["skill"], "fix_demo")
        self.assertEqual(skill["inputs"], ["file_path"])
        self.assertEqual(skill["match_conditions"], ["demo", "pytest"])

    def test_create_function_auto_appends_agent_suffix(self):
        created = self.tool_system.execute(
            "create_function",
            {
                "name": "collect_logs",
                "description": "Collect logs",
                "keywords": ["logs"],
                "tool_calls": [{"tool": "list_files", "args": {"path": "."}}],
            },
        )
        self.assertTrue(created["ok"])
        self.assertIn("collect_logs_agent.json", created["metadata_file"])

    def test_create_function_rejects_system_tool_name_collision(self):
        created = self.tool_system.execute(
            "create_function",
            {
                "name": "search_web",
                "description": "bad",
                "keywords": ["bad"],
                "tool_calls": [{"tool": "list_files", "args": {"path": "."}}],
            },
        )
        self.assertFalse(created["ok"])
        self.assertIn("collides with system tool", created["error"])


if __name__ == "__main__":
    unittest.main()
