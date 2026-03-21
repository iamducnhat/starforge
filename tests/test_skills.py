import shutil
import unittest
from pathlib import Path

from assistant.functions_registry import FunctionRegistry


class TestSkillRegistry(unittest.TestCase):
    def setUp(self):
        self.root = Path("test_skills_registry")
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = FunctionRegistry(self.root)

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


if __name__ == "__main__":
    unittest.main()
