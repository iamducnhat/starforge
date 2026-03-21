import unittest

from assistant.chat_engine import ChatEngine, PlanStep


class TestStructuredPlanner(unittest.TestCase):
    def test_coerce_structured_plan(self):
        payload = {
            "steps": [
                {
                    "step_id": 1,
                    "action": "search_project",
                    "args": {"query": "login"},
                    "depends_on": [],
                    "expected_output": "candidate files",
                },
                {
                    "step_id": 2,
                    "action": "edit_file",
                    "args": {"path": "app.py"},
                    "depends_on": [1],
                    "expected_output": "patched logic",
                },
            ]
        }
        steps = ChatEngine._coerce_structured_plan(payload, "fix login bug")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_id, 1)
        self.assertEqual(steps[1].depends_on, [1])
        self.assertEqual(steps[0].action, "search_project")

    def test_validate_plan_breaks_invalid_dependencies(self):
        steps = [
            PlanStep(step_id=1, action="a", depends_on=[3]),
            PlanStep(step_id=2, action="b", depends_on=[1]),
            PlanStep(step_id=3, action="c", depends_on=[3, 2]),
        ]
        normalized = ChatEngine._validate_plan_steps(steps, "goal")
        self.assertEqual(len(normalized), 3)
        # step 1 cannot depend on a future id after validation
        self.assertEqual(normalized[0].depends_on, [])
        # self-dependency is removed
        self.assertNotIn(normalized[2].step_id, normalized[2].depends_on)


if __name__ == "__main__":
    unittest.main()
