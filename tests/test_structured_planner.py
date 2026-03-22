import unittest
from pathlib import Path
import shutil

from assistant.chat_engine import ChatEngine, PlanStep, TaskState
from assistant.functions_registry import FunctionRegistry
from assistant.memory import MemoryStore
from assistant.tools import ToolSystem
from assistant.workspace_tools import WorkspaceTools


class _DummyModel:
    def __init__(self, response: str = "{}"):
        self.response = response

    def generate(self, messages):
        return self.response

    def stream_generate(self, messages):  # pragma: no cover - not used here
        yield self.response


class _DummyTools:
    def __init__(self, memory_store=None):
        self.memory_store = memory_store


class TestStructuredPlanner(unittest.TestCase):
    def test_extract_explicit_file_paths_includes_directories(self):
        paths = ChatEngine._extract_explicit_file_paths(
            "check workspaces/crypto_research and read README.md"
        )
        self.assertIn("workspaces/crypto_research", paths)
        self.assertIn("README.md", paths)

    def test_preinspect_uses_list_files_for_explicit_directory_paths(self):
        root = Path("test_preinspect_directory_paths")
        memory = MemoryStore(root / "memory_blocks")
        registry = FunctionRegistry(root / "functions")
        workspace_tools = WorkspaceTools(root)
        try:
            target_dir = root / "workspaces" / "crypto_research"
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "FINAL_RANKING_MARCH_APRIL_2026.md").write_text(
                "# ranking\n", encoding="utf-8"
            )
            engine = ChatEngine(
                model=_DummyModel(),
                tools=ToolSystem(memory, registry, workspace_tools),
                system_prompt="test",
            )

            calls = engine._preinspect_tool_calls_for_workspace(
                "check the workspaces/crypto_research folder"
            )

            self.assertIn(
                {
                    "name": "list_files",
                    "args": {"path": "workspaces/crypto_research", "max_entries": 100},
                },
                calls,
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_preinspect_uses_read_file_for_explicit_file_paths(self):
        root = Path("test_preinspect_file_paths")
        memory = MemoryStore(root / "memory_blocks")
        registry = FunctionRegistry(root / "functions")
        workspace_tools = WorkspaceTools(root)
        try:
            target_file = root / "workspaces" / "crypto_research" / "FINAL_RANKING.md"
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text("# ranking\n", encoding="utf-8")
            engine = ChatEngine(
                model=_DummyModel(),
                tools=ToolSystem(memory, registry, workspace_tools),
                system_prompt="test",
            )

            calls = engine._preinspect_tool_calls_for_workspace(
                "read workspaces/crypto_research/FINAL_RANKING.md"
            )

            self.assertIn(
                {
                    "name": "read_file",
                    "args": {
                        "path": "workspaces/crypto_research/FINAL_RANKING.md",
                        "max_chars": 6000,
                    },
                },
                calls,
            )
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_explicit_workspace_file_path_skips_factual_web_presearch(self):
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(),
            system_prompt="test",
        )

        self.assertFalse(
            engine._requires_web_presearch_for_factual(
                "Read workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md and report what happens."
            )
        )

    def test_tool_event_uses_original_args_when_execution_rewrites_them(self):
        engine = ChatEngine(
            model=_DummyModel(
                response='{"tool":"read_file","args":{"path":"workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"}}'
            ),
            tools=_DummyTools(),
            system_prompt="test",
            max_tool_rounds=1,
        )

        def _fake_run_tool_call_with_reflection(user_message, call):
            return {
                "name": "read_file",
                "args": {"path": "workspaces/crypto_research"},
                "result": {"ok": False, "error": "not a file: workspaces/crypto_research"},
                "reflection": {"status": "failed", "retried": False},
                "initial_name": "read_file",
                "initial_args": {
                    "path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"
                },
            }

        engine._run_tool_call_with_reflection = _fake_run_tool_call_with_reflection

        starts = []
        events = []
        answer = engine.handle_turn(
            "Read workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md",
            on_tool_start=lambda name, args: starts.append((name, args)),
            on_tool=lambda name, args, result: events.append((name, args, result)),
            enforce_presearch=False,
            log_interaction=False,
        )

        self.assertEqual(answer, "Tool-call loop limit reached. Return direct answer.")
        self.assertEqual(
            starts,
            [
                (
                    "read_file",
                    {
                        "path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"
                    },
                )
            ],
        )
        self.assertEqual(
            events,
            [
                (
                    "read_file",
                    {
                        "path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"
                    },
                    {"ok": False, "error": "not a file: workspaces/crypto_research"},
                )
            ],
        )

    def test_retry_ignores_cross_tool_rewrite_args(self):
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(),
            system_prompt="test",
        )

        calls = []

        def _fake_execute_tool_call_with_policy(user_message, call):
            calls.append(call)
            if len(calls) == 1:
                return {"ok": False, "error": "first failure"}
            return {"ok": False, "error": "second failure"}

        def _fake_reflect_tool_result(user_message, call, result):
            return {
                "enabled": True,
                "status": "failed",
                "retry": True,
                "succeeded": False,
                "confidence": 1.0,
                "issues": [],
                "reason": "switch tool",
                "revised_call": {
                    "name": "list_files",
                    "args": {"path": "workspaces/crypto_research"},
                },
            }

        engine._execute_tool_call_with_policy = _fake_execute_tool_call_with_policy
        engine._reflect_tool_result = _fake_reflect_tool_result

        executed = engine._run_tool_call_with_reflection(
            "Read workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md",
            {
                "name": "read_file",
                "args": {
                    "path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"
                },
            },
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(calls[1]["name"], "read_file")
        self.assertEqual(
            calls[1]["args"],
            {"path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"},
        )
        self.assertEqual(
            executed.get("args"),
            {"path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"},
        )
        self.assertEqual(executed["reflection"].get("retry_name_ignored"), "list_files")

    def test_retry_accepts_revised_args_when_tool_name_matches(self):
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(),
            system_prompt="test",
        )

        calls = []

        def _fake_execute_tool_call_with_policy(user_message, call):
            calls.append(call)
            if len(calls) == 1:
                return {"ok": False, "error": "first failure"}
            return {"ok": True, "path": call.get("args", {}).get("path", "")}

        def _fake_reflect_tool_result(user_message, call, result):
            return {
                "enabled": True,
                "status": "failed",
                "retry": True,
                "succeeded": False,
                "confidence": 1.0,
                "issues": [],
                "reason": "retry with adjusted path",
                "revised_call": {
                    "name": "read_file",
                    "args": {
                        "path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"
                    },
                },
            }

        engine._execute_tool_call_with_policy = _fake_execute_tool_call_with_policy
        engine._reflect_tool_result = _fake_reflect_tool_result

        executed = engine._run_tool_call_with_reflection(
            "Read file",
            {"name": "read_file", "args": {"path": "workspaces/crypto_research"}},
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["name"], "read_file")
        self.assertEqual(
            calls[1]["args"],
            {"path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"},
        )
        self.assertEqual(
            executed.get("args"),
            {"path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"},
        )

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

    def test_coerce_next_gen_schema_aliases(self):
        payload = {
            "steps": [
                {
                    "id": 1,
                    "type": "tool_call",
                    "tool": "search_project",
                    "args": {"query": "login"},
                    "depends_on": [],
                    "expected": "list of files",
                },
                {
                    "id": 2,
                    "type": "tool_call",
                    "tool": "edit_file",
                    "args": {"path": "app.py"},
                    "depends_on": [1],
                    "expected": "patched code",
                },
            ]
        }
        steps = ChatEngine._coerce_structured_plan(payload, "fix auth flow")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_id, 1)
        self.assertEqual(steps[0].action, "search_project")
        self.assertEqual(steps[0].expected_output, "list of files")
        self.assertEqual(steps[1].depends_on, [1])

    def test_validate_plan_reorders_by_topological_dependencies(self):
        steps = [
            PlanStep(step_id=2, action="edit", depends_on=[1]),
            PlanStep(step_id=1, action="inspect", depends_on=[]),
        ]
        normalized = ChatEngine._validate_plan_steps(steps, "goal")
        self.assertEqual([step.step_id for step in normalized], [1, 2])
        self.assertEqual(normalized[1].depends_on, [1])

    def test_task_state_runnable_queue_respects_dependencies(self):
        state = TaskState(
            goal="goal",
            steps=[
                PlanStep(step_id=1, action="inspect", depends_on=[]),
                PlanStep(step_id=2, action="edit", depends_on=[1]),
                PlanStep(step_id=3, action="test", depends_on=[2]),
            ],
        )
        self.assertEqual(state.runnable_step_ids(), [1])
        state.completed_step_ids.add(1)
        self.assertEqual(state.runnable_step_ids(), [2])
        state.completed_step_ids.add(2)
        self.assertEqual(state.next_runnable_index(), 2)

    def test_validator_uses_structured_test_and_diff_signals(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(
            step_id=1,
            action="implement fix",
            args={},
            depends_on=[],
            expected_output="patched code and passing tests",
        )
        validation = engine._validate_autonomous_step_execution(
            step=step,
            final_text="Implemented the fix and verified it.",
            tool_payloads=[
                {
                    "tool": "run_tests",
                    "result": {
                        "ok": True,
                        "exit_code": 0,
                        "failed": 0,
                        "errors": 0,
                        "passed": 4,
                    },
                },
                {
                    "tool": "execute_command",
                    "result": {"ok": True, "exit_code": 0},
                },
            ],
            workspace_validation={
                "ok": True,
                "tests": {"passed": 4},
                "validation_signals": {
                    "tests_passed": True,
                    "test_exit_code": 0,
                    "failed_tests": 0,
                    "test_errors": 0,
                    "has_diff": True,
                    "changed_file_count": 1,
                },
            },
        )
        self.assertEqual(validation["status"], "success")
        self.assertTrue(validation["signals"]["diff_observed"])
        self.assertGreaterEqual(validation["score"], 0.67)

    def test_planner_reuses_strategy_memory_before_model_planning(self):
        root = Path("test_planner_strategy_memory")
        memory = MemoryStore(root / "memory_blocks")
        try:
            memory.record_strategy(
                goal="fix failing tests",
                strategy=[
                    {
                        "step_id": 1,
                        "action": "run_tests",
                        "args": {"path": ".", "runner": "pytest"},
                        "depends_on": [],
                        "expected_output": "failures identified",
                    },
                    {
                        "step_id": 2,
                        "action": "edit_file",
                        "args": {"path": "app.py"},
                        "depends_on": [1],
                        "expected_output": "patch applied",
                    },
                ],
                success=True,
            )
            engine = ChatEngine(
                model=_DummyModel(response='{"steps":[{"id":1,"tool":"wrong_tool","args":{}}]}'),
                tools=_DummyTools(memory_store=memory),
                system_prompt="test",
            )
            steps = engine._plan_objective_steps("fix failing tests")
            self.assertEqual(steps[0].action, "run_tests")
            self.assertEqual(steps[1].depends_on, [1])
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_hybrid_strategy_ranking_prefers_context_match(self):
        root = Path("test_hybrid_strategy_context")
        memory = MemoryStore(root / "memory_blocks")
        try:
            memory.record_strategy(
                goal="fix failing tests",
                strategy=[
                    {
                        "step_id": 1,
                        "action": "run_tests",
                        "args": {"path": ".", "runner": "pytest"},
                        "depends_on": [],
                        "expected_output": "failures identified",
                    }
                ],
                success=True,
                context={"framework": "pytest", "language": "python"},
            )
            memory.record_strategy(
                goal="fix failing tests",
                strategy=[
                    {
                        "step_id": 1,
                        "action": "execute_command",
                        "args": {"cmd": "npm test"},
                        "depends_on": [],
                        "expected_output": "node failures identified",
                    }
                ],
                success=True,
                context={"language": "node"},
            )
            engine = ChatEngine(
                model=_DummyModel(response='{"steps":[{"id":1,"tool":"wrong_tool","args":{}}]}'),
                tools=_DummyTools(memory_store=memory),
                system_prompt="test",
            )
            steps = engine._reuse_strategy_steps(
                "fix failing pytest tests",
                step_cap=3,
                runtime_context={"framework": "pytest", "language": "python"},
            )
            self.assertGreaterEqual(len(steps), 1)
            self.assertEqual(steps[0].action, "run_tests")
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_test_driven_repair_gate_and_prompt_use_real_failures(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(
            step_id=2,
            action="implement fix",
            args={"path": "app.py"},
            depends_on=[1],
            expected_output="tests pass",
        )
        workspace_validation = {
            "changed_files": ["app.py"],
            "diff_excerpt": "@@ -1 +1 @@\n-print('bad')\n+print('maybe')",
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [
                    {
                        "nodeid": "tests/test_app.py::test_login",
                        "message": "AssertionError: expected 200",
                        "summary": "FAILED tests/test_app.py::test_login - AssertionError: expected 200",
                    }
                ],
            },
            "tests": {"stderr": "", "stdout": ""},
        }
        self.assertTrue(
            engine._should_attempt_test_driven_repair(
                step=step,
                workspace_validation=workspace_validation,
                attempt_no=0,
            )
        )
        prompt = engine._build_test_repair_prompt(
            objective="fix failing login tests",
            step=step,
            workspace_validation=workspace_validation,
            attempt_no=0,
        )
        self.assertIn("tests/test_app.py::test_login", prompt)
        self.assertIn("expected 200", prompt)
        self.assertIn("Changed files: ['app.py']", prompt)

    def test_test_repair_prompt_includes_hypothesis_history(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix auth", expected_output="tests pass")
        workspace_validation = {
            "changed_files": ["auth.py"],
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [
                    {
                        "nodeid": "tests/test_auth.py::test_login",
                        "message": "AssertionError: expected token",
                        "summary": "FAILED tests/test_auth.py::test_login - AssertionError: expected token",
                    }
                ],
            },
        }
        prompt = engine._build_test_repair_prompt(
            objective="fix auth tests",
            step=step,
            workspace_validation=workspace_validation,
            attempt_no=1,
            hypothesis={
                "hypothesis": "Token is not returned from login path",
                "suspected_files": ["auth.py", "tests/test_auth.py"],
                "rationale": "The failing test expects a token but received none.",
                "next_check": "Inspect login return value.",
            },
            repair_history=[
                {
                    "attempt": 1,
                    "hypothesis": "Import path is broken",
                    "outcome": "unchanged",
                    "before_total": 1,
                    "after_total": 1,
                }
            ],
        )
        self.assertIn("Current debugging hypothesis: Token is not returned from login path", prompt)
        self.assertIn("Attempt 1: hypothesis=Import path is broken | outcome=unchanged | failures 1->1", prompt)

    def test_test_failure_hypothesis_fallback_uses_failure_summary(self):
        engine = ChatEngine(model=_DummyModel(response="not json"), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix auth")
        workspace_validation = {
            "changed_files": ["auth.py"],
            "validation_signals": {
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [
                    {
                        "nodeid": "tests/test_auth.py::test_login",
                        "message": "AssertionError: expected token",
                        "summary": "FAILED tests/test_auth.py::test_login - AssertionError: expected token",
                    }
                ],
            },
        }
        hypothesis = engine._propose_test_failure_hypothesis(
            objective="fix auth tests",
            step=step,
            workspace_validation=workspace_validation,
            repair_history=[],
        )
        self.assertIn("expected token", hypothesis["hypothesis"])
        self.assertEqual(hypothesis["suspected_files"], ["auth.py"])

    def test_repair_loop_uses_root_cause_before_hypothesis(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix imports")
        workspace_validation = {
            "validation_signals": {"tests_passed": False, "failed_tests": 1, "test_errors": 0}
        }
        engine._maybe_apply_root_cause_fix = lambda **kwargs: (
            True,
            [{"tool": "execute_command", "args": {"cmd": "pip install requests"}, "result": {"ok": True}}],
            {"validation_signals": {"tests_passed": True, "failed_tests": 0, "test_errors": 0}},
            {"kind": "root_cause", "pattern": "ModuleNotFoundError: ${module}"},
        )
        engine._propose_test_failure_hypothesis = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("hypothesis should not be generated when root cause resolves")
        )
        text, payloads, validation, history = engine._maybe_run_test_driven_repair(
            objective="fix import failures",
            step=step,
            workspace_validation=workspace_validation,
            current_tool_payloads=[],
            project_context={"framework": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0, "test_repair_attempts": 0},
        )
        self.assertIn("root-cause fix template", text or "")
        self.assertEqual(len(payloads), 1)
        self.assertEqual(history[0]["kind"], "root_cause")
        self.assertTrue(validation["validation_signals"]["tests_passed"])

    def test_repair_loop_skips_duplicate_hypotheses(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix auth")
        workspace_validation = {
            "validation_signals": {"tests_passed": False, "failed_tests": 1, "test_errors": 0}
        }
        engine.autonomous_test_repair_attempts = 2
        engine._maybe_apply_root_cause_fix = lambda **kwargs: (
            False,
            [],
            workspace_validation,
            None,
        )
        engine._propose_test_failure_hypothesis = lambda **kwargs: {
            "hypothesis": "missing dependency",
            "suspected_files": ["auth.py"],
            "confidence": 0.7,
        }
        engine.handle_turn_stream = lambda *args, **kwargs: "patched"
        engine._recent_tool_payloads = lambda limit=8: []
        engine._maybe_validate_workspace_after_step = lambda **kwargs: workspace_validation

        _, _, _, history = engine._maybe_run_test_driven_repair(
            objective="fix auth tests",
            step=step,
            workspace_validation=workspace_validation,
            current_tool_payloads=[],
            project_context={},
            auto_metrics={
                "tool_calls": 0,
                "failed_tool_calls": 0,
                "test_repair_attempts": 0,
                "est_tokens_out": 0,
            },
        )
        self.assertGreaterEqual(len(history), 2)
        self.assertEqual(history[-1].get("skipped"), "duplicate_hypothesis")


if __name__ == "__main__":
    unittest.main()
