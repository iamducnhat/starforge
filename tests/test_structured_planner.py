import json
import unittest
from pathlib import Path
import shutil

import assistant.chat_engine as chat_engine_module
from assistant.chat_engine import (
    ChatEngine,
    PlanStep,
    ResolutionContext,
    TaskState,
    compute_confidence,
    resolution_succeeded,
)
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


class _TimeoutCaptureModel:
    def __init__(self):
        self.timeout = 120
        self.seen_timeout = None

    def generate(self, messages):
        self.seen_timeout = self.timeout
        return '{"steps":[]}'

    def stream_generate(self, messages):  # pragma: no cover - not used here
        yield '{"steps":[]}'


class _AutoTools:
    def __init__(self, memory_store=None):
        self.memory_store = memory_store

    def execute(self, name, args):
        if name == "detect_project_context":
            return {"ok": False, "framework": "unknown", "test_runner": "unknown"}
        return {"ok": True}


class _CaptureTools:
    def __init__(self):
        self.calls = []
        self.memory_store = None

    def execute(self, name, args):
        self.calls.append((name, args))
        return {"ok": True, "name": name}


class _ImmediateFixTools:
    def __init__(self):
        self.calls = []
        self.memory_store = None

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "search_project":
            query = str(args.get("query", ""))
            if query == "def add(":
                return {
                    "ok": True,
                    "count": 1,
                    "matches": [
                        {
                            "path": "demo_project/utils.py",
                            "line": 1,
                            "text": "def add(a: int, b: int) -> int:",
                        }
                    ],
                }
            return {"ok": True, "count": 0, "matches": []}
        if name == "read_file":
            path = str(args.get("path", ""))
            if path == "demo_project/utils.py":
                return {
                    "ok": True,
                    "path": path,
                    "start_line": args.get("start_line", 1),
                    "end_line": args.get("end_line", 21),
                    "content": "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
                }
            return {"ok": False, "error": f"unexpected path: {path}"}
        if name == "edit_file":
            return {"ok": True, "path": args.get("path", ""), "replacements": 1}
        if name == "validate_workspace_changes":
            return {
                "ok": True,
                "tests_passed": True,
                "changed_files": [args.get("focus_paths", [""])[0]],
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": True,
                    "failed_tests": 0,
                    "test_errors": 0,
                    "has_diff": True,
                    "changed_file_count": 1,
                },
            }
        return {"ok": True}


class _StubLearningStore:
    def __init__(self, matches=None):
        self.matches = matches or []
        self.upserts = []
        self.repair_pattern_matches = []
        self.repair_patterns = []

    def find_root_causes(self, error_text, context, limit=1):
        return list(self.matches)[: max(1, int(limit))]

    def upsert_root_cause(self, **kwargs):
        self.upserts.append(dict(kwargs))
        return {"ok": True, "created": True, "entry": kwargs}

    def find_repair_patterns(self, pattern, before="", context="", limit=1):
        del pattern, before, context
        return list(self.repair_pattern_matches)[: max(1, int(limit))]

    def match_repair_pattern(self, failure_signature, limit=1):
        del failure_signature
        return list(self.repair_pattern_matches)[: max(1, int(limit))]

    def record_repair_pattern(self, **kwargs):
        self.repair_patterns.append(dict(kwargs))
        return {"ok": True, "created": True, "entry": kwargs}


class TestStructuredPlanner(unittest.TestCase):
    def test_resolution_succeeded_accepts_non_test_signals(self):
        self.assertTrue(
            resolution_succeeded(
                ResolutionContext(build_passed=True, command_exit_code=0)
            )
        )
        self.assertTrue(
            resolution_succeeded(
                ResolutionContext(output_valid=True)
            )
        )
        self.assertFalse(resolution_succeeded(ResolutionContext(command_exit_code=1)))

    def test_compute_confidence_requires_strong_signal_for_learning(self):
        self.assertLess(compute_confidence(ResolutionContext(command_exit_code=0)), 0.8)
        self.assertGreaterEqual(
            compute_confidence(ResolutionContext(output_valid=True)),
            0.8,
        )
        self.assertGreaterEqual(
            compute_confidence(ResolutionContext(tests_passed=True)),
            0.8,
        )

    def test_autonomous_blocks_todo_plan_tools(self):
        tools = _CaptureTools()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=tools,
            system_prompt="test",
        )
        engine._autonomous_execution_active = True
        result = engine._execute_tool_call_with_policy(
            "auto run",
            {"name": "update_todo", "args": {"plan_id": "1", "todo_id": 1, "status": "done"}},
        )
        self.assertTrue(result["ok"])
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("policy"), "autonomous_no_todo_tools")
        self.assertEqual(tools.calls, [])

    def test_autonomous_allows_non_todo_tools(self):
        tools = _CaptureTools()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=tools,
            system_prompt="test",
        )
        engine._autonomous_execution_active = True
        result = engine._execute_tool_call_with_policy(
            "auto run",
            {"name": "read_file", "args": {"path": "README.md"}},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(tools.calls, [("read_file", {"path": "README.md"})])

    def test_extract_explicit_file_paths_includes_directories(self):
        paths = ChatEngine._extract_explicit_file_paths(
            "check workspaces/crypto_research and read README.md"
        )
        self.assertIn("workspaces/crypto_research", paths)
        self.assertIn("README.md", paths)

    def test_looks_familiar_task_detects_reuse_markers(self):
        self.assertTrue(ChatEngine._looks_familiar_task("Can you solve this again?"))
        self.assertTrue(
            ChatEngine._looks_familiar_task("Use a similar fix as before for this bug")
        )
        self.assertFalse(ChatEngine._looks_familiar_task("Explain TLS certificates"))

    def test_bias_memory_before_web_adds_find_memory_for_familiar_search(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        calls = [{"name": "search_web", "args": {"query": "fix import cycle"}}]

        biased = engine._bias_memory_before_web_search(
            user_message="This looks similar to the issue we solved last time",
            tool_calls=calls,
            memory_checked=False,
        )

        self.assertEqual(biased[0]["name"], "find_in_memory")
        self.assertEqual(biased[1]["name"], "search_web")

    def test_bias_memory_before_web_respects_existing_memory_call(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        calls = [
            {"name": "find_in_memory", "args": {"keywords": ["import", "cycle"]}},
            {"name": "search_web", "args": {"query": "python import cycle fix"}},
        ]

        biased = engine._bias_memory_before_web_search(
            user_message="Find the best fix",
            tool_calls=calls,
            memory_checked=False,
        )

        self.assertEqual(biased, calls)

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

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_file")
        self.assertEqual(
            executed.get("args"),
            {"path": "workspaces/crypto_research/FINAL_RANKING_MARCH_APRIL_2026.md"},
        )
        self.assertEqual(executed["reflection"].get("retry_name_ignored"), "list_files")
        self.assertEqual(executed["reflection"].get("retry_skipped"), "duplicate_call")

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

    def test_autonomous_policy_caches_duplicate_reads(self):
        tools = _CaptureTools()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=tools,
            system_prompt="test",
        )
        engine._autonomous_execution_active = True

        first = engine._execute_tool_call_with_policy(
            "read file",
            {"name": "read_file", "args": {"path": "README.md"}},
        )
        second = engine._execute_tool_call_with_policy(
            "read file",
            {"name": "read_file", "args": {"path": "README.md"}},
        )

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(second.get("cached"))
        self.assertEqual(tools.calls, [("read_file", {"path": "README.md"})])

    def test_autonomous_read_file_is_windowed_after_search_hit(self):
        class _SearchAwareTools:
            def __init__(self):
                self.calls = []
                self.memory_store = None

            def execute(self, name, args):
                self.calls.append((name, dict(args)))
                if name == "search_project":
                    return {
                        "ok": True,
                        "count": 1,
                        "matches": [{"path": "demo_project/utils.py", "line": 40, "text": "def add("}],
                    }
                if name == "read_file":
                    return {"ok": True, "path": args.get("path", ""), "content": "snippet"}
                return {"ok": True}

        tools = _SearchAwareTools()
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        engine._autonomous_execution_active = True

        engine._execute_tool_call_with_policy(
            "auto run",
            {"name": "search_project", "args": {"query": "def add(", "path": "."}},
        )
        engine._execute_tool_call_with_policy(
            "auto run",
            {"name": "read_file", "args": {"path": "demo_project/utils.py"}},
        )

        self.assertEqual(
            tools.calls[-1],
            (
                "read_file",
                {
                    "path": "demo_project/utils.py",
                    "start_line": 20,
                    "end_line": 60,
                    "max_chars": 4000,
                },
            ),
        )

    def test_retry_skips_duplicate_failing_execute_command(self):
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(),
            system_prompt="test",
        )

        calls = []

        def _fake_execute_tool_call_with_policy(user_message, call):
            calls.append(call)
            return {"ok": False, "exit_code": 1, "stderr": "tests failed"}

        def _fake_reflect_tool_result(user_message, call, result):
            return {
                "enabled": True,
                "status": "failed",
                "retry": True,
                "succeeded": False,
                "confidence": 0.2,
                "issues": [],
                "reason": "retry pytest",
                "revised_call": {"name": "execute_command", "args": dict(call.get("args", {}))},
            }

        engine._execute_tool_call_with_policy = _fake_execute_tool_call_with_policy
        engine._reflect_tool_result = _fake_reflect_tool_result

        executed = engine._run_tool_call_with_reflection(
            "run tests",
            {"name": "execute_command", "args": {"cmd": "pytest -q demo_project/tests"}},
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(executed["reflection"].get("retry_skipped"), "duplicate_call")

    def test_collect_validation_signals_tracks_validation_completed_separately(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        signals = engine._collect_validation_signals(
            step=PlanStep(step_id=1, action="implement fix"),
            tool_payloads=[],
            workspace_validation={
                "ok": True,
                "tests_passed": False,
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "failure_mode": "assertion_failures",
                    "has_diff": True,
                    "changed_file_count": 1,
                },
            },
        )
        self.assertTrue(signals["workspace_validation_completed"])
        self.assertIn("assertion_failures", signals["failure_modes"])

    def test_infer_validation_test_args_prefers_recent_pytest_scope(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        inferred = engine._infer_validation_test_args(
            "fix failing tests in demo_project only",
            [
                {
                    "tool": "execute_command",
                    "args": {"cmd": "pytest -q demo_project/tests"},
                    "result": {"ok": False, "exit_code": 1},
                }
            ],
        )
        self.assertEqual(inferred, "demo_project/tests")

    def test_infer_search_root_from_test_scope(self):
        self.assertEqual(
            ChatEngine._infer_search_root_from_test_scope("demo_project/tests"),
            "demo_project",
        )
        self.assertEqual(
            ChatEngine._infer_search_root_from_test_scope("demo_project/tests/test_main.py::test_add"),
            "demo_project/tests",
        )

    def test_build_autonomous_step_contract_for_single_target_failure(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        contract = engine._build_autonomous_step_contract(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="run_tests"),
            workspace_validation={
                "validation_signals": {
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {
                            "nodeid": "demo_project/tests/test_main.py::test_add",
                            "summary": "FAILED demo_project/tests/test_main.py::test_add - assert -1 == 5",
                        }
                    ],
                }
            },
            tool_payloads=[],
        )
        self.assertEqual(contract["mode"], "single_target_test_debug")
        self.assertEqual(contract["tool_budget"], 4)
        self.assertEqual(contract["analysis_budget"], 1)

    def test_generate_model_text_restores_timeout_after_bounded_call(self):
        model = _TimeoutCaptureModel()
        engine = ChatEngine(model=model, tools=_DummyTools(), system_prompt="test")
        text = engine._generate_model_text(
            [{"role": "user", "content": "plan"}],
            timeout=9,
        )
        self.assertEqual(text, '{"steps":[]}')
        self.assertEqual(model.seen_timeout, 9)
        self.assertEqual(model.timeout, 120)

    def test_execution_contract_violation_detects_unpatched_trivial_loop(self):
        violation = ChatEngine._execution_contract_violation_reason(
            {
                "mode": "single_target_test_debug",
                "tool_budget": 4,
                "require_search_before_read": True,
            },
            {
                "total": 5,
                "search_count": 1,
                "read_count": 2,
                "edit_count": 0,
                "duplicate_reads": 1,
                "tests_green": False,
            },
        )
        self.assertIn("re-read the same file", violation)

    def test_summarize_tool_activity_tracks_real_repeated_reads(self):
        summary = ChatEngine._summarize_tool_activity(
            [
                {
                    "tool": "lookup_symbol",
                    "args": {"symbol": "add", "path": "demo_project"},
                    "result": {"ok": True, "duplicate_call": True},
                },
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {"ok": True, "duplicate_call": True},
                },
                {
                    "tool": "edit_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {"ok": True},
                },
            ],
            workspace_validation={"tests_passed": True},
        )
        self.assertEqual(summary["duplicate_reads"], 0)
        self.assertEqual(summary["duplicate_calls"], 0)
        self.assertEqual(summary["validation_count"], 1)
        self.assertTrue(summary["tests_green"])

    def test_objective_tests_green_uses_targeted_pytest_command(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        self.assertTrue(
            engine._objective_tests_green(
                objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
                tool_payloads=[
                    {
                        "tool": "execute_command",
                        "args": {"cmd": "pytest -q demo_project/tests"},
                        "result": {"ok": True, "exit_code": 0},
                    }
                ],
                workspace_validation=None,
            )
        )

    def test_objective_tests_green_rejects_unrepaired_failed_edit(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        self.assertFalse(
            engine._objective_tests_green(
                objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
                tool_payloads=[
                    {
                        "tool": "edit_file",
                        "args": {"path": "demo_project/utils.py"},
                        "result": {"ok": False, "error": "find_text not found"},
                    }
                ],
                workspace_validation={
                    "ok": True,
                    "tests_passed": True,
                    "tests": {"command": "pytest -q demo_project/tests"},
                    "validation_signals": {
                        "validation_completed": True,
                        "tests_passed": True,
                        "failed_tests": 0,
                        "test_errors": 0,
                    },
                },
            )
        )

    def test_objective_tests_green_allows_later_successful_edit(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        self.assertTrue(
            engine._objective_tests_green(
                objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
                tool_payloads=[
                    {
                        "tool": "edit_file",
                        "args": {"path": "demo_project/utils.py"},
                        "result": {"ok": False, "error": "find_text not found"},
                    },
                    {
                        "tool": "edit_file",
                        "args": {"path": "demo_project/utils.py"},
                        "result": {"ok": True},
                    },
                ],
                workspace_validation={
                    "ok": True,
                    "tests_passed": True,
                    "tests": {"command": "pytest -q demo_project/tests"},
                    "validation_signals": {
                        "validation_completed": True,
                        "tests_passed": True,
                        "failed_tests": 0,
                        "test_errors": 0,
                    },
                },
            )
        )

    def test_should_run_validation_tests_ignores_read_only_fix_turn(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        should_validate = engine._should_run_validation_tests(
            PlanStep(step_id=1, action="fix auth"),
            payloads=[
                {
                    "tool": "read_file",
                    "args": {"path": "auth.py"},
                    "result": {"ok": True},
                }
            ],
        )
        self.assertFalse(should_validate)

    def test_validate_workspace_after_step_reuses_target_test_scope(self):
        tools = _CaptureTools()
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        result = engine._maybe_validate_workspace_after_step(
            step=PlanStep(step_id=1, action="apply fix"),
            project_context={"test_runner": "pytest"},
            payloads=[
                {
                    "tool": "edit_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {"ok": True},
                },
                {
                    "tool": "execute_command",
                    "args": {"cmd": "pytest -q demo_project/tests"},
                    "result": {"ok": False, "exit_code": 1},
                },
            ],
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            tools.calls[-1],
            (
                "validate_workspace_changes",
                {
                    "path": ".",
                    "test_runner": "pytest",
                    "test_args": "demo_project/tests",
                    "timeout": 180,
                    "focus_paths": ["demo_project/utils.py"],
                },
            ),
        )

    def test_recent_tool_payloads_keeps_interleaved_tool_history(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        engine.history = [
            {"role": "assistant", "content": "{\"tool\":\"run_tests\",\"args\":{}}"},
            {
                "role": "tool",
                "content": json.dumps({"tool": "run_tests", "args": {"path": "."}, "result": {"ok": True}}),
            },
            {"role": "assistant", "content": "status"},
            {
                "role": "tool",
                "content": json.dumps({"tool": "read_file", "args": {"path": "demo_project/utils.py"}, "result": {"ok": True}}),
            },
        ]
        payloads = engine._recent_tool_payloads(limit=8)
        self.assertEqual([item["tool"] for item in payloads], ["run_tests", "read_file"])

    def test_build_test_repair_prompt_includes_execution_contract(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        prompt = engine._build_test_repair_prompt(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="fix failing test"),
            workspace_validation={
                "validation_signals": {
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {
                            "nodeid": "demo_project/tests/test_main.py::test_add",
                            "message": "assert -1 == 5",
                        }
                    ],
                }
            },
            attempt_no=0,
        )
        self.assertIn("Execution contract for this step:", prompt)
        self.assertIn("Tool budget for this step: about 4 calls", prompt)
        self.assertIn("Stop immediately once targeted tests are green", prompt)

    def test_validator_reports_collection_error_not_zero_counts_noise(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        validation = engine._validate_autonomous_step_execution(
            step=PlanStep(step_id=1, action="implement fix"),
            final_text="Tried to fix it.",
            tool_payloads=[],
            workspace_validation={
                "ok": True,
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": False,
                    "failed_tests": 0,
                    "test_errors": 1,
                    "failure_mode": "collection_error",
                    "has_diff": True,
                    "changed_file_count": 1,
                },
            },
        )
        self.assertIn("pytest collection failed", validation["issues"])

    def test_immediate_fix_trigger_patches_operator_mismatch(self):
        tools = _ImmediateFixTools()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=tools,
            system_prompt="test",
        )
        solved, payloads, validation, history = engine._maybe_apply_immediate_fix_from_observation(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="inspect failing tests"),
            workspace_validation={
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {
                            "nodeid": "demo_project/tests/test_main.py::test_add",
                            "message": "AssertionError: assert -1 == 5",
                            "summary": "FAILED demo_project/tests/test_main.py::test_add - AssertionError: assert -1 == 5",
                        }
                    ],
                }
            },
            current_tool_payloads=[
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/tests/test_main.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/tests/test_main.py",
                        "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    },
                },
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/utils.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
                    },
                },
            ],
            project_context={"test_runner": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
        )
        self.assertTrue(solved)
        self.assertEqual(payloads[0]["tool"], "edit_file")
        self.assertEqual(payloads[-1]["args"]["path"], "demo_project/utils.py")
        self.assertFalse(any(item["tool"] == "search_project" for item in payloads))
        self.assertTrue(validation["validation_signals"]["tests_passed"])
        self.assertEqual(history["kind"], "immediate_fix")

    def test_extract_root_cause_error_text_uses_failing_run_tests_payload(self):
        text = ChatEngine._extract_root_cause_error_text(
            None,
            tool_payloads=[
                {
                    "tool": "run_tests",
                    "result": {
                        "ok": True,
                        "tests_passed": False,
                        "stdout": "FAILED test_main.py::test_add - assert -1 == 5",
                        "stderr": "",
                        "test_failures": [
                            {
                                "nodeid": "test_main.py::test_add",
                                "message": "assert -1 == 5",
                                "summary": "FAILED test_main.py::test_add - assert -1 == 5",
                            }
                        ],
                    },
                }
            ],
        )
        self.assertIn("assert -1 == 5", text)

    def test_immediate_fix_trigger_can_read_context_from_run_tests_payload(self):
        class _SupplementTools:
            def __init__(self):
                self.calls = []
                self.memory_store = None

            def execute(self, name, args):
                self.calls.append((name, dict(args)))
                if name == "search_project":
                    if str(args.get("query", "")) == "def add(":
                        return {
                            "ok": True,
                            "count": 1,
                            "matches": [
                                {
                                    "path": "demo_project/utils.py",
                                    "line": 1,
                                    "text": "def add(a: int, b: int) -> int:",
                                }
                            ],
                        }
                    return {"ok": True, "count": 0, "matches": []}
                if name == "read_file":
                    path = str(args.get("path", ""))
                    if path == "demo_project/tests/test_main.py":
                        return {
                            "ok": True,
                            "path": path,
                            "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                        }
                    if path == "demo_project/utils.py":
                        return {
                            "ok": True,
                            "path": path,
                            "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                        }
                if name == "edit_file":
                    return {"ok": True, "path": args.get("path", ""), "replacements": 1}
                if name == "validate_workspace_changes":
                    return {
                        "ok": True,
                        "tests_passed": True,
                        "validation_signals": {
                            "validation_completed": True,
                            "tests_passed": True,
                            "failed_tests": 0,
                            "test_errors": 0,
                            "has_diff": True,
                            "changed_file_count": 1,
                        },
                    }
                return {"ok": True}

        tools = _SupplementTools()
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        solved, payloads, validation, history = engine._maybe_apply_immediate_fix_from_observation(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="inspect failing tests"),
            workspace_validation=None,
            current_tool_payloads=[
                {
                    "tool": "run_tests",
                    "args": {"path": "demo_project/tests", "runner": "pytest", "args": "-q"},
                    "result": {
                        "ok": True,
                        "tests_passed": False,
                        "path": "demo_project/tests",
                        "test_failures": [
                            {
                                "nodeid": "test_main.py::test_add",
                                "message": "assert -1 == 5",
                                "summary": "FAILED test_main.py::test_add - assert -1 == 5",
                            }
                        ],
                        "stdout": "FAILED test_main.py::test_add - assert -1 == 5",
                        "stderr": "",
                    },
                }
            ],
            project_context={"test_runner": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
        )
        self.assertTrue(solved)
        self.assertEqual(payloads[0]["tool"], "search_project")
        self.assertEqual(payloads[0]["args"]["path"], "demo_project")
        self.assertEqual(payloads[1]["tool"], "read_file")
        self.assertTrue(any(item["tool"] == "edit_file" for item in payloads))
        self.assertTrue(validation["validation_signals"]["tests_passed"])
        self.assertEqual(history["kind"], "immediate_fix")

    def test_immediate_fix_context_reads_reuse_existing_step_reads(self):
        class _CachedReadTools:
            def __init__(self):
                self.calls = []
                self.memory_store = None

            def execute(self, name, args):
                self.calls.append((name, dict(args)))
                if name == "search_project":
                    if str(args.get("query", "")) == "def add(":
                        return {
                            "ok": True,
                            "count": 1,
                            "matches": [
                                {
                                    "path": "demo_project/utils.py",
                                    "line": 1,
                                    "text": "def add(a: int, b: int) -> int:",
                                }
                            ],
                        }
                    return {"ok": True, "count": 0, "matches": []}
                if name == "read_file":
                    path = str(args.get("path", ""))
                    if path == "demo_project/utils.py":
                        return {
                            "ok": True,
                            "path": path,
                            "content": "def add(a: int, b: int) -> int:\n    return a - b\n",
                        }
                if name == "edit_file":
                    return {"ok": True, "path": args.get("path", ""), "replacements": 1}
                if name == "validate_workspace_changes":
                    return {
                        "ok": True,
                        "tests_passed": True,
                        "tests": {"command": "pytest -q demo_project/tests"},
                        "validation_signals": {
                            "validation_completed": True,
                            "tests_passed": True,
                            "failed_tests": 0,
                            "test_errors": 0,
                            "has_diff": True,
                            "changed_file_count": 1,
                        },
                    }
                return {"ok": True}

        tools = _CachedReadTools()
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        solved, _, _, _ = engine._maybe_apply_immediate_fix_from_observation(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="inspect failing tests"),
            workspace_validation=None,
            current_tool_payloads=[
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/tests/test_main.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/tests/test_main.py",
                        "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    },
                },
                {
                    "tool": "run_tests",
                    "args": {"path": "demo_project/tests", "runner": "pytest", "args": "-q"},
                    "result": {
                        "ok": True,
                        "tests_passed": False,
                        "path": "demo_project/tests",
                        "test_failures": [
                            {
                                "nodeid": "test_main.py::test_add",
                                "message": "assert -1 == 5",
                                "summary": "FAILED test_main.py::test_add - assert -1 == 5",
                            }
                        ],
                        "stdout": "FAILED test_main.py::test_add - assert -1 == 5",
                        "stderr": "",
                    },
                },
            ],
            project_context={"test_runner": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
        )

        self.assertTrue(solved)
        read_calls = [call for call in tools.calls if call[0] == "read_file"]
        self.assertEqual(
            read_calls,
            [
                (
                    "read_file",
                    {
                        "path": "demo_project/utils.py",
                        "start_line": 1,
                        "end_line": 21,
                        "max_chars": 4000,
                    },
                )
            ],
        )

    def test_immediate_fix_emits_visible_tool_events(self):
        tools = _ImmediateFixTools()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=tools,
            system_prompt="test",
        )
        starts = []
        events = []
        original_start = chat_engine_module.print_tool_start
        original_event = chat_engine_module.print_tool_event
        chat_engine_module.print_tool_start = lambda name, args: starts.append((name, dict(args)))
        chat_engine_module.print_tool_event = (
            lambda name, args, result: events.append((name, dict(args), dict(result)))
        )
        try:
            solved, payloads, validation, history = engine._maybe_apply_immediate_fix_from_observation(
                objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
                step=PlanStep(step_id=1, action="inspect failing tests"),
                workspace_validation={
                    "validation_signals": {
                        "validation_completed": True,
                        "tests_passed": False,
                        "failed_tests": 1,
                        "test_errors": 0,
                        "test_failures": [
                            {
                                "nodeid": "demo_project/tests/test_main.py::test_add",
                                "message": "AssertionError: assert -1 == 5",
                                "summary": "FAILED demo_project/tests/test_main.py::test_add - AssertionError: assert -1 == 5",
                            }
                        ],
                    }
                },
                current_tool_payloads=[
                    {
                        "tool": "read_file",
                        "args": {"path": "demo_project/tests/test_main.py"},
                        "result": {
                            "ok": True,
                            "path": "demo_project/tests/test_main.py",
                            "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                        },
                    },
                    {
                        "tool": "read_file",
                        "args": {"path": "demo_project/utils.py"},
                        "result": {
                            "ok": True,
                            "path": "demo_project/utils.py",
                            "content": "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
                        },
                    },
                ],
                project_context={"test_runner": "pytest"},
                auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
            )
        finally:
            chat_engine_module.print_tool_start = original_start
            chat_engine_module.print_tool_event = original_event
        self.assertTrue(solved)
        self.assertEqual(starts[0][0], "edit_file")
        self.assertEqual(events[0][0], "edit_file")
        self.assertEqual(payloads[-1]["tool"], "edit_file")
        self.assertTrue(any(name == "edit_file" for name, _ in starts))
        self.assertFalse(any(name == "search_project" for name, _ in starts))
        self.assertTrue(validation["validation_signals"]["tests_passed"])
        self.assertEqual(history["kind"], "immediate_fix")

    def test_immediate_fix_records_repair_pattern_on_success(self):
        store = _StubLearningStore()

        class _LearningImmediateFixTools(_ImmediateFixTools):
            def __init__(self, memory_store):
                super().__init__()
                self.memory_store = memory_store

        tools = _LearningImmediateFixTools(store)
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        solved, _, validation, history = engine._maybe_apply_immediate_fix_from_observation(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="inspect failing tests"),
            workspace_validation={
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {
                            "nodeid": "demo_project/tests/test_main.py::test_add",
                            "message": "AssertionError: assert -1 == 5",
                            "summary": "FAILED demo_project/tests/test_main.py::test_add - AssertionError: assert -1 == 5",
                        }
                    ],
                }
            },
            current_tool_payloads=[
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/tests/test_main.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/tests/test_main.py",
                        "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    },
                },
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/utils.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
                    },
                },
            ],
            project_context={"test_runner": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
        )
        self.assertTrue(solved)
        self.assertTrue(validation["validation_signals"]["tests_passed"])
        self.assertEqual(history["kind"], "immediate_fix")
        self.assertEqual(len(store.repair_patterns), 1)
        self.assertEqual(store.repair_patterns[0]["pattern"], "operator mismatch")
        self.assertEqual(store.repair_patterns[0]["function_name"], "add")
        self.assertEqual(store.repair_patterns[0]["context"], "simple arithmetic function")
        self.assertIn("return a - b", store.repair_patterns[0]["before"])
        self.assertIn("return a + b", store.repair_patterns[0]["after"])

    def test_repair_loop_checks_repair_pattern_memory_before_debugging(self):
        store = _StubLearningStore()
        store.repair_pattern_matches = [
            {
                "pattern": "operator mismatch",
                "before": "return a - b",
                "after": "return a + b",
                "context": "simple arithmetic function",
                "confidence": 0.95,
                "score": 0.91,
            }
        ]

        class _LearningImmediateFixTools(_ImmediateFixTools):
            def __init__(self, memory_store):
                super().__init__()
                self.memory_store = memory_store

        tools = _LearningImmediateFixTools(store)
        engine = ChatEngine(model=_DummyModel(), tools=tools, system_prompt="test")
        engine._maybe_apply_immediate_fix_from_observation = lambda **kwargs: self.fail(
            "repair pattern memory should run before immediate debugging"
        )
        engine._maybe_apply_root_cause_fix = lambda **kwargs: self.fail(
            "repair pattern memory should resolve before root-cause templates"
        )

        latest_text, payloads, validation, history = engine._maybe_run_test_driven_repair(
            objective="Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            step=PlanStep(step_id=1, action="inspect failing tests"),
            workspace_validation={
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": False,
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {
                            "nodeid": "demo_project/tests/test_main.py::test_add",
                            "message": "AssertionError: assert -1 == 5",
                            "summary": "FAILED demo_project/tests/test_main.py::test_add - AssertionError: assert -1 == 5",
                        }
                    ],
                }
            },
            current_tool_payloads=[
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/tests/test_main.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/tests/test_main.py",
                        "content": "from demo_project.utils import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                    },
                },
                {
                    "tool": "read_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {
                        "ok": True,
                        "path": "demo_project/utils.py",
                        "content": "def add(a: int, b: int) -> int:\n    return a - b  # BUG\n",
                    },
                },
            ],
            project_context={"test_runner": "pytest"},
            auto_metrics={
                "tool_calls": 0,
                "failed_tool_calls": 0,
                "test_repair_attempts": 0,
                "est_tokens_out": 0,
            },
        )
        self.assertIn("learned repair pattern", latest_text or "")
        self.assertTrue(any(item.get("kind") == "repair_pattern" for item in history))
        self.assertTrue(validation["validation_signals"]["tests_passed"])
        self.assertTrue(any(item.get("tool") == "edit_file" for item in payloads))
        self.assertFalse(any(name == "search_project" for name, _ in tools.calls))
        self.assertFalse(any(name == "read_file" for name, _ in tools.calls))

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

    def test_repair_loop_caps_analysis_for_single_target_failure(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix auth")
        workspace_validation = {
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [
                    {
                        "nodeid": "tests/test_auth.py::test_login",
                        "summary": "FAILED tests/test_auth.py::test_login - AssertionError",
                    }
                ],
            }
        }
        engine.autonomous_test_repair_attempts = 3
        engine._maybe_apply_root_cause_fix = lambda **kwargs: (
            False,
            [],
            workspace_validation,
            None,
        )
        hypothesis_calls = {"n": 0}

        def _fake_hypothesis(**kwargs):
            hypothesis_calls["n"] += 1
            return {
                "hypothesis": "token not returned",
                "suspected_files": ["auth.py"],
                "confidence": 0.7,
            }

        engine._propose_test_failure_hypothesis = _fake_hypothesis
        engine.handle_turn_stream = lambda *args, **kwargs: "patched"
        engine._recent_tool_payloads = lambda limit=8: []
        engine._maybe_validate_workspace_after_step = lambda **kwargs: workspace_validation

        engine._maybe_run_test_driven_repair(
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

        self.assertEqual(hypothesis_calls["n"], 1)

    def test_root_cause_error_text_includes_failed_tool_errors(self):
        text = ChatEngine._extract_root_cause_error_text(
            {"validation_signals": {}},
            tool_payloads=[
                {
                    "tool": "execute_command",
                    "result": {"ok": False, "error": "ModuleNotFoundError: requests"},
                }
            ],
        )
        self.assertIn("ModuleNotFoundError: requests", text)

    def test_root_cause_fix_is_gated_by_match_score(self):
        store = _StubLearningStore(
            matches=[
                {
                    "id": "rc_low",
                    "pattern": "ModuleNotFoundError: ${module}",
                    "context": {"language": "python"},
                    "fix_template": [
                        {"tool": "execute_command", "args": {"cmd": "pip install requests"}}
                    ],
                    "confidence": 0.8,
                    "score": 0.6,
                }
            ]
        )
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(memory_store=store),
            system_prompt="test",
        )
        solved, payloads, _, history = engine._maybe_apply_root_cause_fix(
            objective="fix import failures",
            step=PlanStep(step_id=1, action="fix imports"),
            workspace_validation={
                "validation_signals": {
                    "failed_tests": 1,
                    "test_errors": 0,
                    "test_failures": [
                        {"summary": "FAILED tests/test_app.py - ModuleNotFoundError: requests"}
                    ],
                }
            },
            current_tool_payloads=[],
            project_context={"framework": "pytest"},
            auto_metrics={"tool_calls": 0, "failed_tool_calls": 0},
        )
        self.assertFalse(solved)
        self.assertEqual(payloads, [])
        self.assertEqual(history.get("skipped"), "low_match_score")

    def test_repair_loop_learns_root_cause_on_success(self):
        store = _StubLearningStore()
        engine = ChatEngine(
            model=_DummyModel(),
            tools=_DummyTools(memory_store=store),
            system_prompt="test",
        )
        engine.autonomous_test_repair_attempts = 1
        step = PlanStep(step_id=1, action="fix imports")
        workspace_validation = {
            "changed_files": ["app.py"],
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [
                    {
                        "summary": "FAILED tests/test_app.py::test_boot - ModuleNotFoundError: requests"
                    }
                ],
            },
        }
        engine._maybe_apply_root_cause_fix = lambda **kwargs: (False, [], workspace_validation, None)
        engine._propose_test_failure_hypothesis = lambda **kwargs: {
            "hypothesis": "missing dependency",
            "suspected_files": ["app.py"],
            "confidence": 0.8,
        }
        engine.handle_turn_stream = lambda *args, **kwargs: "patched"
        engine._recent_tool_payloads = lambda limit=8: [
            {
                "tool": "execute_command",
                "args": {"cmd": "pip install requests"},
                "result": {"ok": True},
            }
        ]
        engine._maybe_validate_workspace_after_step = lambda **kwargs: {
            "validation_signals": {
                "tests_passed": True,
                "failed_tests": 0,
                "test_errors": 0,
            }
        }

        engine._maybe_run_test_driven_repair(
            objective="fix import failures",
            step=step,
            workspace_validation=workspace_validation,
            current_tool_payloads=[],
            project_context={"framework": "pytest"},
            auto_metrics={
                "tool_calls": 0,
                "failed_tool_calls": 0,
                "test_repair_attempts": 0,
                "est_tokens_out": 0,
            },
        )
        self.assertEqual(len(store.upserts), 1)
        self.assertEqual(store.upserts[0]["pattern"], "ModuleNotFoundError: ${module}")

    def test_repair_loop_skips_duplicate_fix_signature(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(step_id=1, action="fix auth")
        workspace_validation = {
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [{"summary": "FAILED tests/test_auth.py::test_login - AssertionError"}],
            }
        }
        engine.autonomous_test_repair_attempts = 2
        engine._maybe_apply_root_cause_fix = lambda **kwargs: (False, [], workspace_validation, None)
        hypotheses = [
            {"hypothesis": "first fix", "suspected_files": ["auth.py"], "confidence": 0.7},
            {"hypothesis": "second fix", "suspected_files": ["auth.py"], "confidence": 0.6},
        ]
        engine._propose_test_failure_hypothesis = lambda **kwargs: hypotheses.pop(0)
        engine.handle_turn_stream = lambda *args, **kwargs: "patched"
        engine._recent_tool_payloads = lambda limit=8: [
            {"tool": "edit_file", "args": {"path": "auth.py"}, "result": {"ok": True}}
        ]
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
        self.assertTrue(any(item.get("skipped") == "duplicate_fix_signature" for item in history))

    def test_hybrid_planner_fills_missing_strategy_capabilities(self):
        engine = ChatEngine(
            model=_DummyModel(
                response='{"steps":[{"id":3,"tool":"run_tests","args":{"path":".","runner":"pytest"},"depends_on":[2],"expected":"tests pass"}]}'
            ),
            tools=_DummyTools(),
            system_prompt="test",
        )
        skeleton = [
            PlanStep(step_id=1, action="search_project", args={"query": "auth"}, depends_on=[]),
            PlanStep(step_id=2, action="edit_file", args={"path": "app.py"}, depends_on=[1]),
        ]
        engine._reuse_strategy_steps = lambda *args, **kwargs: skeleton
        steps = engine._plan_objective_steps("fix failing tests", step_cap=4)
        actions = [step.action for step in steps]
        self.assertIn("run_tests", actions)
        run_tests_step = next(step for step in steps if step.action == "run_tests")
        self.assertIn(2, run_tests_step.depends_on)

    def test_state_delta_classifies_improved_regressed_and_unchanged(self):
        improved = ChatEngine._state_delta_from_snapshots(
            before_snapshot={
                "failed_tests": 3,
                "test_errors": 0,
                "signature": "a|b|c",
                "failure_items": ["a", "b", "c"],
            },
            after_snapshot={
                "failed_tests": 1,
                "test_errors": 0,
                "signature": "a",
                "failure_items": ["a"],
            },
            workspace_validation={"changed_files": ["app.py"]},
        )
        self.assertEqual(improved["impact"], "improved")
        self.assertTrue(improved["made_progress"])
        self.assertEqual(improved["tests_fixed"], 2)

        regressed = ChatEngine._state_delta_from_snapshots(
            before_snapshot={
                "failed_tests": 1,
                "test_errors": 0,
                "signature": "a",
                "failure_items": ["a"],
            },
            after_snapshot={
                "failed_tests": 3,
                "test_errors": 0,
                "signature": "a|b|c",
                "failure_items": ["a", "b", "c"],
            },
            workspace_validation={"changed_files": ["app.py"]},
        )
        self.assertEqual(regressed["impact"], "regressed")
        self.assertFalse(regressed["made_progress"])
        self.assertEqual(regressed["new_errors"], 2)

        unchanged = ChatEngine._state_delta_from_snapshots(
            before_snapshot={
                "failed_tests": 1,
                "test_errors": 0,
                "signature": "a",
                "failure_items": ["a"],
            },
            after_snapshot={
                "failed_tests": 1,
                "test_errors": 0,
                "signature": "a",
                "failure_items": ["a"],
            },
            workspace_validation={"changed_files": []},
        )
        self.assertEqual(unchanged["impact"], "unchanged")
        self.assertFalse(unchanged["made_progress"])

    def test_resolve_action_retry_budget_escalates_and_replan_budget_stops(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        engine.autonomous_max_retries_per_step = 1
        engine.autonomous_max_replans = 1
        convergence_state = {"step_retry_counts": {}, "no_progress_streak": 0, "stop_reason": ""}
        auto_metrics = {"replans": 0, "tool_calls": 0, "test_repair_attempts": 0}
        step_fingerprint = "fix::{}"

        action_1, _ = engine._resolve_autonomous_action(
            proposed_action="retry",
            proposed_reason="try again",
            model_done=False,
            model_bored=False,
            validation_status="partial",
            validation_score=0.45,
            state_delta={"made_progress": False},
            step_fingerprint=step_fingerprint,
            convergence_state=convergence_state,
            auto_metrics=auto_metrics,
        )
        self.assertEqual(action_1, "retry")

        action_2, reason_2 = engine._resolve_autonomous_action(
            proposed_action="retry",
            proposed_reason="still failing",
            model_done=False,
            model_bored=False,
            validation_status="partial",
            validation_score=0.45,
            state_delta={"made_progress": False},
            step_fingerprint=step_fingerprint,
            convergence_state=convergence_state,
            auto_metrics=auto_metrics,
        )
        self.assertEqual(action_2, "replan")
        self.assertIn("retry budget exceeded", reason_2)

        auto_metrics["replans"] = 1
        action_3, reason_3 = engine._resolve_autonomous_action(
            proposed_action="retry",
            proposed_reason="still failing",
            model_done=False,
            model_bored=False,
            validation_status="partial",
            validation_score=0.45,
            state_delta={"made_progress": False},
            step_fingerprint=step_fingerprint,
            convergence_state=convergence_state,
            auto_metrics=auto_metrics,
        )
        self.assertEqual(action_3, "bored")
        self.assertIn("replan budget exceeded", reason_3)

    def test_resolve_action_stops_on_global_tool_call_budget(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        engine.autonomous_max_tool_calls = 3
        convergence_state = {"step_retry_counts": {}, "no_progress_streak": 0, "stop_reason": ""}
        action, reason = engine._resolve_autonomous_action(
            proposed_action="advance",
            proposed_reason="ok",
            model_done=False,
            model_bored=False,
            validation_status="success",
            validation_score=0.9,
            state_delta={"made_progress": True},
            step_fingerprint="a::{}",
            convergence_state=convergence_state,
            auto_metrics={"replans": 0, "tool_calls": 3, "test_repair_attempts": 0},
        )
        self.assertEqual(action, "bored")
        self.assertIn("tool-call budget reached", reason)

    def test_no_progress_streak_forces_replan_even_if_reflection_retries(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        engine.autonomous_max_no_progress_streak = 2
        convergence_state = {"step_retry_counts": {}, "no_progress_streak": 1, "stop_reason": ""}
        action, reason = engine._resolve_autonomous_action(
            proposed_action="retry",
            proposed_reason="reflection wants retry",
            model_done=False,
            model_bored=False,
            validation_status="partial",
            validation_score=0.45,
            state_delta={"made_progress": False},
            step_fingerprint="fix::{}",
            convergence_state=convergence_state,
            auto_metrics={"replans": 0, "tool_calls": 1, "test_repair_attempts": 0},
        )
        self.assertEqual(action, "replan")
        self.assertIn("no-progress streak", reason)

    def test_replan_preserves_completed_steps_by_fingerprint(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        previous_steps = [
            PlanStep(step_id=1, action="search_project", args={"query": "auth"}),
            PlanStep(step_id=2, action="edit_file", args={"path": "app.py"}),
            PlanStep(step_id=3, action="run_tests", args={"path": "."}),
        ]
        new_steps = [
            PlanStep(step_id=10, action="search_project", args={"query": "auth"}),
            PlanStep(step_id=11, action="edit_file", args={"path": "app.py"}),
            PlanStep(step_id=12, action="run_tests", args={"path": "."}),
        ]
        preserved = engine._preserve_completed_step_ids(
            previous_steps=previous_steps,
            previous_completed_ids={1, 2},
            new_steps=new_steps,
        )
        self.assertEqual(preserved, {10, 11})

    def test_non_edit_execute_command_does_not_require_validation_tests(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        step = PlanStep(
            step_id=1,
            action="run pytest to capture failing tests",
            args={"cmd": "pytest -q demo_project/tests"},
        )
        payloads = [
            {
                "tool": "execute_command",
                "args": {"cmd": "pytest -q demo_project/tests"},
                "result": {"ok": False, "exit_code": 1},
            }
        ]
        self.assertFalse(engine._should_run_validation_tests(step, payloads))

    def test_run_autonomous_forces_advance_for_non_edit_retry(self):
        engine = ChatEngine(model=_DummyModel(), tools=_AutoTools(), system_prompt="test")
        engine.auto_stop_enabled = False
        engine.autonomous_max_retries_per_step = 10
        engine.autonomous_max_replans = 10
        engine.autonomous_max_tool_calls = 0
        engine.autonomous_max_repair_attempts_total = 0
        engine.autonomous_max_no_progress_streak = 0

        call_count = {"n": 0}
        engine._create_task_state = lambda objective, step_cap=None: TaskState(
            goal=objective,
            steps=[
                PlanStep(
                    step_id=1,
                    action="run pytest to capture failing tests",
                    args={"cmd": "pytest -q demo_project/tests"},
                )
            ],
            current_step=0,
            completed_step_ids=set(),
            history=[],
        )

        def _fake_turn(*args, **kwargs):
            call_count["n"] += 1
            return "Captured failing tests."

        engine.handle_turn_stream = _fake_turn
        engine._recent_tool_payloads = lambda limit=8: [
            {
                "tool": "execute_command",
                "args": {"cmd": "pytest -q demo_project/tests"},
                "result": {"ok": False, "exit_code": 1},
            }
        ]
        engine._maybe_validate_workspace_after_step = lambda **kwargs: None
        engine._maybe_run_test_driven_repair = (
            lambda **kwargs: (
                None,
                kwargs.get("current_tool_payloads", []),
                kwargs.get("workspace_validation"),
                [],
            )
        )
        engine._validate_autonomous_step_execution = lambda **kwargs: {
            "status": "partial",
            "score": 0.45,
            "issues": [],
            "ok_tools": 1,
            "total_tools": 1,
            "signals": {"expects_workspace_change": False},
        }
        engine._reflect_autonomous_progress = lambda **kwargs: {
            "next_action": "replan",
            "reason": "tool-call loop limit reached",
            "confidence": 0.9,
            "issues": [],
            "new_steps": [],
        }

        engine.run_autonomous("capture failures", steps=2)
        # Should advance and finish after one step instead of retrying step 1.
        self.assertEqual(call_count["n"], 1)

    def test_run_autonomous_exits_before_validator_after_green_repair(self):
        engine = ChatEngine(model=_DummyModel(), tools=_AutoTools(), system_prompt="test")
        engine._create_task_state = lambda objective, step_cap=None: TaskState(
            goal=objective,
            steps=[
                PlanStep(
                    step_id=1,
                    action="run_tests",
                    args={"path": ".", "runner": "pytest", "args": "demo_project/tests"},
                )
            ],
            current_step=0,
            completed_step_ids=set(),
            history=[],
        )
        engine.handle_turn_stream = lambda *args, **kwargs: "Captured failing tests."
        engine._recent_tool_payloads = lambda limit=8: [
            {
                "tool": "run_tests",
                "args": {"path": ".", "runner": "pytest", "args": "demo_project/tests"},
                "result": {"ok": False, "exit_code": 1},
            }
        ]
        engine._maybe_validate_workspace_after_step = lambda **kwargs: None
        engine._maybe_run_test_driven_repair = lambda **kwargs: (
            "Applied immediate fix.",
            [
                {
                    "tool": "edit_file",
                    "args": {"path": "demo_project/utils.py"},
                    "result": {"ok": True},
                }
            ],
            {
                "ok": True,
                "tests_passed": True,
                "tests": {"command": "pytest -q demo_project/tests"},
                "validation_signals": {
                    "validation_completed": True,
                    "tests_passed": True,
                    "failed_tests": 0,
                    "test_errors": 0,
                    "has_diff": True,
                    "changed_file_count": 1,
                },
            },
            [{"kind": "immediate_fix"}],
        )
        engine._validate_autonomous_step_execution = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("validator should not run after green repair")
        )
        engine._reflect_autonomous_progress = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("reflection should not run after green repair")
        )

        engine.run_autonomous(
            "Fix failing tests in demo_project only. Run pytest -q demo_project/tests.",
            steps=1,
        )

    def test_run_autonomous_retry_consumes_finite_step_budget(self):
        engine = ChatEngine(model=_DummyModel(), tools=_AutoTools(), system_prompt="test")
        engine.auto_stop_enabled = False
        engine.autonomous_max_retries_per_step = 10
        engine.autonomous_max_replans = 10
        engine.autonomous_max_tool_calls = 0
        engine.autonomous_max_repair_attempts_total = 0
        engine.autonomous_max_no_progress_streak = 0

        call_count = {"n": 0}
        engine._create_task_state = lambda objective, step_cap=None: TaskState(
            goal=objective,
            steps=[PlanStep(step_id=1, action="implement fix", args={"path": "app.py"})],
            current_step=0,
            completed_step_ids=set(),
            history=[],
        )

        def _fake_turn(*args, **kwargs):
            call_count["n"] += 1
            return "Attempted fix."

        engine.handle_turn_stream = _fake_turn
        engine._recent_tool_payloads = lambda limit=8: [
            {"tool": "edit_file", "args": {"path": "app.py"}, "result": {"ok": True}}
        ]
        engine._maybe_validate_workspace_after_step = lambda **kwargs: {
            "ok": False,
            "changed_files": ["app.py"],
            "validation_signals": {
                "tests_passed": False,
                "failed_tests": 1,
                "test_errors": 0,
                "test_failures": [{"summary": "FAILED tests/test_app.py::test_auth"}],
                "has_diff": True,
                "changed_file_count": 1,
            },
        }
        engine._maybe_run_test_driven_repair = (
            lambda **kwargs: (
                None,
                kwargs.get("current_tool_payloads", []),
                kwargs.get("workspace_validation"),
                [],
            )
        )
        engine._validate_autonomous_step_execution = lambda **kwargs: {
            "status": "partial",
            "score": 0.45,
            "issues": [],
            "ok_tools": 1,
            "total_tools": 1,
        }
        engine._reflect_autonomous_progress = lambda **kwargs: {
            "next_action": "retry",
            "reason": "try again",
            "confidence": 0.9,
            "issues": [],
            "new_steps": [],
        }

        engine.run_autonomous("stabilize auth flow", steps=2)
        self.assertEqual(call_count["n"], 2)

    def test_tool_history_payload_is_compacted(self):
        large_stdout = "x" * 12000
        engine = ChatEngine(
            model=_DummyModel(
                response='{"tool":"execute_command","args":{"cmd":"pytest -q demo_project/tests"}}'
            ),
            tools=_DummyTools(),
            system_prompt="test",
            max_tool_rounds=1,
        )
        engine._run_tool_call_with_reflection = lambda user_message, call: {
            "name": "execute_command",
            "args": {"cmd": "pytest -q demo_project/tests"},
            "result": {
                "ok": False,
                "exit_code": 1,
                "stdout": large_stdout,
                "stderr": "",
                "command": "pytest -q demo_project/tests",
            },
            "reflection": {"status": "failed"},
            "initial_name": "execute_command",
            "initial_args": {"cmd": "pytest -q demo_project/tests"},
        }

        engine.handle_turn("Fix tests", enforce_presearch=False, log_interaction=False)

        tool_msgs = [m for m in engine.history if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        payload = json.loads(tool_msgs[0]["content"])
        self.assertIn("result", payload)
        stdout = payload["result"].get("stdout", "")
        self.assertLessEqual(len(stdout), engine.max_history_tool_blob_chars)
        self.assertEqual(payload["result"].get("exit_code"), 1)

    def test_intent_cache_is_bounded(self):
        engine = ChatEngine(model=_DummyModel(), tools=_DummyTools(), system_prompt="test")
        engine.max_intent_cache_size = 3
        engine._ai_intent_flags = lambda user_message: {}

        for i in range(10):
            engine._intent_flags(f"intent message {i}")

        self.assertLessEqual(len(engine._intent_cache), 3)
        self.assertNotIn("intent message 0", engine._intent_cache)


if __name__ == "__main__":
    unittest.main()
