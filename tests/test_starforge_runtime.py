from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from starforge import create_runtime, run
from starforge.adapters import CodeProjectAdapter
from starforge.context import RuntimeContext
from starforge.memory import MemoryStore
from starforge.tools import ListFilesTool


class TestStarforgeMemory:
    def test_memory_store_uses_portable_root_and_similarity_search(self, tmp_path: Path) -> None:
        root = tmp_path / ".starforge"
        store = MemoryStore(root=root)
        store.remember(
            pattern_type="repeated_failure",
            context="pytest command failed with import error",
            resolution_strategy="Inspect module path before retrying the same command.",
            confidence=0.8,
        )

        assert store.patterns_path == root / "patterns.jsonl"
        matches = store.search("import error while running tests", limit=1)
        assert matches
        assert matches[0]["pattern_type"] == "repeated_failure"


class TestStarforgeRuntime:
    @patch("starforge.tools.builtin.requests.get")
    def test_web_search_is_available_as_an_optional_exploration_step(
        self,
        mock_get: Mock,
        tmp_path: Path,
    ) -> None:
        mock_get.return_value = Mock(
            json=Mock(
                return_value={
                    "Heading": "Example API",
                    "AbstractText": "Example API docs.",
                    "AbstractURL": "https://example.com/api",
                    "RelatedTopics": [],
                }
            )
        )

        result = run(
            objective="find how to use the Example API and summarize the required steps",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 3,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert "web_search" in tools
        assert "knowledge_gap" not in result["result"]

    def test_local_objective_does_not_force_web_search(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 2,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert tools == ["list_files", "read_file"]

    def test_run_returns_generic_result_shape_for_cli_workflow(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt and run a quick diagnostic",
            context={
                "working_dir": str(tmp_path),
                "commands": ["python -c \"print('ok')\""],
            },
            config={
                "adapter": "cli",
                "max_steps": 4,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert set(result) == {"success", "steps", "actions", "result", "confidence"}
        assert result["success"] is True
        assert result["steps"] >= 2
        assert any(action["observation"]["type"] == "command_result" for action in result["actions"])
        assert 0.0 <= result["confidence"] <= 1.0
        human_summary = result["result"].get("human_readable", "")
        assert isinstance(human_summary, str)
        assert "objective" in human_summary.lower()
        assert "\n\n" in human_summary

    def test_open_ended_max_steps_zero_runs_until_pending_actions_finish(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        target.write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt and run a quick diagnostic",
            context={
                "working_dir": str(tmp_path),
                "commands": ["python -c \"print('ok')\""],
            },
            config={
                "adapter": "cli",
                "max_steps": 0,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert "list_files" in tools
        assert "read_file" in tools
        assert "run_command" in tools
        assert result["steps"] >= 3

    def test_snapshot_followup_resolves_filename_to_workspace_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspaces" / "CRYPTO"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "agents.md").write_text("Notes only.\n", encoding="utf-8")
        (workspace / "getprice.py").write_text("print('price')\n", encoding="utf-8")

        result = run(
            objective=(
                "In workspaces. Check out the directory CRYPTO. "
                "Read the agents.md and getprice.py."
            ),
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        read_paths = [
            action["arguments"].get("path", "")
            for action in result["actions"]
            if action["tool"] == "read_file"
        ]
        assert "workspaces/CRYPTO/agents.md" in read_paths
        assert "workspaces/CRYPTO/getprice.py" in read_paths
        failed_reads = [
            action
            for action in result["actions"]
            if action["tool"] == "read_file" and action.get("status") == "failed"
        ]
        assert not failed_reads

    def test_hinted_directory_scan_finds_files_in_large_workspace_tree(self, tmp_path: Path) -> None:
        noisy_root = tmp_path / "memory" / "blocks"
        noisy_root.mkdir(parents=True, exist_ok=True)
        for index in range(500):
            subtree = noisy_root / f"segment_{index:03d}"
            subtree.mkdir(parents=True, exist_ok=True)
            (subtree / "note.md").write_text("noise\n", encoding="utf-8")

        workspace = tmp_path / "workspaces" / "CRYPTO"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "agents.md").write_text("Objective: summarize local files\n", encoding="utf-8")
        (workspace / "getprice.py").write_text("print('price')\n", encoding="utf-8")

        result = run(
            objective=(
                "In workspaces. Check out the directory CRYPTO. "
                "Read the agents.md and getprice.py."
            ),
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "memory_root": str(tmp_path / "memory_store"),
            },
        )

        read_paths = [
            action["arguments"].get("path", "")
            for action in result["actions"]
            if action["tool"] == "read_file"
        ]
        assert "workspaces/CRYPTO/agents.md" in read_paths
        assert "workspaces/CRYPTO/getprice.py" in read_paths

    def test_list_files_skips_hidden_git_entries_by_default(self, tmp_path: Path) -> None:
        git_root = tmp_path / ".git" / "objects"
        git_root.mkdir(parents=True, exist_ok=True)
        for index in range(30):
            (git_root / f"{index:02x}.obj").write_text("x\n", encoding="utf-8")
        target = tmp_path / "workspaces" / "CRYPTO" / "agents.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# demo\n", encoding="utf-8")

        runtime_context = RuntimeContext.from_payload(
            objective="list files",
            context={"working_dir": str(tmp_path)},
        )
        observation = ListFilesTool().run({"path": ".", "limit": 10}, runtime_context)
        content = [str(item) for item in observation.content]

        assert "workspaces/CRYPTO/agents.md" in content
        assert all(not entry.startswith(".git/") for entry in content)

    def test_list_files_breadth_first_reaches_workspace_before_deep_memory_tree(self, tmp_path: Path) -> None:
        memory_blocks = tmp_path / "memory" / "blocks" / "heavy"
        memory_blocks.mkdir(parents=True, exist_ok=True)
        for index in range(400):
            (memory_blocks / f"item_{index:03d}.md").write_text("x\n", encoding="utf-8")

        workspace = tmp_path / "workspaces" / "CRYPTO"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "agents.md").write_text("# objective\n", encoding="utf-8")

        runtime_context = RuntimeContext.from_payload(
            objective="list files",
            context={"working_dir": str(tmp_path)},
        )
        observation = ListFilesTool().run({"path": ".", "limit": 120}, runtime_context)
        content = [str(item) for item in observation.content]
        assert "workspaces/CRYPTO/agents.md" in content

    def test_code_adapter_promotes_code_commands_without_core_logic(self) -> None:
        adapter = CodeProjectAdapter()
        merged = adapter.merge_context(
            {
                "test_command": "pytest -q",
                "build_command": "python -m compileall .",
            }
        )
        assert merged["commands"] == ["pytest -q", "python -m compileall ."]

    @patch("starforge.tools.builtin.requests.get")
    @patch("starforge.tools.builtin.requests.request")
    def test_api_research_workflow_normalizes_api_and_search_observations(
        self,
        mock_request: Mock,
        mock_get: Mock,
        tmp_path: Path,
    ) -> None:
        mock_request.return_value = Mock(
            headers={"content-type": "application/json"},
            url="https://api.example.test/btc",
            status_code=200,
            ok=True,
            json=Mock(return_value={"prices": [[1, 100.0], [2, 105.0]]}),
        )

        def fake_get(url: str, *args, **kwargs) -> Mock:
            if "api.duckduckgo.com" in url:
                return Mock(
                    json=Mock(
                        return_value={
                            "Heading": "Bitcoin",
                            "AbstractText": "Bitcoin is a cryptocurrency.",
                            "AbstractURL": "https://example.com/bitcoin",
                            "RelatedTopics": [
                                {"Text": "BTC trend overview", "FirstURL": "https://example.com/overview"},
                                {"Text": "BTC momentum signals", "FirstURL": "https://example.com/momentum"},
                                {"Text": "BTC market structure", "FirstURL": "https://example.com/structure"},
                                {"Text": "BTC macro factors", "FirstURL": "https://example.com/macro"},
                            ],
                        }
                    )
                )
            return Mock(
                text="<html><head><title>BTC Overview</title></head><body><p>Momentum remains positive.</p></body></html>",
                url=url,
                status_code=200,
            )

        mock_get.side_effect = fake_get

        result = run(
            objective="analyze BTC trend and summarize key signals",
            context={
                "working_dir": str(tmp_path),
                "api_requests": [{"url": "https://api.example.test/btc"}],
                "output_path": "btc_summary.md",
            },
            config={
                "adapter": "api",
                "max_steps": 4,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert result["success"] is True
        assert any(action["observation"]["type"] == "api_response" for action in result["actions"])
        assert any(action["observation"]["type"] == "search_results" for action in result["actions"])
        assert any(action["observation"]["type"] == "webpage_read" for action in result["actions"])
        summary_path = tmp_path / "btc_summary.md"
        assert summary_path.exists()
        assert "Starforge Summary" in summary_path.read_text(encoding="utf-8")

    @patch("starforge.tools.builtin.requests.get")
    def test_repeated_failure_switches_to_search_instead_of_blind_retry(
        self,
        mock_get: Mock,
        tmp_path: Path,
    ) -> None:
        mock_get.return_value = Mock(
            json=Mock(
                return_value={
                    "Heading": "Shell troubleshooting",
                    "AbstractText": "Use research when commands keep failing.",
                    "AbstractURL": "https://example.com/shell",
                    "RelatedTopics": [],
                }
            )
        )

        result = run(
            objective="run the broken command and figure out a better approach",
            context={
                "working_dir": str(tmp_path),
                "commands": [
                    "python -c \"import sys; sys.exit(1)\"",
                    "python -c \"raise SystemExit(1)\"",
                ],
            },
            config={
                "adapter": "cli",
                "max_steps": 4,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert tools.count("run_command") == 2
        assert "web_search" in tools
        assert any("failed 2 time(s)" in reflection for reflection in result["result"]["reflections"])

    def test_runtime_can_be_embedded_as_pluggable_component(self, tmp_path: Path) -> None:
        runtime = create_runtime(adapter="cli", memory_root=str(tmp_path / "memory"))

        result = runtime.run(
            objective="run a smoke command",
            context={
                "working_dir": str(tmp_path),
                "commands": ["python -c \"print('embedded')\""],
            },
            config={"max_steps": 2},
        )

        assert result["success"] is True
        assert result["actions"]

    def test_runtime_stream_callback_receives_action_events(self, tmp_path: Path) -> None:
        events = []
        runtime = create_runtime(adapter="cli", memory_root=str(tmp_path / "memory"))
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = runtime.run(
            objective="inspect hello.txt and run a quick diagnostic",
            context={
                "working_dir": str(tmp_path),
                "commands": ["python -c \"print('ok')\""],
            },
            config={
                "max_steps": 3,
                "on_action": lambda event: events.append(dict(event)),
            },
        )

        assert result["actions"]
        assert len(events) == len(result["actions"])
        assert events[0]["index"] == 1
        assert events[0]["tool"] == result["actions"][0]["tool"]

    def test_autonomous_replan_executes_read_python_script_when_objective_requires_execution(
        self,
        tmp_path: Path,
    ) -> None:
        script = tmp_path / "getprice.py"
        script.write_text("print('price-check')\n", encoding="utf-8")

        result = run(
            objective="Read getprice.py and do the objective declared in the instruction.",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert "read_file" in tools
        assert "run_command" in tools
        command_actions = [action for action in result["actions"] if action["tool"] == "run_command"]
        assert any("python getprice.py" in action["arguments"].get("command", "") for action in command_actions)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_feedback_can_suggest_next_tool_when_queue_is_empty(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _StubModel:
            provider = "ollama"

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                return '{"tool":"web_search","args":{"query":"getprice.py usage","limit":5}}'

        mock_build_model.return_value = _StubModel()

        script = tmp_path / "getprice.py"
        script.write_text("print('price-check')\n", encoding="utf-8")

        result = run(
            objective="Read getprice.py and inspect it.",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert any(action["tool"] == "web_search" for action in result["actions"])

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_agents_objective_prefers_model_generated_search_query(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _StubModel:
            provider = "ollama"

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                return '{"tool":"web_search","args":{"query":"btc april may 2026 trading plan risk levels","limit":5}}'

        mock_build_model.return_value = _StubModel()
        (tmp_path / "agents.md").write_text(
            "Objective: generate a high-conviction trading plan for the period April–May 2026\n",
            encoding="utf-8",
        )

        result = run(
            objective="Read agents.md and continue autonomously.",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 4,
                "mode": "autonomous",
                "model_feedback": True,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        search_actions = [action for action in result["actions"] if action["tool"] == "web_search"]
        assert search_actions
        assert search_actions[0]["arguments"].get("query") == "btc april may 2026 trading plan risk levels"

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_feedback_can_return_subplan_calls_and_runtime_executes_them(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _StubModel:
            provider = "ollama"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                self.calls += 1
                if self.calls == 1:
                    return (
                        '{"calls":['
                        '{"tool":"write_file","args":{"path":"archive_prices.py","content":"print(\\"BTC,ETH\\")\\n"}},'
                        '{"tool":"run_command","args":{"command":"python archive_prices.py"}}'
                        ']}'
                    )
                return '{"done":true,"final_answer":"subplan executed"}'

        mock_build_model.return_value = _StubModel()
        (tmp_path / "getprice.py").write_text("print('BTC')\n", encoding="utf-8")

        result = run(
            objective="Read getprice.py, then adapt and implement a better multi-asset output flow.",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert "write_file" in tools
        assert "run_command" in tools
        assert result["result"]["model_final_answer"] == "subplan executed"

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_feedback_unavailable_note_is_recorded(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _FallbackLike:
            provider = "unknown"

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                return "no backend"

        mock_build_model.return_value = _FallbackLike()

        script = tmp_path / "hello.txt"
        script.write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 2,
                "mode": "autonomous",
                "model_feedback": True,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        notes = result["result"].get("notes", [])
        assert any("local fallback replanner" in str(note).lower() for note in notes)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_explicit_model_request_without_backend_keeps_heuristic_execution(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _FallbackLike:
            provider = "unknown"

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                return "no backend"

        mock_build_model.return_value = _FallbackLike()
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 5,
                "mode": "autonomous",
                "model_feedback": True,
                "model_name": "gemini-3.1-pro-preview",
                "feedback_provider": "google",
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert result["actions"]
        assert result["success"] is True
        notes = result["result"].get("notes", [])
        assert any("local fallback replanner" in str(note).lower() for note in notes)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_orchestrated_mode_starts_from_model_actions_and_requires_done_token(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _StubModel:
            provider = "google"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                self.calls += 1
                if self.calls == 1:
                    return '{"tool":"web_search","args":{"query":"hello file validation","limit":5}}'
                if self.calls == 2:
                    return '{"tool":"read_file","args":{"path":"hello.txt"}}'
                if self.calls == 3:
                    return '{"done":true,"final_answer":"draft complete DONE_STOP_AUTONOMOUS"}'
                return '{"done":true,"final_answer":"draft complete DONE_STOP_AUTONOMOUS"}'

        mock_build_model.return_value = _StubModel()
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt and verify completion",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "model_orchestrated": True,
                "model_name": "gemini-3.1-pro-preview",
                "feedback_provider": "google",
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert tools[0] == "web_search"
        assert "web_search" in tools
        assert "read_file" in tools
        assert "DONE_STOP_AUTONOMOUS" in result["result"]["model_final_answer"]
        notes = result["result"].get("notes", [])
        assert any("model-orchestrated execution is enabled" in str(note).lower() for note in notes)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_orchestrated_keeps_poking_until_done_token_or_poke_budget(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _StubModel:
            provider = "google"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                self.calls += 1
                if self.calls == 1:
                    return '{"tool":"read_file","args":{"path":"hello.txt"}}'
                if self.calls < 4:
                    return '{"done":true,"final_answer":"draft complete"}'
                return '{"done":true,"final_answer":"draft complete DONE_STOP_AUTONOMOUS"}'

        mock_build_model.return_value = _StubModel()
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt and verify completion",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "model_orchestrated": True,
                "model_name": "gemini-3.1-pro-preview",
                "feedback_provider": "google",
                "max_done_token_pokes": 5,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert result["success"] is True
        assert "DONE_STOP_AUTONOMOUS" in result["result"]["model_final_answer"]
        notes = result["result"].get("notes", [])
        assert any("without required token" in str(note).lower() for note in notes)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    def test_model_orchestrated_without_backend_stops_with_explicit_note(
        self,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        class _FallbackLike:
            provider = "unknown"

            def generate(self, messages):  # noqa: ANN001 - test double
                del messages
                return "no backend"

        mock_build_model.return_value = _FallbackLike()
        (tmp_path / "hello.txt").write_text("hello\n", encoding="utf-8")

        result = run(
            objective="inspect hello.txt and keep running",
            context={"working_dir": str(tmp_path)},
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "model_orchestrated": True,
                "model_name": "gemini-3.1-pro-preview",
                "feedback_provider": "google",
                "memory_root": str(tmp_path / "memory"),
            },
        )

        assert result["success"] is False
        assert result["actions"] == []
        notes = result["result"].get("notes", [])
        assert any("model-orchestrated mode requested but no external model backend is available" in str(note).lower() for note in notes)

    @patch("starforge.engine._ASSISTANT_BUILD_MODEL")
    @patch("starforge.tools.builtin.requests.get")
    def test_local_fallback_replanner_uses_command_output_for_followup_search(
        self,
        mock_get: Mock,
        mock_build_model: Mock,
        tmp_path: Path,
    ) -> None:
        mock_build_model.return_value = None
        mock_get.return_value = Mock(
            json=Mock(
                return_value={
                    "Heading": "price-check",
                    "AbstractText": "price-check reference",
                    "AbstractURL": "https://example.com/price-check",
                    "RelatedTopics": [],
                }
            )
        )

        result = run(
            objective="run diagnostics and continue autonomously",
            context={
                "working_dir": str(tmp_path),
                "commands": ["python -c \"print('price-check')\""],
            },
            config={
                "adapter": "cli",
                "max_steps": 0,
                "mode": "autonomous",
                "model_feedback": True,
                "memory_root": str(tmp_path / "memory"),
            },
        )

        tools = [action["tool"] for action in result["actions"]]
        assert "run_command" in tools
        assert "web_search" in tools
