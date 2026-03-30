from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import ActionRecord, ActionRequest
from .context import RuntimeContext
from .memory import MemoryStore
from .observations import Observation
from .tools import ToolRegistry


@dataclass(slots=True)
class ExecutionState:
    objective: str
    context: RuntimeContext
    max_steps: int
    mode: str
    available_tools: list[str]
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    pending: list[ActionRequest] = field(default_factory=list)
    repeated_failures: dict[str, int] = field(default_factory=dict)
    failed_strategies: dict[str, int] = field(default_factory=dict)
    seen_search_queries: set[str] = field(default_factory=set)
    seen_urls: set[str] = field(default_factory=set)
    knowledge_gap_detected: bool = False
    knowledge_gap_reason: str = ""
    external_knowledge_acquired: bool = False


class DefaultPlanner:
    def bootstrap(self, state: ExecutionState) -> None:
        metadata = state.context.metadata
        explicit_files = self._extract_explicit_paths(state.objective)
        state.knowledge_gap_detected, state.knowledge_gap_reason = self._detect_knowledge_gap(
            state=state,
            explicit_files=explicit_files,
        )
        state.notes.append(
            "Knowledge gap evaluation: "
            f"{'insufficient information' if state.knowledge_gap_detected else 'enough information to start'}. "
            f"{state.knowledge_gap_reason}"
        )

        if state.available_tools and "list_files" in state.available_tools:
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="list_files",
                    arguments={"path": metadata.get("scan_path", "."), "limit": 50},
                    rationale="Capture the local workspace shape before taking action.",
                ),
            )

        for path in explicit_files[:3]:
            if "read_file" in state.available_tools:
                self._enqueue_unique(
                    state,
                    ActionRequest(
                        tool="read_file",
                        arguments={"path": path},
                        rationale="Inspect explicit files mentioned in the objective.",
                    ),
                )

        if state.knowledge_gap_detected and "web_search" in state.available_tools:
            self._schedule_search(
                state=state,
                query=self._initial_search_query(state),
                rationale="Search first because the objective needs information that is not yet available locally.",
            )

        for command in list(metadata.get("commands", []) or [])[:3]:
            if "run_command" in state.available_tools:
                self._enqueue_unique(
                    state,
                    ActionRequest(
                        tool="run_command",
                        arguments={"command": command},
                        rationale="Run configured CLI diagnostics for the objective.",
                    ),
                )

        for request in list(metadata.get("api_requests", []) or [])[:3]:
            if "http_request" in state.available_tools:
                self._enqueue_unique(
                    state,
                    ActionRequest(
                        tool="http_request",
                        arguments=dict(request),
                        rationale="Collect external structured data required by the workflow.",
                    ),
                )

        output_path = metadata.get("output_path")
        if output_path and "write_file" in state.available_tools:
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="write_file",
                    arguments={"path": output_path, "content": ""},
                    rationale="Prepare an output artifact for the final summary.",
                ),
            )

    def next_action(self, state: ExecutionState) -> ActionRequest | None:
        while state.pending:
            action = state.pending.pop(0)
            if action.tool in state.available_tools:
                return action
        return None

    def observe(self, state: ExecutionState, record: ActionRecord) -> None:
        observation = record.observation
        if record.status == "failed":
            error_key = f"{record.tool}:{json.dumps(observation.content, sort_keys=True, default=str)}"
            state.repeated_failures[error_key] = state.repeated_failures.get(error_key, 0) + 1
            self._register_failed_strategy(state=state, record=record, error_text=str(observation.content))
            state.reflections.append(
                f"Reflection: {record.tool} failed, so the agent should switch strategy rather than repeat the same action blindly."
            )
            return

        if observation.type == "filesystem_snapshot":
            count = int(observation.metadata.get("count", 0) or 0)
            state.notes.append(f"Indexed workspace files: {count}")
        elif observation.type == "command_result":
            content = observation.content
            if isinstance(content, dict) and int(content.get("exit_code", 1)) != 0:
                stderr = str(content.get("stderr", "")).strip()
                stdout = str(content.get("stdout", "")).strip()
                error_text = stderr or stdout or f"Command failed: {content.get('command', '')}"
                state.notes.append(error_text)
                self._register_failed_strategy(state=state, record=record, error_text=error_text)
            else:
                state.reflections.append(
                    f"Reflection: {record.tool} succeeded, so the current plan has concrete execution evidence."
                )
        elif observation.type == "api_response":
            state.notes.append("Captured API response for synthesis.")
            state.reflections.append(
                "Reflection: API data was gathered successfully and can now inform the next action."
            )
        elif observation.type == "search_results":
            count = int(observation.metadata.get("count", 0) or 0)
            state.notes.append(f"Captured {count} search results for synthesis.")
            self._schedule_web_reads_from_search(state=state, observation=observation)
            state.reflections.append(
                "Reflection: search results reduced uncertainty, so the next step should read the most relevant sources."
            )
        elif observation.type == "webpage_read":
            state.external_knowledge_acquired = True
            title = str(observation.metadata.get("title", "")).strip()
            state.notes.append(f"Read webpage: {title or observation.metadata.get('url', '')}")
            state.reflections.append(
                "Reflection: external knowledge has been collected and should shape the next action."
            )
        elif observation.type == "file_read":
            path = observation.metadata.get("path", "")
            state.notes.append(f"Inspected file: {path}")
            state.reflections.append(
                "Reflection: local context is now clearer, which lowers the need for guessing."
            )
        elif observation.type == "file_write":
            path = observation.metadata.get("path", "")
            state.notes.append(f"Wrote artifact: {path}")
            state.reflections.append(
                "Reflection: the runtime produced an output artifact and should now evaluate whether the objective is satisfied."
            )

    def finalize(self, state: ExecutionState) -> tuple[bool, Any, float]:
        summary = {
            "objective": state.objective,
            "working_dir": str(state.context.working_dir),
            "evidence": [record.observation.to_dict() for record in state.actions],
            "notes": state.notes,
            "reflections": state.reflections,
            "memory_hits": state.memory_hits,
            "knowledge_gap": {
                "detected": state.knowledge_gap_detected,
                "reason": state.knowledge_gap_reason,
            },
        }
        output_path = state.context.metadata.get("output_path")
        if output_path:
            summary_text = self._render_summary(state)
            summary["summary"] = summary_text
            self._write_summary_artifact(state=state, output_path=str(output_path), summary_text=summary_text)
        success = self._is_success(state)
        confidence = self._confidence(state, success)
        return success, summary, confidence

    @staticmethod
    def _extract_explicit_paths(objective: str) -> list[str]:
        return re.findall(r"(?:[\w.-]+/)+[\w.-]+|[\w.-]+\.[A-Za-z0-9]{1,6}", objective)

    def _is_success(self, state: ExecutionState) -> bool:
        successful_observations = [
            action.observation
            for action in state.actions
            if action.status == "completed"
        ]
        if not successful_observations:
            return False
        for observation in successful_observations:
            if observation.type == "command_result":
                if int(observation.content.get("exit_code", 1)) == 0:
                    return True
            elif observation.type in {"api_response", "search_results", "file_read", "file_write", "filesystem_snapshot"}:
                return True
        return False

    def _confidence(self, state: ExecutionState, success: bool) -> float:
        if not state.actions:
            return 0.0
        evidence_score = min(0.9, 0.25 + (0.15 * len([a for a in state.actions if a.status == "completed"])))
        diversity_bonus = 0.05 * len({action.observation.type for action in state.actions})
        failure_penalty = 0.1 * len([a for a in state.actions if a.status == "failed"])
        base = evidence_score + diversity_bonus - failure_penalty
        if success:
            base += 0.1
        return round(max(0.0, min(base, 1.0)), 3)

    def _render_summary(self, state: ExecutionState) -> str:
        lines = [f"# Starforge Summary", "", f"Objective: {state.objective}", ""]
        lines.append(
            "Knowledge gap: "
            f"{'detected' if state.knowledge_gap_detected else 'not detected'}"
            + (f" ({state.knowledge_gap_reason})" if state.knowledge_gap_reason else "")
        )
        lines.append("")
        if state.memory_hits:
            lines.append("Relevant memory patterns:")
            for item in state.memory_hits:
                lines.append(
                    f"- {item.get('pattern_type')}: {item.get('resolution_strategy')} (score={item.get('score', 0)})"
                )
            lines.append("")
        lines.append("Observed actions:")
        for action in state.actions:
            lines.append(f"- {action.tool}: {action.observation.type}")
        if state.notes:
            lines.append("")
            lines.append("Notes:")
            for note in state.notes:
                lines.append(f"- {note}")
        if state.reflections:
            lines.append("")
            lines.append("Reflections:")
            for reflection in state.reflections:
                lines.append(f"- {reflection}")
        return "\n".join(lines).strip() + "\n"

    def _detect_knowledge_gap(
        self,
        *,
        state: ExecutionState,
        explicit_files: list[str],
    ) -> tuple[bool, str]:
        objective = state.objective.lower()
        metadata = state.context.metadata
        has_local_context = bool(
            explicit_files
            or metadata.get("commands")
            or metadata.get("api_requests")
            or metadata.get("seed_knowledge")
        )
        external_markers = (
            "research",
            "analyze",
            "analysis",
            "trend",
            "market",
            "latest",
            "current",
            "recent",
            "api",
            "documentation",
            "docs",
            "how to",
            "find",
            "compare",
            "summarize",
        )
        if any(marker in objective for marker in external_markers) and "web_search" in state.available_tools:
            return True, "The objective depends on external knowledge or current information."
        if not has_local_context and "web_search" in state.available_tools:
            return True, "No explicit local context or API inputs were provided, so external research is safer than guessing."
        return False, "Local context and configured tools are sufficient for an initial action."

    def _initial_search_query(self, state: ExecutionState) -> str:
        configured = list(state.context.metadata.get("search_queries", []) or [])
        if configured:
            return str(configured[0])
        return state.objective

    def _enqueue_unique(
        self,
        state: ExecutionState,
        action: ActionRequest,
        *,
        front: bool = False,
    ) -> None:
        action_key = (action.tool, json.dumps(action.arguments, sort_keys=True, default=str))
        for queued in state.pending:
            queued_key = (queued.tool, json.dumps(queued.arguments, sort_keys=True, default=str))
            if queued_key == action_key:
                return
        if front:
            state.pending.insert(0, action)
        else:
            state.pending.append(action)

    def _schedule_search(self, *, state: ExecutionState, query: str, rationale: str) -> None:
        normalized_query = str(query or "").strip()
        if not normalized_query or "web_search" not in state.available_tools:
            return
        if normalized_query in state.seen_search_queries:
            return
        state.seen_search_queries.add(normalized_query)
        self._enqueue_unique(
            state,
            ActionRequest(
                tool="web_search",
                arguments={"query": normalized_query, "limit": 5},
                rationale=rationale,
            ),
            front=True,
        )

    def _schedule_web_reads_from_search(self, *, state: ExecutionState, observation: Observation) -> None:
        if "read_webpage" not in state.available_tools:
            return
        content = observation.content if isinstance(observation.content, list) else []
        for item in content[:2]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url or url in state.seen_urls:
                continue
            state.seen_urls.add(url)
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="read_webpage",
                    arguments={"url": url},
                    rationale="Read the referenced source so the agent can reason from evidence instead of snippets alone.",
                ),
                front=False,
            )

    def _register_failed_strategy(
        self,
        *,
        state: ExecutionState,
        record: ActionRecord,
        error_text: str,
    ) -> None:
        strategy = record.tool
        state.failed_strategies[strategy] = state.failed_strategies.get(strategy, 0) + 1
        attempts = state.failed_strategies[strategy]
        state.reflections.append(
            f"Reflection: strategy '{strategy}' has failed {attempts} time(s); the agent should reduce blind repetition."
        )
        if attempts < 2:
            return
        fallback_query = self._search_query_from_failure(state=state, record=record, error_text=error_text)
        self._schedule_search(
            state=state,
            query=fallback_query,
            rationale="The same strategy failed twice, so the agent should research an alternative approach before trying again.",
        )

    def _search_query_from_failure(
        self,
        *,
        state: ExecutionState,
        record: ActionRecord,
        error_text: str,
    ) -> str:
        compact_error = re.sub(r"\s+", " ", str(error_text)).strip()
        compact_error = compact_error[:160]
        return f"{state.objective} {record.tool} {compact_error}".strip()

    def _write_summary_artifact(self, state: ExecutionState, output_path: str, summary_text: str) -> None:
        target = Path(output_path)
        if not target.is_absolute():
            target = (state.context.working_dir / output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(summary_text, encoding="utf-8")

        for record in reversed(state.actions):
            if record.tool != "write_file":
                continue
            record.arguments["content"] = summary_text
            record.observation = Observation(
                type="file_write",
                content={
                    "path": str(target),
                    "bytes": len(summary_text.encode("utf-8")),
                },
                metadata={
                    "path": str(target),
                    "append": False,
                },
            )
            return


class StarforgeRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        planner: DefaultPlanner | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or DefaultPlanner()
        self.memory_store = memory_store or MemoryStore()

    def run(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = dict(config or {})
        runtime_context = RuntimeContext.from_payload(objective=objective, context=context)
        max_steps = max(1, int(config.get("max_steps", 8)))
        mode = str(config.get("mode", "autonomous"))
        memory_enabled = bool(config.get("memory_enabled", True))
        memory_hits = self.memory_store.search(objective, limit=3) if memory_enabled else []
        state = ExecutionState(
            objective=objective,
            context=runtime_context,
            max_steps=max_steps,
            mode=mode,
            available_tools=self.registry.names(),
            memory_hits=memory_hits,
        )
        self.planner.bootstrap(state)

        for _ in range(max_steps):
            action = self.planner.next_action(state)
            if action is None:
                break
            try:
                observation = self.registry.execute(action.tool, action.arguments, runtime_context)
                record = ActionRecord(
                    tool=action.tool,
                    arguments=dict(action.arguments),
                    observation=observation,
                    rationale=action.rationale,
                )
            except Exception as exc:
                record = ActionRecord(
                    tool=action.tool,
                    arguments=dict(action.arguments),
                    observation=Observation(
                        type="tool_error",
                        content={"error": str(exc)},
                        metadata={"tool": action.tool},
                    ),
                    status="failed",
                    rationale=action.rationale,
                )
            state.actions.append(record)
            self.planner.observe(state, record)

        success, result, confidence = self.planner.finalize(state)
        self._learn_from_run(state=state, success=success, confidence=confidence)
        return {
            "success": success,
            "steps": len(state.actions),
            "actions": [action.to_dict() for action in state.actions],
            "result": result,
            "confidence": confidence,
        }

    def _learn_from_run(self, state: ExecutionState, success: bool, confidence: float) -> None:
        if success and state.actions:
            strategy = " -> ".join(action.tool for action in state.actions)
            self.memory_store.remember(
                pattern_type="successful_sequence",
                context=state.objective,
                resolution_strategy=strategy,
                confidence=confidence,
                metadata={"mode": state.mode},
            )
        for key, count in state.repeated_failures.items():
            if count >= 2:
                self.memory_store.remember(
                    pattern_type="repeated_failure",
                    context=key,
                    resolution_strategy="Avoid repeating the same failing action without changing inputs or switching tools.",
                    confidence=0.6,
                    metadata={"objective": state.objective},
                )
