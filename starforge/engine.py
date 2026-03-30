from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import ActionRecord, ActionRequest
from .context import RuntimeContext
from .memory import MemoryStore
from .observations import Observation
from .tools import ToolRegistry

try:  # Optional model-feedback integration from the assistant stack.
    from assistant.model import build_model as _ASSISTANT_BUILD_MODEL
    from assistant.tool_calls import parse_tool_calls as _ASSISTANT_PARSE_TOOL_CALLS
    from assistant.utils import parse_json_payload as _ASSISTANT_PARSE_JSON_PAYLOAD
except Exception:  # pragma: no cover - optional dependency
    _ASSISTANT_BUILD_MODEL = None
    _ASSISTANT_PARSE_TOOL_CALLS = None
    _ASSISTANT_PARSE_JSON_PAYLOAD = None


@dataclass(slots=True)
class ExecutionState:
    objective: str
    context: RuntimeContext
    max_steps: int
    mode: str
    available_tools: list[str]
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    plan_examples: list[dict[str, Any]] = field(default_factory=list)
    actions: list[ActionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    pending: list[ActionRequest] = field(default_factory=list)
    repeated_failures: dict[str, int] = field(default_factory=dict)
    failed_strategies: dict[str, int] = field(default_factory=dict)
    seen_search_queries: set[str] = field(default_factory=set)
    seen_urls: set[str] = field(default_factory=set)
    external_knowledge_acquired: bool = False
    autonomous_replan_count: int = 0
    no_progress_streak: int = 0
    model_marked_complete: bool = False
    model_final_answer: str = ""
    model_orchestrated: bool = False
    model_feedback_available: bool = False
    require_done_stop_token: bool = False
    done_stop_token: str = "DONE_STOP_AUTONOMOUS"
    done_stop_seen: bool = False
    done_token_poke_count: int = 0
    model_audit: dict[str, Any] = field(default_factory=dict)
    model_audit_count: int = 0


class DefaultPlanner:
    def bootstrap(self, state: ExecutionState) -> None:
        metadata = state.context.metadata
        explicit_files = self._extract_explicit_paths(state.objective)
        targeted_scan_paths = self._candidate_scan_paths(state)
        if state.model_orchestrated:
            state.notes.append("Model-orchestrated mode: waiting for model-generated tool calls.")
            return

        if state.available_tools and "list_files" in state.available_tools:
            for scan_path in targeted_scan_paths[:3]:
                self._enqueue_unique(
                    state,
                    ActionRequest(
                        tool="list_files",
                        arguments={"path": scan_path, "limit": 200},
                        rationale="Inspect objective-hinted directories first to avoid missing key files in large trees.",
                    ),
                )
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="list_files",
                    arguments={"path": metadata.get("scan_path", "."), "limit": 200},
                    rationale="Capture the local workspace shape before taking action.",
                ),
            )

        for path in explicit_files[:3]:
            if "read_file" in state.available_tools:
                normalized = path.replace("\\", "/")
                candidate = Path(path)
                resolved = candidate if candidate.is_absolute() else (state.context.working_dir / candidate)
                should_read_directly = "/" in normalized or resolved.exists()
                if not should_read_directly:
                    continue
                self._enqueue_unique(
                    state,
                    ActionRequest(
                        tool="read_file",
                        arguments={"path": path},
                        rationale="Inspect explicit files mentioned in the objective.",
                    ),
                )

        configured_searches = list(metadata.get("search_queries", []) or [])
        if configured_searches and "web_search" in state.available_tools:
            self._schedule_search(
                state=state,
                query=str(configured_searches[0]),
                rationale="Use the configured search query to gather extra context when helpful.",
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

    def autonomous_replan(self, state: ExecutionState) -> bool:
        if str(state.mode).strip().lower() != "autonomous":
            return False
        before = len(state.pending)
        state.autonomous_replan_count += 1

        self._schedule_followup_search_if_empty(state)
        self._schedule_script_execution_from_read_files(state)
        self._schedule_exploratory_step(state)

        return len(state.pending) > before

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
            files = observation.content if isinstance(observation.content, list) else []
            self._handle_filesystem_snapshot_followups(state=state, files=files)
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
            self._handle_file_read_followups(
                state=state,
                path=str(path),
                content=str(observation.content),
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
            "model_final_answer": state.model_final_answer,
            "model_audit": state.model_audit,
            "human_readable": self._render_human_readable_summary(state),
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
        if state.model_marked_complete:
            return True
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
            elif observation.type in {"api_response", "search_results", "file_read", "file_write", "webpage_read"}:
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
        if state.model_final_answer:
            lines.append("")
            lines.append("Model Final Answer:")
            lines.append(state.model_final_answer)
        if state.model_audit:
            lines.append("")
            lines.append("Model Self-Audit:")
            passed = bool(state.model_audit.get("pass"))
            lines.append(f"- pass: {passed}")
            score = state.model_audit.get("score")
            if score is not None:
                lines.append(f"- score: {score}")
            reason = str(state.model_audit.get("reason", "")).strip()
            if reason:
                lines.append(f"- reason: {reason}")
            gaps = state.model_audit.get("gaps")
            if isinstance(gaps, list):
                for item in gaps[:5]:
                    lines.append(f"- gap: {item}")
        return "\n".join(lines).strip() + "\n"

    def _render_human_readable_summary(self, state: ExecutionState) -> str:
        completed = len([action for action in state.actions if action.status == "completed"])
        failed = len([action for action in state.actions if action.status == "failed"])
        observation_types = sorted(
            {
                action.observation.type
                for action in state.actions
                if action.status == "completed"
            }
        )

        intro = (
            f'The agent worked on the objective "{state.objective}" in '
            f"{state.context.working_dir}. It executed {len(state.actions)} step(s), "
            f"with {completed} completed and {failed} failed action(s)."
        )
        context_paragraph = "The loop stayed action-oriented: take action, observe result, adapt, and continue."
        evidence_paragraph = (
            "Execution evidence includes these observation types: "
            + (", ".join(observation_types) if observation_types else "none")
            + "."
        )
        notes_paragraph = (
            "Key notes: "
            + ("; ".join(state.notes[:4]) if state.notes else "none captured")
            + "."
        )

        return "\n\n".join([intro, context_paragraph, evidence_paragraph, notes_paragraph]).strip()

    def _initial_search_query(self, state: ExecutionState) -> str:
        configured = list(state.context.metadata.get("search_queries", []) or [])
        if configured:
            return str(configured[0])
        return self._compact_search_query(state.objective)

    def _enqueue_unique(
        self,
        state: ExecutionState,
        action: ActionRequest,
        *,
        front: bool = False,
    ) -> None:
        action_key = (action.tool, json.dumps(action.arguments, sort_keys=True, default=str))
        for record in state.actions:
            record_key = (record.tool, json.dumps(record.arguments, sort_keys=True, default=str))
            if record_key == action_key:
                return
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
            if not url.startswith(("http://", "https://")):
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

    def _handle_file_read_followups(
        self,
        *,
        state: ExecutionState,
        path: str,
        content: str,
    ) -> None:
        filename = Path(path).name.lower()

        if filename == "agents.md":
            derived_objective = self._extract_embedded_objective(content)
            if derived_objective:
                state.notes.append(f"Derived nested objective from {filename}: {derived_objective}")
                state.reflections.append(
                    "Reflection: the instruction file contains a more specific objective, so the agent should continue from that objective instead of stopping after inspection."
                )
                if state.model_feedback_available:
                    state.notes.append(
                        "Model feedback is available; deferring search query wording to model-generated tool calls."
                    )
                    return
                self._schedule_search(
                    state=state,
                    query=derived_objective,
                    rationale="The instruction file defines the real task, and the agent needs supporting knowledge to complete it.",
                )

    def _schedule_followup_search_if_empty(self, state: ExecutionState) -> None:
        if "web_search" not in state.available_tools:
            return
        if state.seen_search_queries and state.external_knowledge_acquired:
            return

        search_actions = [
            action
            for action in state.actions
            if action.tool == "web_search" and action.status == "completed"
        ]
        if not search_actions:
            return

        last_search = search_actions[-1]
        metadata = last_search.observation.metadata if isinstance(last_search.observation.metadata, dict) else {}
        count = int(metadata.get("count", 0) or 0)
        if count > 0:
            return

        if len(state.seen_search_queries) >= 3:
            return

        fallback = self._fallback_search_query(state)
        self._schedule_search(
            state=state,
            query=fallback,
            rationale="Previous search returned no results, so the agent should reformulate the query before stopping.",
        )

    def _schedule_exploratory_step(self, state: ExecutionState) -> None:
        if state.pending:
            return
        # Skip auto-search for purely local objectives (e.g., "read X, run Y, write Z").
        if self._objective_is_local_task(state.objective):
            return
        if "web_search" in state.available_tools and len(state.seen_search_queries) < 2:
            query = self._initial_search_query(state)
            if state.seen_search_queries:
                query = self._fallback_search_query(state)
            self._schedule_search(
                state=state,
                query=query,
                rationale="Exploration step: try an additional strategy when the queue is empty.",
            )

    def _schedule_script_execution_from_read_files(self, state: ExecutionState) -> None:
        if "run_command" not in state.available_tools:
            return
        objective = state.objective.lower()
        execution_markers = (
            "do the objective",
            "execute",
            "run",
            "perform",
            "complete",
        )
        if not any(marker in objective for marker in execution_markers):
            return

        read_paths: list[Path] = []
        for action in state.actions:
            if action.status != "completed" or action.observation.type != "file_read":
                continue
            raw_path = str((action.observation.metadata or {}).get("path", "")).strip()
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.suffix.lower() != ".py":
                continue
            read_paths.append(path)

        if not read_paths:
            return

        for path in read_paths[-3:]:
            command = self._python_command_for_path(state=state, path=path)
            if not command:
                continue
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="run_command",
                    arguments={"command": command},
                    rationale="Autonomous follow-up: execute the discovered Python script to gather concrete output for the objective.",
                ),
            )

    @staticmethod
    def _python_command_for_path(*, state: ExecutionState, path: Path) -> str:
        try:
            candidate = path.resolve()
        except OSError:
            return ""
        try:
            relative = candidate.relative_to(state.context.working_dir)
        except ValueError:
            return ""
        rel_text = relative.as_posix()
        if not rel_text:
            return ""
        return f"python {shlex.quote(rel_text)}"

    @staticmethod
    def _fallback_search_query(state: ExecutionState) -> str:
        source_text = state.objective
        for action in reversed(state.actions):
            if action.status != "completed" or action.observation.type != "file_read":
                continue
            content = str(action.observation.content or "").strip()
            if not content:
                continue
            source_text = content
            break

        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", source_text)
        tokens = [token for token in cleaned.split() if len(token) > 2]
        if not tokens:
            return state.objective
        compact = " ".join(tokens[:12])
        return compact

    @staticmethod
    def _objective_is_local_task(objective: str) -> bool:
        """Heuristic: avoid web_search when the objective is a local file/command workflow."""
        lowered = objective.lower()
        local_markers = (
            "workspace",
            "workspaces",
            "directory",
            "folder",
            "file",
            "read ",
            "write ",
            "run ",
            "using write_file",
            "using run_command",
            ".py",
            ".md",
            "/",
        )
        return any(marker in lowered for marker in local_markers)

    @staticmethod
    def _compact_search_query(text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = [token for token in cleaned.split() if len(token) > 2]
        if not tokens:
            return text.strip()
        return " ".join(tokens[:12])

    def _handle_filesystem_snapshot_followups(
        self,
        *,
        state: ExecutionState,
        files: list[Any],
    ) -> None:
        if "read_file" not in state.available_tools:
            return

        normalized_files: list[str] = []
        for item in files:
            if isinstance(item, str) and item.strip():
                normalized_files.append(item.replace("\\", "/"))
        if not normalized_files:
            return

        objective_paths = self._extract_explicit_paths(state.objective)
        if not objective_paths:
            return

        directory_hints = self._extract_directory_hints(state.objective)
        for raw_path in objective_paths:
            token = raw_path.replace("\\", "/").lstrip("./")
            if not token:
                continue
            matches = self._match_paths_from_snapshot(
                candidate=token,
                files=normalized_files,
                directory_hints=directory_hints,
            )
            if len(matches) != 1:
                continue
            self._enqueue_unique(
                state,
                ActionRequest(
                    tool="read_file",
                    arguments={"path": matches[0]},
                    rationale="Resolve explicit filenames to real workspace paths before concluding the task.",
                ),
            )

    @staticmethod
    def _extract_directory_hints(objective: str) -> list[str]:
        hints: list[str] = []
        patterns = (
            r"(?:directory|folder)\s+([A-Za-z0-9_.-]+)",
            r"in\s+([A-Za-z0-9_.-]+)\s+(?:directory|folder)",
            r"(?:in|inside|under)\s+([A-Za-z0-9_.-]+)",
        )
        seen_casefold: set[str] = set()
        for pattern in patterns:
            for match in re.findall(pattern, objective, flags=re.IGNORECASE):
                hint = str(match or "").strip().strip("./")
                key = hint.casefold()
                if hint and key not in seen_casefold:
                    seen_casefold.add(key)
                    hints.append(hint)
        return hints

    @staticmethod
    def _match_paths_from_snapshot(
        *,
        candidate: str,
        files: list[str],
        directory_hints: list[str],
    ) -> list[str]:
        candidate_name = Path(candidate).name.lower()
        has_directory = "/" in candidate
        matches: list[str] = []
        for file_path in files:
            normalized = file_path.strip().lstrip("./")
            if not normalized:
                continue
            lowered = normalized.lower()
            if has_directory:
                if lowered == candidate.lower() or lowered.endswith(f"/{candidate.lower()}"):
                    matches.append(normalized)
            else:
                if Path(lowered).name == candidate_name:
                    matches.append(normalized)

        if len(matches) <= 1 or not directory_hints:
            return sorted(dict.fromkeys(matches))

        preferred = [
            path
            for path in matches
            if any(f"/{hint.lower()}/" in f"/{path.lower()}/" for hint in directory_hints)
        ]
        if len(preferred) == 1:
            return preferred
        return sorted(dict.fromkeys(matches))

    def _candidate_scan_paths(self, state: ExecutionState) -> list[str]:
        objective = state.objective
        hints = self._extract_directory_hints(objective)
        raw_candidates: list[str] = []
        if "workspaces" in objective.lower():
            raw_candidates.append("workspaces")
        for hint in hints:
            raw_candidates.append(hint)
            raw_candidates.append(f"workspaces/{hint}")

        unique_paths: list[str] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            resolved = self._resolve_existing_relative_dir(
                working_dir=state.context.working_dir,
                relative_path=candidate,
            )
            if not resolved:
                continue
            key = resolved.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(resolved)
        return unique_paths

    @staticmethod
    def _resolve_existing_relative_dir(*, working_dir: Path, relative_path: str) -> str:
        raw = str(relative_path or "").strip().strip("/")
        if not raw:
            return ""
        rel = Path(raw)
        if rel.is_absolute():
            return ""

        direct = (working_dir / rel).resolve()
        if direct.is_dir():
            return rel.as_posix()

        current = working_dir
        resolved_parts: list[str] = []
        for part in rel.parts:
            if part in {"", ".", ".."}:
                continue
            try:
                directory_entries = [entry for entry in current.iterdir() if entry.is_dir()]
            except OSError:
                return ""
            match_name = ""
            lower_part = part.casefold()
            for entry in directory_entries:
                if entry.name.casefold() == lower_part:
                    match_name = entry.name
                    break
            if not match_name:
                return ""
            resolved_parts.append(match_name)
            current = current / match_name

        return "/".join(resolved_parts)

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
        if strategy == "read_webpage":
            return
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

    @staticmethod
    def _extract_embedded_objective(content: str) -> str:
        patterns = (
            r"Your task is to\s+(.+?)(?:\.\s|\n|$)",
            r"Objective:\s*(.+?)(?:\n|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return re.sub(r"\s+", " ", match.group(1)).strip()
        return ""

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
        on_action = config.get("on_action")
        raw_max_steps = config.get("max_steps", 8)
        try:
            configured_max_steps = int(raw_max_steps)
        except (TypeError, ValueError):
            configured_max_steps = 8
        max_steps = configured_max_steps if configured_max_steps > 0 else 0
        mode = str(config.get("mode", "autonomous"))
        model_feedback_enabled = bool(config.get("model_feedback", True)) and mode.strip().lower() == "autonomous"
        max_done_token_pokes = max(0, int(config.get("max_done_token_pokes", 0)))
        max_no_progress_streak = max(1, int(config.get("max_no_progress_streak", 3)))
        model_orchestrated_requested = bool(config.get("model_orchestrated", False))
        done_stop_token = str(config.get("done_stop_token", "DONE_STOP_AUTONOMOUS")).strip() or "DONE_STOP_AUTONOMOUS"
        feedback_model = self._build_feedback_model(config=config) if model_feedback_enabled else None
        model_orchestrated = bool(model_feedback_enabled and feedback_model is not None and model_orchestrated_requested)
        require_done_stop_token = bool(
            config.get("require_done_stop_token", model_orchestrated_requested)
        )
        model_backend_missing_for_orchestrated = bool(
            model_feedback_enabled and model_orchestrated_requested and feedback_model is None
        )
        local_feedback_fallback = bool(
            model_feedback_enabled and feedback_model is None and not model_orchestrated_requested
        )
        memory_enabled = bool(config.get("memory_enabled", True))
        memory_hits = self.memory_store.search(objective, limit=3) if memory_enabled else []
        plan_examples = self._load_plan_examples(
            objective=objective,
            working_dir=runtime_context.working_dir,
            limit=2,
        )
        state = ExecutionState(
            objective=objective,
            context=runtime_context,
            max_steps=max_steps,
            mode=mode,
            available_tools=self.registry.names(),
            memory_hits=memory_hits,
            plan_examples=plan_examples,
            model_orchestrated=model_orchestrated,
            model_feedback_available=feedback_model is not None,
            require_done_stop_token=require_done_stop_token,
            done_stop_token=done_stop_token,
        )
        self.planner.bootstrap(state)
        if model_orchestrated:
            state.notes.append(
                "Model-orchestrated execution is enabled."
            )
        if state.require_done_stop_token:
            state.notes.append(
                f"Autonomous stop token required: '{state.done_stop_token}'."
            )
        if model_backend_missing_for_orchestrated:
            state.notes.append(
                "Model-orchestrated mode requested but no external model backend is available."
            )
            success, result, confidence = self.planner.finalize(state)
            result["stop_reason"] = "model_backend_unavailable_for_model_orchestrated"
            return {
                "success": False,
                "steps": len(state.actions),
                "actions": [action.to_dict() for action in state.actions],
                "result": result,
                "confidence": confidence,
            }
        if local_feedback_fallback:
            state.notes.append(
                "Model feedback is enabled but no external model backend is available; using local fallback replanner."
            )

        while True:
            if max_steps > 0 and len(state.actions) >= max_steps:
                break
            action = self.planner.next_action(state)
            if action is None:
                if state.model_marked_complete:
                    if state.require_done_stop_token and not state.done_stop_seen:
                        state.notes.append(
                            f"Model marked completion without token '{state.done_stop_token}'; continuing."
                        )
                        state.model_marked_complete = False
                    else:
                        state.notes.append("Stopping: objective appears complete from model synthesis.")
                        break

                should_request_model_feedback = feedback_model is not None and (
                    not state.require_done_stop_token
                    or (
                        not state.done_stop_seen
                        and (max_done_token_pokes == 0 or state.done_token_poke_count < max_done_token_pokes)
                    )
                )
                if should_request_model_feedback:
                    if state.require_done_stop_token and not state.done_stop_seen:
                        state.done_token_poke_count += 1
                    before = len(state.pending)
                    suggested_actions = self._model_feedback_actions(
                        state=state,
                        model=feedback_model,
                    )
                    for suggested_action in suggested_actions:
                        self.planner._enqueue_unique(state, suggested_action)
                    if len(state.pending) > before or state.model_marked_complete:
                        continue

                if self.planner.autonomous_replan(state):
                    continue

                if local_feedback_fallback:
                    suggested_action = self._local_feedback_action(state=state)
                    if suggested_action is not None:
                        self.planner._enqueue_unique(state, suggested_action)
                        continue

                if state.require_done_stop_token:
                    if (
                        feedback_model is not None
                        and (max_done_token_pokes == 0 or state.done_token_poke_count < max_done_token_pokes)
                    ):
                        state.notes.append(
                            "Completion token still missing; continuing autonomous pokes with latest evidence."
                        )
                        continue
                    if max_done_token_pokes > 0 and state.done_token_poke_count >= max_done_token_pokes:
                        state.notes.append(
                            f"Stopping: completion token '{state.done_stop_token}' not received after "
                            f"{state.done_token_poke_count} model pokes."
                        )
                        break
                    if feedback_model is None:
                        state.notes.append(
                            f"Stopping: token '{state.done_stop_token}' is required but no model backend is available."
                        )
                        break
                    continue

                if self.planner._is_success(state):
                    state.notes.append("Stopping: objective appears completed based on gathered evidence.")
                    break
                if state.no_progress_streak >= max_no_progress_streak:
                    state.notes.append("Stopping: no meaningful progress is being made.")
                    break
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
            self._update_no_progress_streak(state=state, record=record)
            if callable(on_action):
                try:
                    on_action(
                        {
                            "index": len(state.actions),
                            "tool": record.tool,
                            "status": record.status,
                            "observation_type": record.observation.type,
                            "arguments": dict(record.arguments),
                            "rationale": record.rationale,
                        }
                    )
                except Exception:
                    pass

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

    @staticmethod
    def _update_no_progress_streak(*, state: ExecutionState, record: ActionRecord) -> None:
        if StarforgeRuntime._action_made_progress(record):
            state.no_progress_streak = 0
            return
        state.no_progress_streak += 1

    @staticmethod
    def _action_made_progress(record: ActionRecord) -> bool:
        if record.status != "completed":
            return False
        observation = record.observation
        if observation.type == "command_result":
            content = observation.content if isinstance(observation.content, dict) else {}
            return int(content.get("exit_code", 1)) == 0
        if observation.type in {"search_results", "filesystem_snapshot"}:
            content = observation.content if isinstance(observation.content, list) else []
            return bool(content)
        if observation.type in {"file_read", "webpage_read"}:
            return bool(str(observation.content or "").strip())
        if observation.type in {"file_write", "api_response"}:
            return True
        return observation.type != "tool_error"

    @staticmethod
    def _build_feedback_model(config: dict[str, Any]) -> Any | None:
        if _ASSISTANT_BUILD_MODEL is None:
            return None
        model_name = str(
            config.get("feedback_model_name")
            or config.get("model_name")
            or os.getenv("STARFORGE_MODEL_NAME", "qwen2.5:7b")
        ).strip()
        provider = str(config.get("feedback_provider") or os.getenv("STARFORGE_MODEL_PROVIDER", "auto")).strip()
        try:
            model = _ASSISTANT_BUILD_MODEL(model_name=model_name, provider=provider)
        except Exception:
            return None
        if model is None:
            return None
        provider_name = str(getattr(model, "provider", "")).strip().lower()
        if provider_name in {"", "unknown"} or model.__class__.__name__.lower() == "fallbackmodel":
            return None
        return model

    @staticmethod
    def _keywords(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-zA-Z0-9_]{3,}", str(text or "").lower())}

    @staticmethod
    def _load_plan_examples(
        *,
        objective: str,
        working_dir: Path,
        limit: int = 2,
    ) -> list[dict[str, Any]]:
        plans_dir = (working_dir / "memory" / "plans").resolve()
        if not plans_dir.is_dir():
            return []
        objective_terms = StarforgeRuntime._keywords(objective)
        if not objective_terms:
            return []

        ranked: list[tuple[float, dict[str, Any]]] = []
        for path in sorted(plans_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            title = str(payload.get("title", "")).strip()
            goal = str(payload.get("goal", "")).strip()
            terms = StarforgeRuntime._keywords(f"{title} {goal}")
            if not terms:
                continue
            overlap = objective_terms.intersection(terms)
            if not overlap:
                continue
            score = len(overlap) / max(1, len(objective_terms))
            todos = payload.get("todos", [])
            normalized_todos: list[str] = []
            if isinstance(todos, list):
                for item in todos:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text", "")).strip()
                    status = str(item.get("status", "")).strip().lower()
                    if not text:
                        continue
                    if status == "done":
                        continue
                    normalized_todos.append(text)
                    if len(normalized_todos) >= 5:
                        break
            ranked.append(
                (
                    score,
                    {
                        "title": title,
                        "goal": goal,
                        "todos": normalized_todos,
                    },
                )
            )

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[: max(1, limit)]]

    def _model_feedback_actions(
        self,
        *,
        state: ExecutionState,
        model: Any,
    ) -> list[ActionRequest]:
        if _ASSISTANT_PARSE_JSON_PAYLOAD is None:
            return []
        messages = self._build_model_feedback_messages(state)
        try:
            raw = str(model.generate(messages) or "").strip()
        except Exception as exc:
            state.notes.append(f"Model feedback failed: {exc}")
            return []
        if not raw:
            return []

        suggested = self._parse_model_tool_calls(state=state, raw=raw)
        if suggested:
            state.reflections.append(
                "Reflection: model feedback proposed follow-up tool calls, so autonomous execution should continue."
            )
            return suggested
        if state.require_done_stop_token and state.done_stop_token.lower() in raw.lower():
            state.done_stop_seen = True
            state.model_marked_complete = True
            state.model_final_answer = raw[:4000]
            state.notes.append(
                f"Model supplied completion with required token '{state.done_stop_token}'."
            )
            return []

        payload = _ASSISTANT_PARSE_JSON_PAYLOAD(raw)
        if not isinstance(payload, dict):
            return []

        done = bool(payload.get("done")) or str(payload.get("status", "")).strip().lower() in {"done", "complete"}
        final_answer = str(payload.get("final_answer") or payload.get("summary") or "").strip()
        if done or final_answer:
            if final_answer:
                state.model_final_answer = final_answer[:4000]
            if state.require_done_stop_token:
                token = state.done_stop_token
                token_seen = token.lower() in raw.lower()
                if token_seen:
                    state.done_stop_seen = True
                    state.model_marked_complete = True
                    state.notes.append(
                        f"Model supplied completion with required token '{token}'."
                    )
                else:
                    state.model_marked_complete = False
                    state.notes.append(
                        f"Model signaled completion without required token '{token}'."
                    )
            else:
                state.model_marked_complete = True
                state.notes.append("Model supplied a final synthesis from gathered evidence.")
        return []

    def _local_feedback_action(self, *, state: ExecutionState) -> ActionRequest | None:
        if "web_search" in state.available_tools:
            command_actions = [
                action
                for action in state.actions
                if action.status == "completed"
                and action.observation.type == "command_result"
                and isinstance(action.observation.content, dict)
                and int(action.observation.content.get("exit_code", 1)) == 0
            ]
            if command_actions:
                last_command = command_actions[-1]
                content = last_command.observation.content if isinstance(last_command.observation.content, dict) else {}
                signal = str(content.get("stdout") or content.get("stderr") or "").strip()
                if signal:
                    compact_signal = re.sub(r"\s+", " ", signal)[:140]
                    query = f"{state.objective} {compact_signal}".strip()
                    if query and query not in state.seen_search_queries:
                        state.seen_search_queries.add(query)
                        return ActionRequest(
                            tool="web_search",
                            arguments={"query": query, "limit": 5},
                            rationale="Local fallback replanner: use command output as evidence to drive another research iteration.",
                        )

        if "read_webpage" in state.available_tools and not state.external_knowledge_acquired:
            for action in reversed(state.actions):
                if action.status != "completed" or action.observation.type != "search_results":
                    continue
                content = action.observation.content if isinstance(action.observation.content, list) else []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url", "")).strip()
                    if not url or not url.startswith(("http://", "https://")):
                        continue
                    if url in state.seen_urls:
                        continue
                    state.seen_urls.add(url)
                    return ActionRequest(
                        tool="read_webpage",
                        arguments={"url": url},
                        rationale="Local fallback replanner: inspect a concrete source to continue evidence-based iteration.",
                    )

        return None

    @staticmethod
    def _tool_aliases() -> dict[str, str]:
        return {
            "search_web": "web_search",
            "web_search": "web_search",
            "web_search(query: str)": "web_search",
            "web_search(query:str)": "web_search",
            "read_web": "read_webpage",
            "read_webpage": "read_webpage",
            "execute_command": "run_command",
            "run_command": "run_command",
            "read_file": "read_file",
            "write_file": "write_file",
            "list_files": "list_files",
            "http_request": "http_request",
        }

    def _action_request_from_candidate(
        self,
        *,
        state: ExecutionState,
        candidate: dict[str, Any],
        default_rationale: str,
    ) -> ActionRequest | None:
        aliases = self._tool_aliases()
        raw_name = str(candidate.get("name") or candidate.get("tool") or "").strip()
        mapped = aliases.get(raw_name, raw_name)
        if mapped not in state.available_tools:
            return None
        args = candidate.get("args")
        if args is None:
            args = candidate.get("arguments")
        if isinstance(args, str) and mapped == "web_search":
            args = {"query": args}
        if args is None and mapped == "web_search":
            query = str(candidate.get("query") or "").strip()
            if query:
                args = {"query": query}
        if not isinstance(args, dict):
            return None
        rationale = str(candidate.get("rationale") or "").strip() or default_rationale
        return ActionRequest(
            tool=mapped,
            arguments=args,
            rationale=rationale,
        )

    def _parse_model_tool_calls(self, *, state: ExecutionState, raw: str) -> list[ActionRequest]:
        candidates: list[dict[str, Any]] = []
        if _ASSISTANT_PARSE_TOOL_CALLS is not None:
            try:
                candidates.extend(_ASSISTANT_PARSE_TOOL_CALLS(raw))
            except Exception:
                candidates = []
        if _ASSISTANT_PARSE_JSON_PAYLOAD is not None:
            payload = _ASSISTANT_PARSE_JSON_PAYLOAD(raw)
            if isinstance(payload, dict):
                raw_calls = payload.get("calls")
                if isinstance(raw_calls, list):
                    for item in raw_calls:
                        if isinstance(item, dict):
                            candidates.append(item)
                name = payload.get("tool") or payload.get("name")
                args = payload.get("args") or payload.get("arguments") or {}
                if isinstance(name, str) and isinstance(args, dict):
                    candidates.append({"name": name, "args": args})
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        candidates.append(item)

        actions: list[ActionRequest] = []
        for call in candidates:
            if not isinstance(call, dict):
                continue
            action = self._action_request_from_candidate(
                state=state,
                candidate=call,
                default_rationale="Model-feedback replan: choose next action from latest observations.",
            )
            if action is not None:
                actions.append(action)
        return actions

    @staticmethod
    def _build_model_feedback_messages(
        state: ExecutionState,
    ) -> list[dict[str, str]]:
        action_history = StarforgeRuntime._action_history_json(state)
        memory_hits_text = json.dumps(state.memory_hits[:3], ensure_ascii=False)
        plan_examples_text = json.dumps(state.plan_examples[:2], ensure_ascii=False)
        system_prompt = (
            "You are the autonomous replanner for Starforge.\n"
            "Before proposing any action, define an internal completion standard for the objective "
            "(required evidence + final deliverable constraints) and use it to evaluate progress on every turn.\n"
            "Choose the next tool call(s) that best advance the objective.\n"
            "You may revise the previous approach if new evidence suggests a better move.\n"
            "You may return a short subplan as multiple tool calls when one step is not enough.\n"
            "If current output only partially satisfies the objective, propose concrete follow-up actions.\n"
            "Output JSON only.\n"
            "Use either:\n"
            "- {\"tool\":\"<name>\",\"args\":{...},\"rationale\":\"...\"}\n"
            "- {\"calls\":[{\"tool\":\"<name>\",\"args\":{...},\"rationale\":\"...\"}, ...]}\n"
            "If the objective is fully complete, output: {\"done\":true,\"final_answer\":\"...\"}.\n"
            "Do not invent tools outside the allowed list."
        )
        if state.require_done_stop_token:
            system_prompt += (
                f"\nStrict completion rule: only declare completion when final_answer includes "
                f"the exact token '{state.done_stop_token}'."
                "\nBefore every completion attempt, verify the objective is fully satisfied; "
                "if not, return concrete tool calls instead of done=true."
            )
        user_prompt = (
            f"Objective:\n{state.objective}\n\n"
            f"Available tools:\n{', '.join(state.available_tools)}\n\n"
            f"Relevant memory hits (if any):\n{memory_hits_text}\n\n"
            f"Relevant prior plans from memory/plans (if any):\n{plan_examples_text}\n\n"
            f"All previous tool results as JSON:\n{action_history}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _action_history_json(state: ExecutionState) -> str:
        if not state.actions:
            return "[]"
        payload: list[dict[str, Any]] = []
        for index, action in enumerate(state.actions, start=1):
            payload.append(
                {
                    "index": index,
                    "tool": action.tool,
                    "status": action.status,
                    "arguments": action.arguments,
                    "rationale": action.rationale,
                    "observation": action.observation.to_dict(),
                }
            )
        return json.dumps(payload, ensure_ascii=False)
