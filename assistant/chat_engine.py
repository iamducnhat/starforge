from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cli_format import (StreamRenderer, extract_answer_text,
                         print_answer_only, print_formatted_output,
                         print_phase, print_tool_event, print_tool_start)
from .model import BaseModel
from .tool_calls import parse_tool_calls
from .tools import ToolSystem
from .utils import get_env_bool, get_env_float, get_env_int, parse_json_payload


@dataclass
class PlanStep:
    step_id: int
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    expected_output: str = ""
    status: str = "pending"

    def short_label(self) -> str:
        dep = f" deps={self.depends_on}" if self.depends_on else ""
        expect = (
            f" -> {self.expected_output}" if isinstance(self.expected_output, str) and self.expected_output else ""
        )
        return f"[{self.step_id}] {self.action}{dep}{expect}"


@dataclass
class TaskState:
    goal: str
    steps: list[PlanStep]
    current_step: int = 0
    completed_step_ids: set[int] = field(default_factory=set)
    history: list[dict[str, Any]] = field(default_factory=list)
    dependency_order: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.refresh_dependency_order()

    def refresh_dependency_order(self) -> None:
        ordered = [step.step_id for step in self.steps if step.step_id > 0]
        self.dependency_order = ordered
        if not self.steps:
            self.current_step = 0
            return
        if self.current_step < 0 or self.current_step >= len(self.steps):
            self.current_step = 0

    def step_map(self) -> dict[int, PlanStep]:
        return {step.step_id: step for step in self.steps}

    def current_step_text(self) -> str:
        if not self.steps:
            return "No plan available."
        idx = max(0, min(self.current_step, len(self.steps) - 1))
        return self.steps[idx].short_label()

    def runnable_step_ids(self) -> list[int]:
        step_map = self.step_map()
        runnable: list[int] = []
        for step_id in self.dependency_order:
            step = step_map.get(step_id)
            if (
                step is None
                or step_id in self.completed_step_ids
                or step.status != "pending"
            ):
                continue
            if all(dep in self.completed_step_ids for dep in step.depends_on):
                runnable.append(step_id)
        return runnable

    def next_runnable_index(self) -> int | None:
        if not self.steps:
            return None
        for step_id in self.runnable_step_ids():
            for idx, step in enumerate(self.steps):
                if step.step_id == step_id:
                    return idx
        return None

    def remaining_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "pending")


@dataclass
class RepairState:
    attempts: int = 0
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    tried_fixes: list[str] = field(default_factory=list)
    tried_hypotheses: list[str] = field(default_factory=list)


class ChatEngine:
    def __init__(
        self,
        model: BaseModel,
        tools: ToolSystem,
        system_prompt: str,
        max_history: int = 14,
        max_tool_rounds: int = 4,
        autonomous_enabled: bool = False,
        autonomous_steps: int = 6,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_tool_rounds = max_tool_rounds
        self.autonomous_enabled = autonomous_enabled
        self.autonomous_steps = (
            0 if autonomous_steps <= 0 else max(1, min(autonomous_steps, 1000))
        )
        self.autonomous_default_objective = (
            "Continuously self-improve this project inside workspace root only: "
            "plan tasks, research, refactor safely, and keep iterating."
        )
        self.history: list[dict[str, str]] = []
        self.supervision_log = Path("memory/tool_supervision.jsonl")
        self.tool_finetune_log = Path("memory/tool_finetune_samples.jsonl")
        self.interaction_log = Path("memory/interaction_trajectories.jsonl")
        self.session_dir = Path("memory/sessions")
        self._session_name: str = datetime.now(timezone.utc).strftime(
            "session_%Y%m%d_%H%M%S"
        )
        self._session_start: str = datetime.now(timezone.utc).isoformat()
        self._session_named: bool = False  # True once AI name has been assigned
        self._session_save_ptr: int = 0  # messages already written to the log
        self._intent_cache: dict[str, dict[str, Any]] = {}
        self.compact_context_enabled = get_env_bool("ASSISTANT_COMPACT_CONTEXT", True)
        self.max_context_chars = get_env_int("ASSISTANT_MAX_CONTEXT_CHARS", 180000)
        self.compacted_context_note = ""
        self.auto_stop_enabled = False
        self.tool_reflection_enabled = get_env_bool("ASSISTANT_TOOL_REFLECTION", True)
        self.autonomous_plan_step_cap = get_env_int("ASSISTANT_PLAN_STEP_CAP", 8)
        self.execution_validation_threshold = max(
            0.0, min(get_env_float("ASSISTANT_EXEC_VALIDATION_THRESHOLD", 0.55), 1.0)
        )
        self.autonomous_skill_learning_enabled = get_env_bool(
            "ASSISTANT_AUTO_LEARN_SKILLS", True
        )
        self.autonomous_validate_changes = get_env_bool(
            "ASSISTANT_AUTO_VALIDATE_CHANGES", True
        )
        self.autonomous_test_repair_attempts = max(
            0, min(get_env_int("ASSISTANT_AUTO_TEST_REPAIR_ATTEMPTS", 3), 5)
        )
        self.autonomous_token_budget = max(
            0, get_env_int("ASSISTANT_AUTO_TOKEN_BUDGET", 0)
        )
        self.interaction_logging_enabled = get_env_bool(
            "ASSISTANT_LOG_INTERACTIONS", True
        )
        # Set to a callable (e.g. cli_format.print_phase) to display internal
        # phase labels in the CLI.  Left as None in server mode to stay silent.
        self.on_status: Callable[[str], None] | None = None

    def _status(self, label: str) -> None:
        """Call on_status callback if configured (CLI mode only)."""
        if self.on_status is not None:
            try:
                self.on_status(label)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _generate_session_name(self) -> str:
        """Ask the model for a short slug name based on the conversation topic.
        Returns a sanitised slug, or falls back to the current timestamp name."""
        # Find the best topic text: prefer the actual human objective over
        # autonomous-prompt boilerplate.
        topic = ""
        for m in self.history:
            if m.get("role") != "user":
                continue
            content = m.get("content", "")
            # Autonomous prompts embed the real objective after "Objective:"
            obj_match = re.search(r"(?i)^Objective:\s*(.+)$", content, re.MULTILINE)
            if obj_match:
                topic = obj_match.group(1).strip()
                break
            # Skip the internal autonomous-continue boilerplate lines
            if content.startswith("Autonomous "):
                continue
            topic = content.strip()
            break

        if not topic:
            return self._session_name

        prompt = [
            {
                "role": "user",
                "content": (
                    "Output a snake_case session name (3-5 words) that describes this topic.\n"
                    "Rules: only lowercase letters, digits and underscores. No punctuation. No explanation.\n"
                    f"Topic: {topic[:300]}"
                ),
            }
        ]
        try:
            raw = self.model.generate(prompt)
            # Strip any chain-of-thought / think blocks
            raw = self._strip_thinking(raw)
            # Take only the first non-empty line so stray prose is ignored
            first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
            slug = re.sub(r"[^\w]+", "_", first_line).strip("_").lower()
            slug = re.sub(r"_+", "_", slug)[:60]
            if slug:
                return slug
        except Exception:
            pass
        return self._session_name

    def _flush_to_session_log(self) -> None:
        """Write any unsaved messages to the session log using the current
        (possibly timestamp) name. No name generation happens here.
        Safe to call from inside _compact_context_if_needed."""
        ptr = self._session_save_ptr
        if ptr >= len(self.history):
            return
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{self._session_name}.jsonl"
        self._append_session_messages(path=path, ptr=ptr, session_name=self._session_name)

    def _append_session_messages(self, path: Path, ptr: int, session_name: str) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                if ptr == 0:
                    header = {
                        "type": "header",
                        "name": session_name,
                        "created_at": self._session_start,
                    }
                    f.write(json.dumps(header, ensure_ascii=False) + "\n")
                now = datetime.now(timezone.utc).isoformat()
                for msg in self.history[ptr:]:
                    entry = dict(msg)
                    entry["logged_at"] = now
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._session_save_ptr = len(self.history)
        except Exception:
            pass

    def _save_session(self, name: str | None = None) -> str:
        """Append new messages to memory/sessions/<name>.jsonl.
        The file is never rewritten — only new entries are appended (log style).
        Returns the session name used."""
        if not self.history:
            return self._session_name

        # On first auto-save (no explicit name), ask the model for an AI slug
        if name is None and not self._session_named:
            old_name = self._session_name
            generated = self._generate_session_name()
            self._session_name = generated
            self._session_named = True
            # Rename the existing log file if it was already created
            old_path = self.session_dir / f"{old_name}.jsonl"
            new_path = self.session_dir / f"{generated}.jsonl"
            try:
                if old_path.exists():
                    old_path.rename(new_path)
            except Exception:
                pass

        if name:
            # Explicit /session save <name>: flush under that name from scratch
            used_name = name.replace(" ", "_").replace("/", "-").replace("\\", "-")
            self._session_name = used_name
            self._session_named = True
            self._session_save_ptr = 0  # rewrite from scratch into new file

        used_name = self._session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        path = self.session_dir / f"{used_name}.jsonl"
        self._append_session_messages(
            path=path, ptr=self._session_save_ptr, session_name=used_name
        )
        return used_name

    def _load_session(self, name: str) -> bool | None:
        """Load a session by name (or partial name) from the .jsonl log.
        Falls back to legacy .json for old sessions.
        Returns True on success, None if ambiguous, False if not found."""

        def _find_path(ext: str) -> Path | None:
            exact = self.session_dir / f"{name}{ext}"
            if exact.exists():
                return exact
            matches = sorted(self.session_dir.glob(f"*{name}*{ext}"))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None  # signal ambiguous
            return False  # type: ignore[return-value]

        # Try .jsonl first, then legacy .json
        result = _find_path(".jsonl")
        if result is None:
            return None  # ambiguous
        if result is False:  # type: ignore[comparison-overlap]
            result = _find_path(".json")
            if result is None:
                return None
            if result is False:  # type: ignore[comparison-overlap]
                return False
            # Legacy .json load
            try:
                data = json.loads(result.read_text(encoding="utf-8"))
                self.history = data.get("history", [])
                self._session_name = data.get("name", result.stem)
                self._session_start = data.get(
                    "created_at", datetime.now(timezone.utc).isoformat()
                )
                self._session_named = True
                self._session_save_ptr = len(self.history)
                return True
            except Exception:
                return False

        # .jsonl load — reconstruct history from log lines
        try:
            history: list[dict[str, str]] = []
            session_name = result.stem
            created_at = datetime.now(timezone.utc).isoformat()
            for raw_line in result.read_text(encoding="utf-8").splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    continue
                if entry.get("type") == "header":
                    session_name = entry.get("name", session_name)
                    created_at = entry.get("created_at", created_at)
                    continue
                # It's a message entry
                msg = {k: v for k, v in entry.items() if k != "logged_at"}
                if msg.get("role"):
                    history.append(msg)  # type: ignore[arg-type]
            self.history = history
            self._session_name = session_name
            self._session_start = created_at
            self._session_named = True
            self._session_save_ptr = len(self.history)
            return True
        except Exception:
            return False

    def _list_sessions(self) -> list[str]:
        """Return sorted list of saved session names (.jsonl preferred, .json legacy)."""
        if not self.session_dir.exists():
            return []
        names: set[str] = set()
        for p in self.session_dir.glob("*.jsonl"):
            names.add(p.stem)
        for p in self.session_dir.glob("*.json"):
            names.add(p.stem)
        return sorted(names)

    def _new_session(self) -> None:
        """Save current session then start a fresh one."""
        self._save_session()
        self.history = []
        self._session_name = datetime.now(timezone.utc).strftime(
            "session_%Y%m%d_%H%M%S"
        )
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._session_named = False
        self._session_save_ptr = 0

    def _generate_with_stream_fallback(self, messages: list[dict[str, str]]) -> str:
        text = self.model.generate(messages)
        if "endpoint available but incompatible" not in text.lower():
            return text

        chunks: list[str] = []
        for chunk in self.model.stream_generate(messages):
            if chunk:
                chunks.append(chunk)
        streamed = "".join(chunks).strip()
        return streamed or text

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Keep hidden reasoning out of history to reduce response drift/repetition.
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def _recover_final_answer(self, raw_assistant_text: str) -> tuple[str, str] | None:
        self._status("recovering final answer from model output…")
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "user",
                "content": (
                    "Return the final user-facing answer only. "
                    "No reasoning, no tool call JSON, no analysis."
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        clean = self._strip_thinking(recovered)
        if clean:
            return recovered, clean
        return None

    def _recover_action_or_answer(
        self, user_message: str, raw_assistant_text: str
    ) -> tuple[str, list[dict[str, object]], str]:
        self._status("recovering action or answer…")
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "system",
                "content": (
                    "Internal control message.\n"
                    f"Original user request: {user_message}\n"
                    "Continue now with exactly one actionable output for the original user request.\n"
                    'Option A: JSON tool call only ({"tool":"...","args":{...}} or {"tool_calls":[...]}).\n'
                    "Option B: final user-facing answer text only.\n"
                    "Do not output reasoning. Do not discuss this control message."
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        return recovered, parse_tool_calls(recovered), self._strip_thinking(recovered)

    @staticmethod
    def _contains_internal_prompt_echo(text: str) -> bool:
        t = text.lower()
        patterns = (
            "tool execution is complete",
            "continue generation using the latest tool result",
            "if another tool is needed",
            "otherwise return final user-facing answer text only",
            "internal control message",
            "option a:",
            "option b:",
            "original user request:",
        )
        return any(p in t for p in patterns)

    @staticmethod
    def _contains_tool_denial(text: str) -> bool:
        t = text.lower()
        patterns = (
            "can't use tool",
            "cannot use tool",
            "i can not use tool",
            "i can't access tool",
            "i cannot access tool",
            "since i can't use tool",
            "since i cannot use tool",
            "unable to use tool",
            "do not have access to tool",
            "don't have access to tool",
        )
        return any(p in t for p in patterns)

    def _recover_tool_calls(
        self, user_message: str, raw_assistant_text: str
    ) -> list[dict[str, object]]:
        self._status("recovering tool calls (model denied tools)…")
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "user",
                "content": (
                    "You CAN use tools in this runtime.\n"
                    "If reliable external info is needed, return JSON tool call now.\n"
                    "Use canonical tool names only (search_web/read_web/scrape_web). "
                    "Do not use google_search or web_search.\n"
                    "Format only:\n"
                    '{"tool":"name","args":{...}} or {"tool_calls":[...]}\n'
                    "If no tool is needed, return {}.\n"
                    f"Latest user message: {user_message}"
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        return parse_tool_calls(recovered)

    def _log_supervision(
        self, event: str, user_message: str, assistant_text: str
    ) -> None:
        try:
            self.supervision_log.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "event": event,
                "user_message": user_message,
                "assistant_text": self._strip_thinking(assistant_text)[:2000],
            }
            with self.supervision_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_tool_training_sample(
        self,
        user_message: str,
        assistant_text: str,
        tool_calls: list[dict[str, object]],
    ) -> None:
        try:
            self.tool_finetune_log.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_text},
                ],
                "tool_calls": tool_calls,
            }
            with self.tool_finetune_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _canonical_tool_call_json(tool_calls: list[dict[str, Any]]) -> str:
        payload_calls: list[dict[str, Any]] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "") or call.get("tool", "")).strip()
            args = call.get("args", {})
            if not name:
                continue
            if not isinstance(args, dict):
                args = {}
            payload_calls.append({"tool": name, "args": args})
        return json.dumps({"tool_calls": payload_calls}, ensure_ascii=False)

    @staticmethod
    def _truncate_text(text: str, limit: int = 800) -> str:
        raw = str(text or "").strip()
        if len(raw) <= limit:
            return raw
        return raw[: max(0, limit - 3)] + "..."

    def _serialize_plan_steps(self, steps: list[PlanStep] | None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for step in steps or []:
            if not isinstance(step, PlanStep):
                continue
            out.append(
                {
                    "step_id": step.step_id,
                    "action": step.action,
                    "args": step.args,
                    "depends_on": step.depends_on,
                    "expected_output": step.expected_output,
                    "status": step.status,
                }
            )
        return out

    def _tool_trace_from_payloads(
        self, payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            result = payload.get("result", {})
            reflection = payload.get("reflection", {})
            row = {
                "tool": str(payload.get("tool", "")).strip(),
                "args": payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {},
                "ok": bool(isinstance(result, dict) and result.get("ok", False)),
                "error": self._truncate_text(
                    str(result.get("error", "")) if isinstance(result, dict) else "",
                    240,
                ),
                "exit_code": result.get("exit_code") if isinstance(result, dict) else None,
                "tests_passed": bool(result.get("tests_passed", False))
                if isinstance(result, dict)
                else False,
                "passed": int(result.get("passed", 0) or 0)
                if isinstance(result, dict)
                else 0,
                "failed": int(result.get("failed", 0) or 0)
                if isinstance(result, dict)
                else 0,
                "retried": bool(
                    isinstance(reflection, dict) and reflection.get("retried", False)
                ),
                "confidence": (
                    float(reflection.get("confidence", 0.0) or 0.0)
                    if isinstance(reflection, dict)
                    else 0.0
                ),
            }
            if isinstance(result, dict):
                stdout = str(result.get("stdout", "")).strip()
                stderr = str(result.get("stderr", "")).strip()
                if stdout:
                    row["stdout_excerpt"] = self._truncate_text(stdout, 240)
                if stderr:
                    row["stderr_excerpt"] = self._truncate_text(stderr, 240)
            trace.append(row)
        return trace

    @staticmethod
    def _estimate_interaction_quality(
        final_text: str,
        tool_payloads: list[dict[str, Any]],
        *,
        success_hint: bool | None = None,
        score_hint: float | None = None,
    ) -> tuple[bool, float, int, int]:
        retry_count = 0
        error_count = 0
        ok_tools = 0
        for payload in tool_payloads:
            result = payload.get("result", {})
            reflection = payload.get("reflection", {})
            if isinstance(reflection, dict) and reflection.get("retried", False):
                retry_count += 1
            if isinstance(result, dict) and result.get("ok", False):
                ok_tools += 1
            else:
                error_count += 1

        if score_hint is not None:
            score = max(0.0, min(float(score_hint), 1.0))
        else:
            total_tools = len(tool_payloads)
            score = 0.2 if final_text.strip() else 0.05
            if total_tools:
                score += 0.45 * (ok_tools / max(1, total_tools))
            else:
                score += 0.25
            if retry_count == 0:
                score += 0.1
            if error_count == 0:
                score += 0.15
            lower = final_text.lower()
            if any(marker in lower for marker in ("error", "failed", "traceback", "exception")):
                score -= 0.2
            score = max(0.0, min(score, 1.0))

        success = bool(success_hint) if success_hint is not None else score >= 0.67
        return success, round(score, 3), retry_count, error_count

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_interaction_trajectory(
        self,
        *,
        source: str,
        goal: str,
        prompt: str,
        final_text: str,
        tool_payloads: list[dict[str, Any]],
        plan: list[PlanStep] | None = None,
        assistant_action: str = "",
        success_hint: bool | None = None,
        score_hint: float | None = None,
        validator: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.interaction_logging_enabled:
            return
        success, score, retry_count, error_count = self._estimate_interaction_quality(
            final_text,
            tool_payloads,
            success_hint=success_hint,
            score_hint=score_hint,
        )
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source or "chat_turn"),
            "goal": str(goal or "").strip(),
            "prompt": str(prompt or "").strip(),
            "plan": self._serialize_plan_steps(plan),
            "tool_calls": self._tool_trace_from_payloads(tool_payloads),
            "assistant_action": str(assistant_action or "").strip(),
            "assistant_final": str(final_text or "").strip(),
            "result": str(final_text or "").strip(),
            "success": success,
            "score": score,
            "retry_count": retry_count,
            "error_count": error_count,
        }
        if isinstance(validator, dict) and validator:
            payload["validator"] = validator
        if isinstance(metrics, dict) and metrics:
            payload["metrics"] = metrics
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        self._append_jsonl(self.interaction_log, payload)

    @staticmethod
    def _fallback_plan_steps(fallback_goal: str) -> list[PlanStep]:
        return [
            PlanStep(
                step_id=1,
                action="inspect_workspace",
                args={"path": ".", "max_entries": 200},
                depends_on=[],
                expected_output="workspace file list and project shape",
            ),
            PlanStep(
                step_id=2,
                action="implement_high_impact_change",
                args={"objective": fallback_goal},
                depends_on=[1],
                expected_output="concrete code or config updates aligned to goal",
            ),
            PlanStep(
                step_id=3,
                action="validate_result",
                args={"method": "tests_or_diff_review"},
                depends_on=[2],
                expected_output="validation evidence and any follow-up issues",
            ),
        ]

    @staticmethod
    def _coerce_structured_plan(payload: Any, fallback_goal: str) -> list[PlanStep]:
        items: list[Any] = []
        if isinstance(payload, dict):
            maybe = payload.get("steps")
            if isinstance(maybe, list):
                items = maybe
        elif isinstance(payload, list):
            items = payload

        out: list[PlanStep] = []
        seen_ids = set()
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                if isinstance(item, str) and item.strip():
                    action = item.strip()
                else:
                    continue
                step = PlanStep(
                    step_id=idx,
                    action=action,
                    args={},
                    depends_on=([idx - 1] if idx > 1 else []),
                    expected_output="completed step output",
                )
                out.append(step)
                seen_ids.add(step.step_id)
                continue

            raw_step_id = item.get("step_id", item.get("id", idx))
            try:
                step_id = int(raw_step_id)
            except Exception:
                step_id = idx
            if step_id <= 0 or step_id in seen_ids:
                step_id = idx if idx not in seen_ids else max(seen_ids or {0}) + 1
            seen_ids.add(step_id)

            step_type = str(item.get("type", "")).strip().lower()
            action = str(
                item.get("action", "")
                or item.get("tool", "")
                or item.get("step", "")
            ).strip()
            if step_type in {"tool_call", "tool"} and not action:
                action = str(item.get("tool", "")).strip()
            if not action:
                action = f"step_{step_id}"

            args = item.get("args", {})
            if not isinstance(args, dict):
                args = {}

            deps_raw = item.get("depends_on", [])
            deps: list[int] = []
            if isinstance(deps_raw, list):
                for d in deps_raw:
                    try:
                        di = int(d)
                    except Exception:
                        continue
                    if di > 0 and di != step_id and di not in deps:
                        deps.append(di)
            expected_output = str(
                item.get("expected_output", "") or item.get("expected", "")
            ).strip()

            out.append(
                PlanStep(
                    step_id=step_id,
                    action=action,
                    args=args,
                    depends_on=deps,
                    expected_output=expected_output,
                )
            )

        if not out:
            return ChatEngine._fallback_plan_steps(fallback_goal)
        return out

    @staticmethod
    def _validate_plan_steps(steps: list[PlanStep], fallback_goal: str) -> list[PlanStep]:
        if not steps:
            return ChatEngine._fallback_plan_steps(fallback_goal)

        normalized: list[PlanStep] = []
        used_ids: set[int] = set()
        for idx, step in enumerate(steps, start=1):
            try:
                raw_step_id = int(step.step_id)
            except Exception:
                raw_step_id = idx
            step_id = raw_step_id if raw_step_id > 0 else idx
            if step_id in used_ids:
                step_id = max(used_ids or {0}) + 1
            used_ids.add(step_id)
            normalized.append(
                PlanStep(
                    step_id=step_id,
                    action=step.action.strip() or f"step_{idx}",
                    args=step.args if isinstance(step.args, dict) else {},
                    depends_on=[],
                    expected_output=step.expected_output.strip(),
                    status=(
                        step.status
                        if step.status in {"pending", "in_progress", "done"}
                        else "pending"
                    ),
                )
            )

        valid_ids = {step.step_id for step in normalized}
        incoming: dict[int, list[int]] = {step.step_id: [] for step in normalized}
        originals = {step.step_id: idx for idx, step in enumerate(normalized)}
        for idx, step in enumerate(steps, start=1):
            normalized_step = normalized[idx - 1]
            deps: list[int] = []
            for dep in step.depends_on:
                try:
                    dep_id = int(dep)
                except Exception:
                    continue
                if dep_id <= 0 or dep_id == normalized_step.step_id:
                    continue
                if dep_id not in valid_ids or dep_id in deps:
                    continue
                deps.append(dep_id)
            incoming[normalized_step.step_id] = deps
            normalized_step.depends_on = deps

        ordered = ChatEngine._topological_sort_plan_steps(
            normalized, incoming=incoming, original_positions=originals
        )
        if not ordered:
            return ChatEngine._fallback_plan_steps(fallback_goal)
        return ordered

    @staticmethod
    def _topological_sort_plan_steps(
        steps: list[PlanStep],
        *,
        incoming: dict[int, list[int]] | None = None,
        original_positions: dict[int, int] | None = None,
    ) -> list[PlanStep]:
        if not steps:
            return []

        step_map = {step.step_id: step for step in steps}
        original_positions = original_positions or {
            step.step_id: idx for idx, step in enumerate(steps)
        }
        deps_map: dict[int, list[int]] = {}
        for step in steps:
            deps = incoming.get(step.step_id, list(step.depends_on)) if incoming else list(step.depends_on)
            deps_map[step.step_id] = [dep for dep in deps if dep in step_map and dep != step.step_id]

        outgoing: dict[int, list[int]] = {step.step_id: [] for step in steps}
        indegree: dict[int, int] = {step.step_id: 0 for step in steps}
        for step_id, deps in deps_map.items():
            indegree[step_id] = len(deps)
            for dep in deps:
                outgoing.setdefault(dep, []).append(step_id)

        ready = sorted(
            [step_id for step_id, degree in indegree.items() if degree == 0],
            key=lambda step_id: (
                original_positions.get(step_id, 0),
                step_id,
            ),
        )
        ordered_ids: list[int] = []
        while ready:
            current = ready.pop(0)
            ordered_ids.append(current)
            for child in sorted(
                outgoing.get(current, []),
                key=lambda step_id: (
                    original_positions.get(step_id, 0),
                    step_id,
                ),
            ):
                indegree[child] = max(0, indegree.get(child, 0) - 1)
                if indegree[child] == 0 and child not in ordered_ids and child not in ready:
                    ready.append(child)
            ready.sort(
                key=lambda step_id: (
                    original_positions.get(step_id, 0),
                    step_id,
                )
            )

        if len(ordered_ids) != len(steps):
            scheduled = set(ordered_ids)
            remaining = sorted(
                [step_id for step_id in step_map if step_id not in scheduled],
                key=lambda step_id: (
                    original_positions.get(step_id, 0),
                    step_id,
                ),
            )
            for step_id in remaining:
                deps_map[step_id] = [dep for dep in deps_map.get(step_id, []) if dep in scheduled]
                ordered_ids.append(step_id)
                scheduled.add(step_id)

        ordered_steps: list[PlanStep] = []
        scheduled: set[int] = set()
        for step_id in ordered_ids:
            source = step_map[step_id]
            deps = [dep for dep in deps_map.get(step_id, []) if dep in scheduled]
            ordered_steps.append(
                PlanStep(
                    step_id=source.step_id,
                    action=source.action,
                    args=source.args,
                    depends_on=deps,
                    expected_output=source.expected_output,
                    status=source.status,
                )
            )
            scheduled.add(step_id)
        return ordered_steps

    @staticmethod
    def _token_overlap_similarity(a: str, b: str) -> float:
        def _tokens(text: str) -> set[str]:
            return {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())}

        ta = _tokens(a)
        tb = _tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1.0, len(ta | tb))

    @staticmethod
    def _infer_strategy_context_from_objective(objective: str) -> dict[str, Any]:
        text = str(objective or "").lower()
        context: dict[str, Any] = {}
        if "pytest" in text or "test" in text:
            context["test_runner"] = "pytest"
            context["framework"] = "pytest"
        if "node" in text or "npm" in text or "jest" in text:
            context["language"] = "node"
        elif "python" in text or "pip" in text:
            context["language"] = "python"
        return context

    @staticmethod
    def _strategy_context_match_score(
        expected: dict[str, Any], runtime: dict[str, Any]
    ) -> float:
        if not expected:
            return 0.15
        total = 0
        matched = 0
        for key, value in expected.items():
            total += 1
            if str(runtime.get(key, "")).strip().lower() == str(value).strip().lower():
                matched += 1
        if total <= 0:
            return 0.15
        return matched / total

    def _plan_objective_steps(
        self, objective: str, step_cap: int | None = None
    ) -> list[PlanStep]:
        self._status("planning task steps…")
        cap = (
            max(3, min(int(step_cap), 20))
            if step_cap is not None
            else max(3, min(self.autonomous_plan_step_cap, 20))
        )
        runtime_context = self._infer_strategy_context_from_objective(objective)
        reused = self._reuse_strategy_steps(
            objective, step_cap=cap, runtime_context=runtime_context
        )
        if reused:
            return reused[:cap]
        planning_prompt = (
            "Break this objective into actionable execution steps with strict structure.\n"
            f"Objective: {objective}\n"
            "Return strict JSON only. Preferred shape:\n"
            "{\n"
            '  "steps":[\n'
            "    {\n"
            '      "id": 1,\n'
            '      "type": "tool_call",\n'
            '      "tool": "search_project",\n'
            '      "args": {"query":"..."},\n'
            '      "depends_on": [],\n'
            '      "expected": "what this step should produce"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Compatibility aliases accepted: step_id<->id, action<->tool, expected_output<->expected.\n"
            f"Rules: 1..{cap} steps, unique ids, depends_on references earlier step ids only."
        )
        try:
            raw = self.model.generate(
                [
                    {
                        "role": "system",
                        "content": "You are an execution planner. Return JSON only.",
                    },
                    {"role": "user", "content": planning_prompt},
                ]
            )
            payload = parse_json_payload(self._strip_thinking(raw))
            coerced = self._coerce_structured_plan(payload, fallback_goal=objective)
            steps = self._validate_plan_steps(coerced, fallback_goal=objective)
        except Exception:
            steps = self._fallback_plan_steps(fallback_goal=objective)
        return steps[:cap]

    def _create_task_state(self, objective: str, step_cap: int | None = None) -> TaskState:
        steps = self._plan_objective_steps(objective, step_cap=step_cap)
        return TaskState(
            goal=objective,
            steps=steps,
            current_step=0,
            completed_step_ids=set(),
            history=[],
        )

    def _strategy_memory_store(self) -> Any | None:
        return getattr(self.tools, "memory_store", None)

    def _reuse_strategy_steps(
        self,
        objective: str,
        step_cap: int | None = None,
        runtime_context: dict[str, Any] | None = None,
    ) -> list[PlanStep]:
        store = self._strategy_memory_store()
        if store is None or not hasattr(store, "find_strategies"):
            return []
        try:
            matches = store.find_strategies(query=objective, limit=5)
        except Exception:
            return []
        if not isinstance(matches, list) or not matches:
            return []
        ctx = runtime_context if isinstance(runtime_context, dict) else {}
        ranked: list[tuple[float, dict[str, Any]]] = []
        for match in matches:
            goal_similarity = self._token_overlap_similarity(
                objective, str(match.get("goal", ""))
            )
            context_score = self._strategy_context_match_score(
                match.get("context", {}) if isinstance(match.get("context"), dict) else {},
                ctx,
            )
            success_rate = float(match.get("success_rate", 0.0) or 0.0)
            base_score = float(match.get("score", 0.0) or 0.0)
            hybrid_score = (
                (base_score * 0.6)
                + (goal_similarity * 0.35)
                + (context_score * 0.2)
                + (success_rate * 0.25)
            )
            ranked.append((hybrid_score, match))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_score, top = ranked[0]
        if top_score < 0.7:
            return []
        steps = self._validate_plan_steps(
            self._coerce_structured_plan(top.get("strategy", []), fallback_goal=objective),
            fallback_goal=objective,
        )
        if step_cap is not None:
            steps = steps[: max(1, step_cap)]
        return steps

    def _record_strategy_outcome(
        self,
        objective: str,
        steps: list[PlanStep],
        success: bool,
        notes: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        store = self._strategy_memory_store()
        if store is None or not hasattr(store, "record_strategy"):
            return
        serialized = [
            {
                "step_id": step.step_id,
                "action": step.action,
                "args": step.args,
                "depends_on": step.depends_on,
                "expected_output": step.expected_output,
            }
            for step in steps
        ]
        try:
            store.record_strategy(
                goal=objective,
                strategy=serialized,
                success=success,
                source="autonomous",
                notes=notes[:400],
                context=context if isinstance(context, dict) else None,
            )
        except Exception:
            pass

    @staticmethod
    def _print_auto_metrics(auto_metrics: dict[str, Any]) -> None:
        steps = max(1, int(auto_metrics.get("steps", 0) or 0))
        failed_tools = int(auto_metrics.get("failed_tool_calls", 0) or 0)
        tool_calls = int(auto_metrics.get("tool_calls", 0) or 0)
        step_successes = int(auto_metrics.get("step_successes", 0) or 0)
        step_retries = int(auto_metrics.get("retries", 0) or 0)
        replans = int(auto_metrics.get("replans", 0) or 0)
        validator_failures = int(auto_metrics.get("validator_failures", 0) or 0)
        test_repairs = int(auto_metrics.get("test_repair_attempts", 0) or 0)
        success_rate = round(step_successes / steps, 3)
        tool_failure_rate = round(failed_tools / max(1, tool_calls), 3)
        print(
            "[auto] metrics: "
            f"steps={steps}, tools={tool_calls}, failed_tools={failed_tools}, "
            f"tool_failure_rate={tool_failure_rate}, success_rate={success_rate}, "
            f"retries={step_retries}, replans={replans}, "
            f"test_repairs={test_repairs}, "
            f"validator_failures={validator_failures}, "
            f"est_tokens_out~{int(auto_metrics.get('est_tokens_out', 0) or 0)}"
        )

    def _reflect_tool_result(
        self,
        user_message: str,
        call: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.tool_reflection_enabled:
            return {
                "enabled": False,
                "retry": False,
                "succeeded": bool(result.get("ok", False)),
                "reason": "reflection_disabled",
            }

        reflection_prompt = (
            "Review this tool execution and decide if it succeeded for the user intent.\n"
            "Return strict JSON only with keys:\n"
            "status (success|partial|failed), succeeded (bool), confidence (0..1), issues (array), retry (bool), reason (str), revised_call (object|null).\n"
            "If retry=true and revised_call is provided, format revised_call as:\n"
            '{"tool":"tool_name","args":{...}}\n'
            f"User request: {user_message[:1200]}\n"
            f"Tool call: {json.dumps({'tool': call.get('name', ''), 'args': call.get('args', {})}, ensure_ascii=False)[:3000]}\n"
            f"Tool result: {json.dumps(result, ensure_ascii=False)[:5000]}"
        )
        default = {
            "enabled": True,
            "status": "success" if bool(result.get("ok", False)) else "failed",
            "retry": False,
            "succeeded": bool(result.get("ok", False)),
            "confidence": 0.7 if bool(result.get("ok", False)) else 0.25,
            "issues": [],
            "reason": "",
            "revised_call": None,
        }
        try:
            raw = self.model.generate(
                [
                    {
                        "role": "system",
                        "content": "You are a tool-call supervisor. Return JSON only.",
                    },
                    {"role": "user", "content": reflection_prompt},
                ]
            )
            payload = parse_json_payload(self._strip_thinking(raw))
            if not isinstance(payload, dict):
                return default

            out = dict(default)
            status = str(payload.get("status", out["status"])).strip().lower()
            if status not in {"success", "partial", "failed"}:
                status = out["status"]
            out["status"] = status
            out["succeeded"] = bool(payload.get("succeeded", out["succeeded"]))
            out["retry"] = bool(payload.get("retry", False))
            out["reason"] = str(payload.get("reason", "")).strip()[:400]
            try:
                conf = float(payload.get("confidence", out["confidence"]))
            except Exception:
                conf = float(out["confidence"])
            out["confidence"] = max(0.0, min(conf, 1.0))
            raw_issues = payload.get("issues", [])
            issues: list[str] = []
            if isinstance(raw_issues, list):
                for issue in raw_issues:
                    s = str(issue).strip()
                    if s:
                        issues.append(s[:200])
            out["issues"] = issues
            revised = payload.get("revised_call")
            if isinstance(revised, dict):
                parsed = parse_tool_calls(json.dumps(revised, ensure_ascii=False))
                if parsed:
                    out["revised_call"] = parsed[0]
            if out["retry"] and out["revised_call"] is None:
                out["revised_call"] = {"name": call.get("name", ""), "args": call.get("args", {})}
            return out
        except Exception:
            return default

    def _run_tool_call_with_reflection(
        self,
        user_message: str,
        call: dict[str, Any],
    ) -> dict[str, Any]:
        initial_name = str(call.get("name", ""))
        initial_args = call.get("args", {})
        if not isinstance(initial_args, dict):
            initial_args = {}

        result = self._execute_tool_call_with_policy(user_message, {"name": initial_name, "args": initial_args})
        reflection = self._reflect_tool_result(
            user_message=user_message,
            call={"name": initial_name, "args": initial_args},
            result=result if isinstance(result, dict) else {"ok": False, "error": "invalid tool result"},
        )
        if (
            not reflection.get("retry", False)
            and not reflection.get("succeeded", False)
            and float(reflection.get("confidence", 0.0) or 0.0) < 0.45
        ):
            reflection["retry"] = True
            if reflection.get("revised_call") is None:
                reflection["revised_call"] = {"name": initial_name, "args": initial_args}
            reason = str(reflection.get("reason", "")).strip()
            reflection["reason"] = (
                f"{reason} | auto-retry on low-confidence failure".strip(" |")
            )
        executed_name = initial_name
        executed_args = initial_args

        if reflection.get("retry"):
            retry_call = reflection.get("revised_call")
            suggested_name = ""
            suggested_args = initial_args
            if isinstance(retry_call, dict):
                suggested_name = str(retry_call.get("name", "")).strip()
                suggested_args = retry_call.get("args", {})
                if not isinstance(suggested_args, dict):
                    suggested_args = initial_args
            retry_args = suggested_args
            # Keep retries on the same tool to avoid confusing start/result mismatches
            # and prevent reflection from jumping to hallucinated tool names.
            if suggested_name and suggested_name != initial_name:
                reflection["retry_name_ignored"] = suggested_name
                # If the reflected call switches tools, ignore its args as well.
                # Mixing list_files args into read_file (or vice versa) causes
                # path-type errors like "not a file"/"not a directory".
                retry_args = initial_args
            retry_name = initial_name
            retry_result = self._execute_tool_call_with_policy(
                user_message, {"name": retry_name, "args": retry_args}
            )
            reflection["retried"] = True
            reflection["retry_tool"] = retry_name
            reflection["retry_ok"] = bool(
                isinstance(retry_result, dict) and retry_result.get("ok", False)
            )
            result = retry_result
            executed_name = retry_name
            executed_args = retry_args
        else:
            reflection["retried"] = False

        # Cross-session learning hooks: feed outcomes back into memory/skill stats.
        try:
            conf = float(reflection.get("confidence", 0.5) or 0.5)
        except Exception:
            conf = 0.5
        succeeded = bool(reflection.get("succeeded", False))
        if isinstance(result, dict):
            if executed_name in {"find_in_memory", "search_memory"} and result.get("ok", False):
                matches = result.get("matches", [])
                if isinstance(matches, list):
                    for m in matches[:3]:
                        if not isinstance(m, dict):
                            continue
                        block_name = str(m.get("block", "") or m.get("name", "")).strip()
                        if not block_name:
                            continue
                        self.tools.execute(
                            "record_memory_feedback",
                            {
                                "block_name": block_name,
                                "success": succeeded,
                                "confidence": conf,
                                "source": "tool_reflection",
                            },
                        )
            if executed_name == "run_function":
                skill_name = ""
                if isinstance(executed_args, dict):
                    skill_name = str(executed_args.get("name", "")).strip()
                if skill_name:
                    self.tools.execute(
                        "record_skill_outcome",
                        {
                            "name": skill_name,
                            "success": succeeded,
                            "confidence": conf,
                            "notes": str(reflection.get("reason", ""))[:200],
                        },
                    )

        return {
            "name": executed_name,
            "args": executed_args,
            "result": result,
            "reflection": reflection,
            "initial_name": initial_name,
            "initial_args": initial_args,
        }

    def _reflect_autonomous_progress(
        self,
        state: TaskState,
        step_text: str,
        final_text: str,
    ) -> dict[str, Any]:
        prompt = (
            "You are supervising an autonomous coding loop.\n"
            "Return strict JSON with keys:\n"
            "next_action (advance|retry|replan|done|bored), reason (string), confidence (0..1), issues (array), new_steps (array optional).\n"
            f"Goal: {state.goal}\n"
            f"Current step: {step_text}\n"
            f"Latest assistant output: {final_text[:2500]}"
        )
        fallback = {
            "next_action": "advance",
            "reason": "",
            "confidence": 0.5,
            "issues": [],
            "new_steps": [],
        }
        try:
            raw = self.model.generate(
                [
                    {
                        "role": "system",
                        "content": "You evaluate agent progress. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            payload = parse_json_payload(self._strip_thinking(raw))
            if not isinstance(payload, dict):
                return fallback
            action = str(payload.get("next_action", "advance")).strip().lower()
            if action not in {"advance", "retry", "replan", "done", "bored"}:
                action = "advance"
            try:
                confidence = float(payload.get("confidence", 0.5))
            except Exception:
                confidence = 0.5
            confidence = max(0.0, min(confidence, 1.0))
            raw_issues = payload.get("issues", [])
            issues: list[str] = []
            if isinstance(raw_issues, list):
                for issue in raw_issues:
                    s = str(issue).strip()
                    if s:
                        issues.append(s[:200])
            raw_new_steps = payload.get("new_steps", None)
            new_steps: list[PlanStep] = []
            if raw_new_steps is not None:
                new_steps = self._validate_plan_steps(
                    self._coerce_structured_plan(raw_new_steps, state.goal),
                    fallback_goal=state.goal,
                )
            return {
                "next_action": action,
                "reason": str(payload.get("reason", "")).strip()[:300],
                "confidence": confidence,
                "issues": issues,
                "new_steps": new_steps,
            }
        except Exception:
            return fallback

    @staticmethod
    def _continuation_poke(
        user_message: str,
        prefer_copyable_function: bool = False,
        require_file_tools: bool = False,
    ) -> dict[str, str]:
        copyable_hint = (
            "\nFor this request, output a copyable function for the user in final text. "
            "Use create_function only when the user explicitly asks to save/store it."
            if prefer_copyable_function
            else ""
        )
        file_hint = (
            "\nThis is a workspace-edit task. Apply changes via create_file/edit_file tools, not code-only explanation."
            if require_file_tools
            else ""
        )
        return {
            "role": "system",
            "content": (
                "Internal control message.\n"
                f"Original user request: {user_message}\n"
                "Tool execution is complete. Continue generation using the latest tool result for the original user request.\n"
                "If another tool is needed, return JSON tool call only.\n"
                "Otherwise return final user-facing answer text only.\n"
                f"{copyable_hint}{file_hint}\n"
                "Do not discuss this control message."
            ),
        }

    def _explicit_store_request(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("store_request", False))

    def _prefer_copyable_function_reply(self, user_message: str) -> bool:
        return self._requires_presearch_for_code(
            user_message
        ) and not self._explicit_store_request(user_message)

    def _execute_tool_call_with_policy(
        self, user_message: str, call: dict[str, Any]
    ) -> dict[str, Any]:
        name = call["name"]
        args = call.get("args", {})
        if name == "create_function" and self._prefer_copyable_function_reply(
            user_message
        ):
            code = ""
            if isinstance(args, dict):
                raw_code = args.get("code", "")
                if isinstance(raw_code, str):
                    code = raw_code.strip()
            return {
                "ok": True,
                "skipped": True,
                "policy": "copyable_function",
                "message": "create_function skipped because user asked for copyable function output.",
                "code": code,
            }
        return self.tools.execute(name, args)

    @staticmethod
    def _extract_keywords(text: str, limit: int = 6) -> list[str]:
        stop = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "this",
            "that",
            "what",
            "when",
            "where",
            "who",
            "how",
            "want",
            "learn",
            "more",
            "about",
            "please",
            "give",
            "some",
            "example",
            "examples",
            "to",
            "in",
            "on",
            "at",
            "of",
        }
        words = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
        out: list[str] = []
        seen = set()
        for w in words:
            if w in stop:
                continue
            if w not in seen:
                out.append(w)
                seen.add(w)
            if len(out) >= limit:
                break
        return out or ["general"]

    def _looks_coding_request(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("coding", False))

    def _looks_smalltalk(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("smalltalk", False))

    def _looks_creative_request(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("creative", False))

    def _looks_personal_or_companion_chat(self, text: str) -> bool:
        return bool(self._intent_flags(text).get("companion", False))

    def _heuristic_intent_flags(self, user_message: str) -> dict[str, Any]:
        t = user_message.lower()
        compact = re.sub(r"\s+", " ", user_message.strip().lower())

        coding = any(
            m in t
            for m in (
                "python",
                "javascript",
                "typescript",
                "java",
                "go ",
                "rust",
                "code",
                "class",
                "function",
                "exception",
                "stack trace",
                "debug",
                "bug",
                "test",
                "api",
                "sql",
            )
        )
        smalltalk_exact = {
            "hi",
            "hello",
            "hey",
            "yo",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "nice",
            "cool",
            "let's code",
            "lets code",
        }
        smalltalk = compact in smalltalk_exact or (
            len(compact.split()) <= 2
            and all(
                w in {"hi", "hello", "hey", "thanks", "ok", "okay"}
                for w in compact.split()
            )
        )
        creative = any(
            m in t
            for m in (
                "write a poem",
                "poem",
                "story",
                "joke",
                "translate",
                "rewrite this",
                "paraphrase",
            )
        )
        companion = any(
            m in t
            for m in (
                "be my partner",
                "be my girlfriend",
                "be my boyfriend",
                "chat with me",
                "talk with me",
                "keep me company",
                "can you stay with me",
                "can you be with me",
            )
        )
        store_request = any(
            m in t
            for m in (
                "save as function",
                "store as function",
                "register function",
                "persist function",
                "create_function",
                "save this function",
                "store this function",
                "add to functions",
                "put in functions",
                "save reusable function",
                "store reusable function",
            )
        )
        code_generation = any(
            m in t
            for m in (
                "create function",
                "write function",
                "my own function",
                "their own function",
                "create your own function",
                "custom function",
                "implement function",
                "build function",
                "create a function",
                "write code",
                "implement code",
                "download file",
                "read research",
                "parse paper",
            )
        )
        workspace_edit = any(
            m in t
            for m in (
                "read file",
                "read files",
                "search project",
                "find in project",
                "grep",
                "open file",
                "edit file",
                "update file",
                "modify file",
                "refactor",
                "fix bug",
                "fix this",
                "in this project",
                "in this repo",
                "codebase",
                "make plan",
                "todo",
                "to-do",
            )
        ) or bool(self._extract_explicit_file_paths(user_message))
        factual = any(
            m in t
            for m in (
                "what",
                "who",
                "when",
                "where",
                "which",
                "why",
                "latest",
                "current",
                "today",
                "news",
                "learn more",
                "about",
                "overview",
                "explain",
                "recommend",
                "best",
            )
        ) or ("?" in t and len(t.split()) >= 8)
        if coding or smalltalk or creative or companion:
            factual = False

        return {
            "coding": coding,
            "smalltalk": smalltalk,
            "creative": creative,
            "companion": companion,
            "factual": factual,
            "workspace_edit": workspace_edit,
            "code_generation": code_generation,
            "store_request": store_request,
            "optimized_query": "",
        }

    def _ai_intent_flags(self, user_message: str) -> dict[str, Any]:
        prompt = (
            "Classify user intent and optimize search query.\n"
            "Return JSON only with keys:\n"
            "coding, smalltalk, creative, companion, factual, workspace_edit, code_generation, store_request, optimized_query.\n"
            "Booleans for all flags. optimized_query should be concise and high-signal for web search.\n"
            "If no web search is needed, optimized_query can be empty.\n"
            f"User: {user_message}"
        )
        try:
            raw = self.model.generate(
                [
                    {
                        "role": "system",
                        "content": "You are an intent classifier. Return strict JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            clean = self._strip_thinking(raw)
            payload = parse_json_payload(clean)
            if not isinstance(payload, dict):
                return {}
            out: dict[str, Any] = {}
            for k in (
                "coding",
                "smalltalk",
                "creative",
                "companion",
                "factual",
                "workspace_edit",
                "code_generation",
                "store_request",
            ):
                if k in payload:
                    out[k] = bool(payload[k])
            q = payload.get("optimized_query", "")
            if isinstance(q, str):
                out["optimized_query"] = q.strip()
            return out
        except Exception:
            return {}

    def _intent_flags(self, user_message: str) -> dict[str, Any]:
        key = user_message.strip()
        if key in self._intent_cache:
            return self._intent_cache[key]

        heuristic = self._heuristic_intent_flags(user_message)
        self._status("classifying intent…")
        ai = self._ai_intent_flags(user_message)
        merged = dict(heuristic)
        for k, v in ai.items():
            if k == "optimized_query":
                if isinstance(v, str) and v:
                    merged[k] = v
            elif isinstance(v, bool):
                merged[k] = v
        self._intent_cache[key] = merged
        return merged

    def _requires_web_presearch_for_factual(self, user_message: str) -> bool:
        flags = self._intent_flags(user_message)
        if flags.get("workspace_edit", False):
            return False
        return bool(flags.get("factual", False))

    @staticmethod
    def _is_time_sensitive_factual(user_message: str) -> bool:
        t = user_message.lower()
        markers = (
            "latest",
            "newest",
            "current",
            "today",
            "this year",
            "now",
            "recent",
            "2024",
            "2025",
            "2026",
            "trend",
            "release",
            "price",
            "news",
        )
        return any(m in t for m in markers)

    def _ensure_datetime_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        datetime_executed: bool,
    ) -> list[dict[str, Any]]:
        factual_tools = {
            "get_current_datetime",
            "find_in_memory",
            "search_memory",
            "search_web",
        }
        # Respect explicit model tool selection for this turn.
        if any(str(call.get("name", "")) not in factual_tools for call in tool_calls):
            return tool_calls
        if not self._requires_web_presearch_for_factual(user_message):
            return tool_calls
        if datetime_executed:
            return tool_calls
        if any(call.get("name") == "get_current_datetime" for call in tool_calls):
            return tool_calls
        if not self._is_time_sensitive_factual(user_message):
            return tool_calls
        return [{"name": "get_current_datetime", "args": {}}] + tool_calls

    def _presearch_tool_calls_for_factual(
        self, user_message: str
    ) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        query = self._optimize_search_query(user_message)
        calls = [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_memory", "args": {"query": query, "limit": 5}},
            {"name": "search_web", "args": {"query": query, "level": "auto"}},
        ]
        if self._is_time_sensitive_factual(user_message):
            calls.insert(0, {"name": "get_current_datetime", "args": {}})
        return calls

    def _optimize_search_query(self, user_message: str) -> str:
        inferred = self._intent_flags(user_message).get("optimized_query", "")
        if isinstance(inferred, str) and inferred.strip():
            return inferred.strip()

        raw = re.sub(r"\s+", " ", user_message.strip())
        q = raw.lower()

        # Remove low-signal conversational wrappers.
        wrappers = (
            "i want to learn more about",
            "i want to know about",
            "tell me about",
            "can you tell me about",
            "help me understand",
            "i want to learn about",
        )
        for w in wrappers:
            if q.startswith(w):
                q = q[len(w) :].strip(" ,.-")
                break

        # Topic-specific normalization for better retrieval quality.
        if any(k in q for k in ("sex", "sexual", "intimacy", "partner")):
            return (
                "how to improve sexual intimacy and communication with partner consent"
            )

        # Generic fallback: compact keywords from original user text.
        keys = re.findall(r"[a-zA-Z0-9_]{3,}", q)
        stop = {
            "want",
            "learn",
            "more",
            "about",
            "please",
            "help",
            "tell",
            "know",
            "with",
            "from",
            "this",
            "that",
        }
        compact = [k for k in keys if k not in stop]
        if compact:
            return " ".join(compact[:10])
        return raw

    @staticmethod
    def _extract_explicit_file_paths(text: str, limit: int = 3) -> list[str]:
        # Capture lightweight path mentions such as src/app.py, README.md, or
        # workspaces/crypto_research so preinspection can inspect the right kind of path.
        patterns = (
            re.compile(r"([A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+/?)"),
            re.compile(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9_]{1,8})"),
        )
        out: list[str] = []
        seen = set()
        for pattern in patterns:
            for match in pattern.findall(text):
                candidate = str(match).strip().strip(".,;:()[]{}\"'")
                if not candidate or candidate.startswith(("http://", "https://")):
                    continue
                if candidate.endswith("/") and candidate != "/":
                    candidate = candidate.rstrip("/")
                if candidate not in seen:
                    out.append(candidate)
                    seen.add(candidate)
                if len(out) >= limit:
                    return out
        return out

    def _preinspect_call_for_explicit_path(self, path: str) -> dict[str, Any]:
        workspace_tools = getattr(self.tools, "workspace_tools", None)
        if workspace_tools is None:
            return {"name": "read_file", "args": {"path": path, "max_chars": 6000}}

        try:
            resolved = workspace_tools._resolve(path)
        except Exception:
            resolved = None

        if resolved is not None and resolved.exists():
            if resolved.is_dir():
                return {"name": "list_files", "args": {"path": path, "max_entries": 100}}
            if resolved.is_file():
                return {"name": "read_file", "args": {"path": path, "max_chars": 6000}}

        if "." not in Path(path).name:
            return {"name": "list_files", "args": {"path": path, "max_entries": 100}}
        return {"name": "read_file", "args": {"path": path, "max_chars": 6000}}

    @staticmethod
    def _extract_symbol_hints(text: str, limit: int = 3) -> list[str]:
        patterns = (
            r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
            r"\bsymbol\s+([A-Za-z_][A-Za-z0-9_]*)",
        )
        out: list[str] = []
        seen = set()
        for pat in patterns:
            for match in re.findall(pat, text, flags=re.IGNORECASE):
                symbol = str(match).strip()
                if not symbol:
                    continue
                key = symbol.lower()
                if key in seen:
                    continue
                out.append(symbol)
                seen.add(key)
                if len(out) >= limit:
                    return out
        return out

    def _requires_workspace_preinspect(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("workspace_edit", False))

    def _preinspect_tool_calls_for_workspace(
        self, user_message: str
    ) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = [
            {"name": "list_files", "args": {"path": ".", "max_entries": 200}},
            {"name": "detect_project_context", "args": {"path": "."}},
            {
                "name": "index_symbols",
                "args": {"path": ".", "max_files": 250, "max_symbols": 4000},
            },
        ]
        keyword_query = " ".join(self._extract_keywords(user_message, limit=5))
        if keyword_query:
            calls.append(
                {
                    "name": "search_project",
                    "args": {"query": keyword_query, "path": ".", "max_matches": 60},
                }
            )
        for symbol in self._extract_symbol_hints(user_message):
            calls.append(
                {
                    "name": "lookup_symbol",
                    "args": {"symbol": symbol, "path": ".", "max_results": 20},
                }
            )
        for p in self._extract_explicit_file_paths(user_message):
            calls.append(self._preinspect_call_for_explicit_path(p))
        return calls

    def _ensure_web_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        web_search_executed: bool,
    ) -> list[dict[str, Any]]:
        factual_tools = {
            "get_current_datetime",
            "find_in_memory",
            "search_memory",
            "search_web",
        }
        # Respect explicit model tool selection for this turn.
        if any(str(call.get("name", "")) not in factual_tools for call in tool_calls):
            return tool_calls
        if web_search_executed or not self._requires_web_presearch_for_factual(
            user_message
        ):
            return tool_calls
        if any(call.get("name") == "search_web" for call in tool_calls):
            return tool_calls
        return tool_calls + [
            {
                "name": "search_web",
                "args": {
                    "query": self._optimize_search_query(user_message),
                    "level": "auto",
                },
            }
        ]

    def _ensure_memory_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        memory_checked: bool,
    ) -> list[dict[str, Any]]:
        factual_tools = {
            "get_current_datetime",
            "find_in_memory",
            "search_memory",
            "search_web",
        }
        # Respect explicit model tool selection for this turn.
        if any(str(call.get("name", "")) not in factual_tools for call in tool_calls):
            return tool_calls
        if memory_checked or not self._requires_web_presearch_for_factual(user_message):
            return tool_calls
        if any(
            call.get("name") in {"find_in_memory", "search_memory"} for call in tool_calls
        ):
            return tool_calls
        mem_call = {
            "name": "find_in_memory",
            "args": {"keywords": self._extract_keywords(user_message)},
        }
        if tool_calls and tool_calls[0].get("name") == "get_current_datetime":
            return [tool_calls[0], mem_call] + tool_calls[1:]
        return [mem_call] + tool_calls

    def _emergency_tool_calls(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        query = self._optimize_search_query(user_message)
        calls: list[dict[str, Any]] = [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_memory", "args": {"query": query, "limit": 4}},
        ]
        if not self._looks_coding_request(user_message):
            calls.append(
                {"name": "search_web", "args": {"query": user_message, "level": "auto"}}
            )
        return calls

    def _requires_presearch_for_code(self, user_message: str) -> bool:
        return bool(self._intent_flags(user_message).get("code_generation", False))

    def _presearch_tool_calls_for_code(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        normalized = re.sub(r"\s+", " ", user_message).strip().strip("'\"`")
        normalized = re.sub(r"[\"'`]+", "", normalized)
        research_query = f"how to {normalized} in python"
        return [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_memory", "args": {"query": normalized, "limit": 5}},
            {"name": "search_web", "args": {"query": research_query, "level": "deep"}},
        ]

    def _fallback_answer_from_tools(self) -> str | None:
        for msg in reversed(self.history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            try:
                payload = json.loads(content)
            except Exception:
                continue

            tool_name = payload.get("tool")
            result = payload.get("result", {})
            if tool_name == "search_web" and isinstance(result, dict):
                search = result.get("search", {})
                results = search.get("results", []) if isinstance(search, dict) else []
                if not results:
                    continue
                lines = ["I fetched web sources. Top results:"]
                for i, item in enumerate(results[:5], start=1):
                    title = str(item.get("title", "")).strip()
                    url = str(item.get("url", "")).strip()
                    excerpt = (
                        str(item.get("page_excerpt", "")).strip()
                        or str(item.get("snippet", "")).strip()
                    )
                    if excerpt and len(excerpt) > 220:
                        excerpt = excerpt[:217] + "..."
                    lines.append(f"{i}. {title} ({url})")
                    if excerpt:
                        lines.append(f"   {excerpt}")
                return "\n".join(lines)

            if tool_name == "search_project" and isinstance(result, dict):
                matches = result.get("matches", [])
                if not isinstance(matches, list) or not matches:
                    continue
                lines = ["I searched the project and found matches:"]
                for i, m in enumerate(matches[:8], start=1):
                    path = str(m.get("path", "")).strip()
                    line = m.get("line", "")
                    text = str(m.get("text", "")).strip()
                    if len(text) > 160:
                        text = text[:157] + "..."
                    lines.append(f"{i}. {path}:{line} {text}")
                return "\n".join(lines)

            if tool_name == "find_in_memory" and isinstance(result, dict):
                matches = result.get("matches", [])
                if not matches:
                    continue
                top = matches[0]
                name = str(top.get("name", "memory"))
                topic = str(top.get("topic", ""))
                knowledge = str(top.get("knowledge", "")).strip()
                if len(knowledge) > 450:
                    knowledge = knowledge[:447] + "..."
                return f"I found relevant memory block `{name}` ({topic}).\n{knowledge}"

            if tool_name == "search_memory" and isinstance(result, dict):
                matches = result.get("matches", [])
                if not isinstance(matches, list) or not matches:
                    continue
                lines = ["I found semantically similar memory:"]
                for i, m in enumerate(matches[:5], start=1):
                    name = str(m.get("name", "memory")).strip()
                    topic = str(m.get("topic", "")).strip()
                    score = m.get("score", "")
                    lines.append(f"{i}. {name} ({topic}) score={score}")
                return "\n".join(lines)

            if tool_name == "lookup_symbol" and isinstance(result, dict):
                matches = result.get("matches", [])
                if not isinstance(matches, list) or not matches:
                    continue
                lines = ["I found matching symbols:"]
                for i, m in enumerate(matches[:8], start=1):
                    name = str(m.get("name", "")).strip()
                    sym_type = str(m.get("type", "")).strip()
                    path = str(m.get("path", "")).strip()
                    line = m.get("line", "")
                    lines.append(f"{i}. {sym_type} {name} at {path}:{line}")
                return "\n".join(lines)

            if tool_name == "index_symbols" and isinstance(result, dict):
                count = int(result.get("count", 0) or 0)
                indexed_files = int(result.get("indexed_files", 0) or 0)
                if count > 0:
                    return f"I indexed {count} symbols across {indexed_files} files."

            if tool_name == "summarize_file" and isinstance(result, dict):
                summary = str(result.get("summary", "")).strip()
                if summary:
                    return summary

            if tool_name == "detect_project_context" and isinstance(result, dict):
                framework = str(result.get("framework", "")).strip() or "unknown"
                runner = str(result.get("test_runner", "")).strip() or "unknown"
                entry = result.get("entry_points", [])
                if isinstance(entry, list) and entry:
                    return (
                        f"Detected project context: framework={framework}, "
                        f"test_runner={runner}, entry={entry[0]}"
                    )
                return f"Detected project context: framework={framework}, test_runner={runner}"

            if tool_name == "run_tests" and isinstance(result, dict):
                ok = bool(result.get("ok", False))
                passed = int(result.get("passed", 0) or 0)
                failed = int(result.get("failed", 0) or 0)
                command = str(result.get("command", "")).strip()
                return (
                    f"Test run {'passed' if ok else 'failed'} "
                    f"(passed={passed}, failed={failed}) using `{command}`."
                )

            if tool_name == "validate_workspace_changes" and isinstance(result, dict):
                ok = bool(result.get("ok", False))
                changed = int(result.get("changed_file_count", 0) or 0)
                return (
                    f"Workspace validation {'passed' if ok else 'failed'} "
                    f"with {changed} changed file(s)."
                )

            if tool_name == "get_current_datetime" and isinstance(result, dict):
                dt = result.get("datetime", {})
                if isinstance(dt, dict):
                    date = str(dt.get("date", "")).strip()
                    time = str(dt.get("time", "")).strip()
                    tz = str(dt.get("timezone", "")).strip()
                    if date:
                        return f"Current runtime date/time: {date} {time} {tz}".strip()

            if tool_name == "create_function" and isinstance(result, dict):
                if result.get("policy") == "copyable_function":
                    code = str(result.get("code", "")).strip()
                    if code:
                        return (
                            "Here is the function you can copy and use:\n```python\n"
                            + code
                            + "\n```"
                        )
        return None

    def _messages(self) -> list[dict[str, str]]:
        self._compact_context_if_needed()
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self.compacted_context_note:
            msgs.append({"role": "system", "content": self.compacted_context_note})
        msgs.extend(self.history[-self.max_history :])
        return msgs

    def _compact_context_if_needed(self) -> None:
        if not self.compact_context_enabled:
            return
        if not self.history:
            return
        total_chars = sum(len(m.get("content", "")) for m in self.history)
        if (
            len(self.history) <= (self.max_history * 2)
            and total_chars <= self.max_context_chars
        ):
            return

        keep_tail = max(self.max_history, 12)
        if len(self.history) <= keep_tail + 4:
            return

        # Flush any unsaved messages BEFORE we discard the head of history,
        # so they are persisted to the session log even if compacted away.
        self._flush_to_session_log()

        head = self.history[:-keep_tail]
        tail = self.history[-keep_tail:]

        summary_lines = ["Compacted conversation context (older turns):"]
        for msg in head[-120:]:
            role = msg.get("role", "unknown")
            content = self._strip_thinking(msg.get("content", ""))
            content = re.sub(r"\s+", " ", content).strip()
            if len(content) > 220:
                content = content[:217] + "..."
            if not content:
                continue
            summary_lines.append(f"- {role}: {content}")
        summary = "\n".join(summary_lines)
        if len(summary) > 18000:
            summary = summary[:17997] + "..."

        self.compacted_context_note = summary
        self.history = tail
        # After truncation the absolute index is no longer valid — reset it to
        # the new list length (everything older was already flushed above).
        self._session_save_ptr = len(tail)

    def handle_turn(
        self,
        user_message: str,
        on_tool: Optional[Callable[[str, Dict[str, Any], Any], None]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        enforce_presearch: bool = True,
        log_interaction: bool = True,
    ) -> str:
        # Keep a single orchestration path by delegating to streaming engine.
        # We still expose the optional on_chunk callback for CLI/debug consumers.
        chunk_collector: list[str] = []

        def _collect_chunk(chunk: str) -> None:
            chunk_collector.append(chunk)
            if on_chunk:
                on_chunk(chunk)

        return self.handle_turn_stream(
            user_message=user_message,
            on_chunk=_collect_chunk,
            on_tool=on_tool,
            on_tool_start=on_tool_start,
            on_tool_phase=None,
            enforce_presearch=enforce_presearch,
            log_interaction=log_interaction,
        )

    def handle_turn_stream(
        self,
        user_message: str,
        on_chunk: Callable[[str], None],
        on_tool: (
            Callable[[str, dict[str, object], dict[str, object]], None] | None
        ) = None,
        on_tool_start: Callable[[str, dict[str, object]], None] | None = None,
        on_tool_phase: Callable[[], None] | None = None,
        enforce_presearch: bool = True,
        log_interaction: bool = True,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False
        web_search_executed_this_turn = False
        datetime_executed_this_turn = False
        memory_checked_this_turn = False
        turn_tool_payloads: list[dict[str, Any]] = []
        assistant_action_text = ""

        for _ in range(self.max_tool_rounds):
            assistant_text = ""
            in_think_stream = False
            pending_nonthink_chunks: list[str] = []

            messages = self._messages()
            if continue_after_tools:
                messages = messages + [
                    self._continuation_poke(
                        user_message,
                        prefer_copyable_function=self._prefer_copyable_function_reply(
                            user_message
                        ),
                        require_file_tools=self._requires_workspace_preinspect(
                            user_message
                        ),
                    )
                ]

            for chunk in self.model.stream_generate(messages):
                if not chunk:
                    continue

                assistant_text += chunk

                if chunk == "<think>":
                    in_think_stream = True
                    on_chunk(chunk)
                    continue
                if chunk == "</think>":
                    in_think_stream = False
                    on_chunk(chunk)
                    continue

                if in_think_stream:
                    on_chunk(chunk)
                else:
                    pending_nonthink_chunks.append(chunk)

            tool_calls = parse_tool_calls(assistant_text)
            if not tool_calls and self._contains_tool_denial(
                self._strip_thinking(assistant_text)
            ):
                self._log_supervision(
                    "tool_denial_detected", user_message, assistant_text
                )
                tool_calls = self._recover_tool_calls(
                    user_message=user_message, raw_assistant_text=assistant_text
                )
            if enforce_presearch:
                tool_calls = self._ensure_datetime_call_for_factual(
                    user_message=user_message,
                    tool_calls=tool_calls,
                    datetime_executed=datetime_executed_this_turn,
                )
                tool_calls = self._ensure_memory_call_for_factual(
                    user_message=user_message,
                    tool_calls=tool_calls,
                    memory_checked=memory_checked_this_turn,
                )
                tool_calls = self._ensure_web_call_for_factual(
                    user_message=user_message,
                    tool_calls=tool_calls,
                    web_search_executed=web_search_executed_this_turn,
                )
            if not tool_calls:
                pending_nonthink = "".join(pending_nonthink_chunks)
                clean = self._strip_thinking(assistant_text)
                if clean and self._contains_internal_prompt_echo(clean):
                    clean = ""
                if not clean:
                    recovered_text, recovered_calls, recovered_clean = (
                        self._recover_action_or_answer(user_message, assistant_text)
                    )
                    if recovered_calls:
                        assistant_text = recovered_text
                        tool_calls = recovered_calls
                    elif recovered_clean:
                        assistant_text = recovered_text
                        clean = recovered_clean
                    else:
                        recovered = self._recover_final_answer(assistant_text)
                        if recovered:
                            assistant_text, clean = recovered
                        else:
                            assistant_text = ""
                            clean = ""
                if not tool_calls and not clean and not emergency_tools_used:
                    emergency = self._emergency_tool_calls(user_message)
                    if emergency:
                        emergency_tools_used = True
                        tool_calls = emergency
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})}
                                    for call in emergency
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision(
                            "emergency_tool_calls", user_message, assistant_text
                        )
                if not tool_calls:
                    if not clean:
                        fallback_from_tools = self._fallback_answer_from_tools()
                        if fallback_from_tools:
                            assistant_text = fallback_from_tools
                            clean = assistant_text
                        else:
                            assistant_text = (
                                "I could not generate a final answer text from the model output. "
                                "Please try again or use a model/config that emits final content."
                            )
                            clean = assistant_text
                    presearch: list[dict[str, Any]] = []
                    event_name = ""
                    if (
                        enforce_presearch
                        and clean
                        and self._requires_workspace_preinspect(user_message)
                        and not tools_executed_this_turn
                    ):
                        self._status("pre-inspecting workspace…")
                        presearch = self._preinspect_tool_calls_for_workspace(
                            user_message
                        )
                        event_name = "workspace_preinspect"
                    elif (
                        enforce_presearch
                        and clean
                        and self._requires_presearch_for_code(user_message)
                        and not tools_executed_this_turn
                    ):
                        self._status("pre-searching for code context…")
                        presearch = self._presearch_tool_calls_for_code(user_message)
                        event_name = "presearch_for_code"
                    elif (
                        enforce_presearch
                        and clean
                        and self._requires_web_presearch_for_factual(user_message)
                        and not web_search_executed_this_turn
                    ):
                        self._status("pre-searching web for factual context…")
                        presearch = self._presearch_tool_calls_for_factual(user_message)
                        event_name = "presearch_for_factual"
                    if presearch:
                        tool_calls = presearch
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})}
                                    for call in presearch
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision(event_name, user_message, assistant_text)
                        clean = ""
                    if tool_calls:
                        pass
                    else:
                        if pending_nonthink:
                            on_chunk(pending_nonthink)
                        self.history.append({"role": "assistant", "content": clean})
                        if log_interaction:
                            self._log_interaction_trajectory(
                                source="chat_turn",
                                goal=user_message,
                                prompt=user_message,
                                final_text=clean,
                                tool_payloads=turn_tool_payloads,
                                assistant_action=assistant_action_text,
                            )
                        return assistant_text

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history[-1]["content"] = self._strip_thinking(
                self.history[-1]["content"]
            )
            assistant_action_text = self._canonical_tool_call_json(tool_calls)
            self._log_tool_training_sample(
                user_message=user_message,
                assistant_text=assistant_text,
                tool_calls=tool_calls,
            )
            if on_tool_phase:
                on_tool_phase()

            for call in tool_calls:
                start_name = str(call.get("name", ""))
                start_args = call.get("args", {})
                if not isinstance(start_args, dict):
                    start_args = {}
                start_args = dict(start_args)
                if on_tool_start:
                    try:
                        on_tool_start(start_name, start_args)
                    except Exception:
                        pass
                try:
                    executed = self._run_tool_call_with_reflection(user_message, call)
                except Exception as e:
                    executed = {
                        "name": start_name,
                        "args": start_args,
                        "result": {"ok": False, "error": f"tool execution crashed: {e}"},
                        "reflection": {"status": "failed", "reason": "exception"},
                        "initial_name": start_name,
                        "initial_args": start_args,
                    }
                result = executed.get("result", {})
                tool_name = str(executed.get("name", call.get("name", "")))
                tool_args = executed.get("args", call.get("args", {}))
                display_name = str(executed.get("initial_name", start_name))
                display_args = executed.get("initial_args", start_args)
                if not isinstance(display_args, dict):
                    display_args = start_args
                else:
                    display_args = dict(display_args)
                if on_tool:
                    try:
                        on_tool(display_name, display_args, result)
                    except Exception:
                        pass
                if (
                    tool_name == "search_web"
                    and isinstance(result, dict)
                    and result.get("ok", False)
                ):
                    web_search_executed_this_turn = True
                if (
                    tool_name == "get_current_datetime"
                    and isinstance(result, dict)
                    and result.get("ok", False)
                ):
                    datetime_executed_this_turn = True
                if tool_name in {"find_in_memory", "search_memory"} and isinstance(
                    result, dict
                ):
                    memory_checked_this_turn = True
                tool_payload = {
                    "tool": tool_name,
                    "args": tool_args,
                    "display_tool": display_name,
                    "display_args": display_args,
                    "initial_tool": executed.get("initial_name", call.get("name", "")),
                    "initial_args": executed.get("initial_args", call.get("args", {})),
                    "reflection": executed.get("reflection", {}),
                    "result": result,
                }
                self.history.append(
                    {
                        "role": "tool",
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
                turn_tool_payloads.append(tool_payload)
            tools_executed_this_turn = True
            continue_after_tools = True

        final_text = (
            self._fallback_answer_from_tools()
            or "Tool-call loop limit reached. Return direct answer."
        )
        self.history.append({"role": "assistant", "content": final_text})
        if log_interaction:
            self._log_interaction_trajectory(
                source="chat_turn",
                goal=user_message,
                prompt=user_message,
                final_text=final_text,
                tool_payloads=turn_tool_payloads,
                assistant_action=assistant_action_text,
            )
        return final_text

    def run_cli(self) -> None:
        # Enable phase status messages for CLI mode
        self.on_status = print_phase

        print("Local Coding Assistant")
        print("Type 'exit' or 'quit' to stop.")
        print(
            "Commands: /reset, /status, /maxout <n>, /stream, /temperature <n>, /top_p <n>, /compact, /autosize, /autostop, /session"
        )
        print("More help: /help")
        print(f"session: {self._session_name}")

        if self.autonomous_enabled:
            steps_text = (
                "infinite" if self.autonomous_steps <= 0 else str(self.autonomous_steps)
            )
            print(f"autonomous boot: on (steps={steps_text})")
            self.run_autonomous(
                self.autonomous_default_objective, self.autonomous_steps
            )

        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                return

            if not user_input:
                if self.autonomous_enabled:
                    self.run_autonomous(
                        self.autonomous_default_objective, self.autonomous_steps
                    )
                continue
            if user_input.lower() in {"exit", "quit"}:
                self._save_session()
                print("bye")
                return
            if user_input.lower() in {"/reset", "reset"}:
                self._new_session()
                print(f"context reset | new session: {self._session_name}")
                continue
            if user_input.startswith("/"):
                parts = user_input.strip().split()
                cmd = parts[0].lower()
                if cmd in {"/quit", "/exit"}:
                    self._save_session()
                    print("bye")
                    return
                if cmd in {"/maxout", "/maxoutput", "/max_output"}:
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_max_output_tokens", None)
                        current = getter() if callable(getter) else None
                        if current is None:
                            print("max output tokens: unavailable for this model")
                        else:
                            print(f"max output tokens: {current}")
                        continue
                    try:
                        value = int(parts[1])
                    except ValueError:
                        print("usage: /maxout <positive_integer>")
                        continue
                    setter = getattr(self.model, "set_max_output_tokens", None)
                    if not callable(setter):
                        print("this model does not support changing max output tokens")
                        continue
                    ok, msg = setter(value)
                    print(msg)
                    continue
                if cmd in {"/ctx", "/context"}:
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_context_window", None)
                        current = getter() if callable(getter) else None
                        if current is None:
                            print("context window: unavailable for this model")
                        else:
                            print(f"context window: {current}")
                        continue
                    try:
                        value = int(parts[1])
                    except ValueError:
                        print("usage: /ctx <positive_integer>")
                        continue
                    setter = getattr(self.model, "set_context_window", None)
                    if not callable(setter):
                        print("this model does not support changing context window")
                        continue
                    ok, msg = setter(value)
                    print(msg)
                    continue
                if cmd == "/stream":
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_stream_mode", None)
                        mode = getter() if callable(getter) else "unknown"
                        print(f"stream mode: {mode}")
                        continue
                    setter = getattr(self.model, "set_stream_mode", None)
                    if not callable(setter):
                        print("this model does not support stream mode changes")
                        continue
                    ok, msg = setter(parts[1])
                    print(msg)
                    continue
                if cmd == "/temperature":
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_temperature", None)
                        temp = getter() if callable(getter) else "unknown"
                        print(f"temperature: {temp}")
                        continue
                    setter = getattr(self.model, "set_temperature", None)
                    if not callable(setter):
                        print("this model does not support temperature changes")
                        continue
                    ok, msg = setter(parts[1])
                    print(msg)
                    continue
                if cmd == "/top_p" or cmd == "/topp":
                    if len(parts) == 1:
                        getter = getattr(self.model, "get_top_p", None)
                        top_p = getter() if callable(getter) else "unknown"
                        print(f"top_p: {top_p}")
                        continue
                    setter = getattr(self.model, "set_top_p", None)
                    if not callable(setter):
                        print("this model does not support top_p changes")
                        continue
                    ok, msg = setter(parts[1])
                    print(msg)
                    continue
                if cmd == "/compact":
                    if len(parts) == 1 or parts[1].lower() == "status":
                        mode = "on" if self.compact_context_enabled else "off"
                        print(
                            f"compact context: {mode} (threshold_chars={self.max_context_chars})"
                        )
                        continue
                    sub = parts[1].lower()
                    if sub in {"on", "off"}:
                        self.compact_context_enabled = sub == "on"
                        print(f"compact context: {sub}")
                        continue
                    print("usage: /compact [on|off|status]")
                    continue
                if cmd in {"/autosize", "/autolimits", "/autolimit"}:
                    getter = getattr(self.model, "get_auto_limits", None)
                    applier = getattr(self.model, "apply_auto_limits", None)
                    if len(parts) == 1 or parts[1].lower() == "apply":
                        if not callable(applier):
                            print("this model does not support auto limit tuning")
                            continue
                        ok, msg = applier()
                        print(msg)
                        continue
                    if parts[1].lower() in {"status", "show"}:
                        data = getter() if callable(getter) else None
                        if isinstance(data, dict) and data:
                            print("auto limits:")
                            for k, v in data.items():
                                print(f"- {k}: {v}")
                        else:
                            print("auto limits: unavailable for this model")
                        continue
                    print("usage: /autosize [apply|status]")
                    continue
                if cmd == "/status":
                    getter_stream = getattr(self.model, "get_stream_mode", None)
                    getter_maxout = getattr(self.model, "get_max_output_tokens", None)
                    getter_ctx = getattr(self.model, "get_context_window", None)
                    getter_auto = getattr(self.model, "get_auto_limits", None)
                    stream_mode = (
                        getter_stream() if callable(getter_stream) else "unknown"
                    )
                    maxout = getter_maxout() if callable(getter_maxout) else None
                    ctx = getter_ctx() if callable(getter_ctx) else None
                    auto_limits = getter_auto() if callable(getter_auto) else None
                    print("status:")
                    print(f"- stream: {stream_mode}")
                    print(f"- maxout: {maxout if maxout is not None else 'n/a'}")
                    print(f"- ctx: {ctx if ctx is not None else 'n/a'}")
                    if isinstance(auto_limits, dict) and auto_limits:
                        parts_auto = ", ".join(
                            f"{k}={v}" for k, v in auto_limits.items()
                        )
                        print(f"- autosize: {parts_auto}")
                    print(
                        f"- compact: {'on' if self.compact_context_enabled else 'off'} ({self.max_context_chars} chars)"
                    )
                    print(f"- autonomous: {'on' if self.autonomous_enabled else 'off'}")
                    print(f"- autostop: {'on' if self.auto_stop_enabled else 'off'}")
                    continue
                if cmd in {"/help", "help"}:
                    print("Commands:")
                    print("- /reset")
                    print("- /exit, /quit")
                    print("- /status       show runtime settings")
                    print("- /maxout <n>   set max output tokens")
                    print("- /maxout       show current max output tokens")
                    print("- /ctx <n>      set context window (if supported)")
                    print("- /ctx          show current context window")
                    print("- /autosize     apply auto context/maxout tuning")
                    print("- /autosize status")
                    print("- /stream <auto|native|chunk>   set stream mode")
                    print("- /stream       show current stream mode")
                    print("- /temperature <n>   set temperature (0.0-2.0)")
                    print("- /temperature       show current temperature")
                    print("- /top_p <n>    set top_p (0.0-1.0)")
                    print("- /top_p        show current top_p")
                    print("- /compact [on|off|status]")
                    print(
                        "- /autostop [on|off]  toggle repeated-output/loop auto-stop (default: off)"
                    )
                    print("- /session             show current session name")
                    print("- /session list        list all saved sessions")
                    print("- /session new         save current and start fresh")
                    print("- /session save <name> save session with custom name")
                    print("- /session open <name> load a saved session")
                    print("- /auto         show autonomous status")
                    print("- /auto on [steps|infinite] / /auto off")
                    continue
                if cmd == "/session":
                    sub = parts[1].lower() if len(parts) > 1 else ""
                    if not sub:
                        print(
                            f"session: {self._session_name} ({len(self.history)} messages)"
                        )
                        continue
                    if sub == "list":
                        sessions = self._list_sessions()
                        if not sessions:
                            print("no saved sessions")
                        else:
                            for s in sessions:
                                marker = (
                                    " <-- current" if s == self._session_name else ""
                                )
                                print(f"  {s}{marker}")
                        continue
                    if sub == "new":
                        self._new_session()
                        print(f"new session: {self._session_name}")
                        continue
                    if sub == "save":
                        if len(parts) < 3:
                            print("usage: /session save <name>")
                            continue
                        name = "_".join(parts[2:])
                        saved = self._save_session(name)
                        print(f"session saved: {saved}")
                        continue
                    if sub in {"open", "load"}:
                        if len(parts) < 3:
                            print("usage: /session open <name>")
                            continue
                        name = "_".join(parts[2:])
                        result = self._load_session(name)
                        if result is True:
                            print(
                                f"session loaded: {self._session_name} ({len(self.history)} messages)"
                            )
                        elif result is None:
                            sessions = self._list_sessions()
                            matches = [s for s in sessions if name in s]
                            print(f"ambiguous — did you mean: {', '.join(matches)}?")
                        else:
                            print(f"session not found: {name}")
                        continue
                    print(
                        "usage: /session | /session list | /session new | /session save <name> | /session open <name>"
                    )
                    continue
                if cmd == "/autostop":
                    if len(parts) == 1:
                        print(f"autostop: {'on' if self.auto_stop_enabled else 'off'}")
                        continue
                    sub = parts[1].lower()
                    if sub in {"on", "off"}:
                        self.auto_stop_enabled = sub == "on"
                        print(f"autostop: {sub}")
                        continue
                    print("usage: /autostop [on|off]")
                    continue
                if cmd in {"/auto", "/autonomous"}:
                    if len(parts) == 1:
                        mode = "on" if self.autonomous_enabled else "off"
                        steps_text = (
                            "infinite"
                            if self.autonomous_steps <= 0
                            else str(self.autonomous_steps)
                        )
                        print(f"autonomous: {mode} (steps={steps_text})")
                        continue
                    sub = parts[1].lower()
                    if sub == "off":
                        self.autonomous_enabled = False
                        print("autonomous: off")
                        continue
                    if sub == "on":
                        if len(parts) >= 3:
                            raw_steps = parts[2].strip().lower()
                            if raw_steps in {"0", "inf", "infinite", "forever"}:
                                self.autonomous_steps = 0
                            else:
                                try:
                                    self.autonomous_steps = max(
                                        1, min(int(raw_steps), 1000)
                                    )
                                except ValueError:
                                    print("usage: /auto on [steps|infinite]")
                                    continue
                        else:
                            self.autonomous_steps = 0
                        self.autonomous_enabled = True
                        steps_text = (
                            "infinite"
                            if self.autonomous_steps <= 0
                            else str(self.autonomous_steps)
                        )
                        print(f"autonomous: on (steps={steps_text})")
                        print(
                            "tip: press Enter to start default autonomous objective, or type a custom objective."
                        )
                        continue
                    print("usage: /auto | /auto on [steps|infinite] | /auto off")
                    continue

            if self.autonomous_enabled:
                self.run_autonomous(user_input, self.autonomous_steps)
                self._save_session()
                continue

            renderer = StreamRenderer()
            response = self.handle_turn_stream(
                user_input,
                renderer.feed,
                print_tool_event,
                print_tool_start,
                renderer.prepare_tool_output,
            )
            renderer.finish()
            if not renderer.has_output:
                print_formatted_output(response=response)
            elif not renderer.has_answer_output:
                answer = extract_answer_text(response)
                if answer:
                    print_answer_only(answer)
            self._save_session()

    def run_autonomous(self, objective: str, steps: int) -> None:
        finite_steps = steps > 0
        max_steps = max(1, min(steps, 1000)) if finite_steps else 0
        steps_text = str(max_steps) if finite_steps else "infinite"
        print(f"autonomous run: objective='{objective}' | steps={steps_text}")
        print("scope: workspace root only")
        plan_cap = min(max_steps, self.autonomous_plan_step_cap) if finite_steps else None
        state = self._create_task_state(objective, step_cap=plan_cap)
        print("[auto] plan:")
        for idx, step in enumerate(state.steps, start=1):
            print(f"  {idx}. {step.short_label()}")
        learned_signatures: set[str] = set()
        auto_metrics = {
            "tool_calls": 0,
            "failed_tool_calls": 0,
            "est_tokens_out": 0,
            "steps": 0,
            "step_successes": 0,
            "retries": 0,
            "replans": 0,
            "validator_failures": 0,
            "test_repair_attempts": 0,
        }
        print_tool_start("detect_project_context", {"path": ".", "include_runtime": True})
        project_context_result = self.tools.execute(
            "detect_project_context", {"path": ".", "include_runtime": True}
        )
        print_tool_event(
            "detect_project_context",
            {"path": ".", "include_runtime": True},
            project_context_result,
        )
        project_context: dict[str, Any] = {}
        if isinstance(project_context_result, dict) and project_context_result.get("ok", False):
            project_context = project_context_result
            framework = str(project_context.get("framework", "")).strip() or "unknown"
            runner = str(project_context.get("test_runner", "")).strip() or "unknown"
            print(f"[auto] context: framework={framework}, test_runner={runner}")

        repeated_signature_count = 0
        last_signature = ""
        stale_count = 0
        last_text = ""
        rate_limit_count = 0
        i = 0
        try:
            while True:
                next_idx = state.next_runnable_index()
                if next_idx is None:
                    if finite_steps:
                        print("[auto] done: planned steps completed")
                        self._record_strategy_outcome(
                            objective,
                            state.steps,
                            success=True,
                            notes="planned steps completed",
                            context=project_context,
                        )
                        self._print_auto_metrics(auto_metrics)
                        return
                    state = self._create_task_state(objective, step_cap=None)
                    print("[auto] regenerated plan:")
                    for idx, step in enumerate(state.steps, start=1):
                        print(f"  {idx}. {step.short_label()}")
                    next_idx = state.next_runnable_index()
                    if next_idx is None:
                        print("[auto] stopped: no runnable steps after replan")
                        self._print_auto_metrics(auto_metrics)
                        return
                state.current_step = next_idx

                i += 1
                if finite_steps and i > max_steps:
                    print("[auto] step limit reached")
                    return

                step_label = f"{i}/{max_steps}" if finite_steps else f"{i}/infinite"
                plan_lines: list[str] = []
                for idx, step in enumerate(state.steps, start=1):
                    marker = ">>" if idx - 1 == state.current_step else "  "
                    plan_lines.append(
                        f"{marker} {idx}. {step.short_label()} [status={step.status}]"
                    )
                current_step = state.steps[state.current_step]
                current_step.status = "in_progress"
                context_line = ""
                if project_context:
                    context_line = (
                        "Detected project context: "
                        f"framework={project_context.get('framework', 'unknown')}, "
                        f"test_runner={project_context.get('test_runner', 'auto')}, "
                        f"entry_points={project_context.get('entry_points', [])[:3]}\n"
                    )
                prompt = (
                    "Autonomous mode enabled.\n"
                    f"Objective: {objective}\n"
                    f"Loop step: {step_label}\n"
                    "Planned task list:\n"
                    + "\n".join(plan_lines)
                    + "\n"
                    + context_line
                    + f"Current plan step ({state.current_step + 1}/{len(state.steps)}): {current_step.short_label()}\n"
                    + f"Current step args: {json.dumps(current_step.args, ensure_ascii=False)}\n"
                    + "Execute this current plan step now. Use tools as needed and stay in workspace root only.\n"
                    + "After execution, include a short status sentence.\n"
                    + "If finished, output a final line that starts with: AUTONOMOUS_DONE\n"
                    + "If no useful next action remains, output a final line that starts with: AUTONOMOUS_BORED"
                )

                print(f"\n[auto step {i}/{steps_text}]")
                renderer = StreamRenderer()
                response = self.handle_turn_stream(
                    prompt,
                    renderer.feed,
                    print_tool_event,
                    print_tool_start,
                    renderer.prepare_tool_output,
                    enforce_presearch=False,
                    log_interaction=False,
                )
                renderer.finish()
                # Save after every step so messages are on disk before the next
                # step's _messages() call can compact them away.
                self._flush_to_session_log()
                final_text = extract_answer_text(response).strip()
                auto_metrics["steps"] += 1
                auto_metrics["est_tokens_out"] += max(1, len(response) // 4)
                tool_payloads = self._recent_tool_payloads(limit=8)
                auto_metrics["tool_calls"] += len(tool_payloads)
                auto_metrics["failed_tool_calls"] += sum(
                    1
                    for p in tool_payloads
                    if not (
                        isinstance(p.get("result"), dict)
                        and p.get("result", {}).get("ok", False)
                    )
                )
                tool_signature = self._autonomous_tool_signature()
                if tool_signature and tool_signature == last_signature:
                    repeated_signature_count += 1
                else:
                    repeated_signature_count = 0
                    last_signature = tool_signature

                if final_text == last_text and final_text:
                    stale_count += 1
                else:
                    stale_count = 0
                    last_text = final_text

                lower_text = final_text.lower()
                model_done = bool(re.search(r"(?mi)^\s*AUTONOMOUS_DONE\b", final_text))
                model_bored = bool(re.search(r"(?mi)^\s*AUTONOMOUS_BORED\b", final_text))
                if "429" in final_text or "too many requests" in lower_text:
                    wait = min(120, 15 * (2**rate_limit_count))
                    print(
                        f"[auto] rate limited, waiting {wait}s before retrying step {i}..."
                    )
                    time.sleep(wait)
                    rate_limit_count += 1
                    i -= 1  # retry the same step
                    continue
                rate_limit_count = 0
                if (
                    "model backend not available" in lower_text
                    or "openrouter unavailable" in lower_text
                ):
                    print("[auto] stopped: model backend unavailable")
                    self._print_auto_metrics(auto_metrics)
                    return
                if "payment required" in lower_text or "402" in final_text:
                    print(
                        "[auto] stopped: model requires payment (402). Add credits or set OPENROUTER_FALLBACK_MODEL to a free model."
                    )
                    self._print_auto_metrics(auto_metrics)
                    return

                workspace_validation = self._maybe_validate_workspace_after_step(
                    step=current_step,
                    project_context=project_context,
                    payloads=tool_payloads,
                )
                repair_text, repaired_payloads, repaired_validation, repair_history = (
                    self._maybe_run_test_driven_repair(
                        objective=objective,
                        step=current_step,
                        workspace_validation=workspace_validation,
                        current_tool_payloads=tool_payloads,
                        project_context=project_context,
                        auto_metrics=auto_metrics,
                    )
                )
                if repair_text:
                    final_text = repair_text
                    tool_payloads = repaired_payloads
                    workspace_validation = repaired_validation
                else:
                    repair_history = []
                validation = self._validate_autonomous_step_execution(
                    step=current_step,
                    final_text=final_text,
                    tool_payloads=tool_payloads,
                    workspace_validation=workspace_validation,
                )
                validation_score = float(validation.get("score", 0.0) or 0.0)
                if validation_score >= 0.67:
                    validation_status = "success"
                elif validation_score >= 0.4:
                    validation_status = "partial"
                else:
                    validation_status = "failed"
                validation["score"] = round(validation_score, 3)
                validation["status"] = validation_status
                if validation_status == "success":
                    auto_metrics["step_successes"] += 1
                else:
                    auto_metrics["validator_failures"] += 1
                print(
                    "[auto] validator: "
                    f"status={validation_status}, score={validation['score']}, "
                    f"tools_ok={validation.get('ok_tools', 0)}/{validation.get('total_tools', 0)}"
                )

                progress = self._reflect_autonomous_progress(
                    state=state,
                    step_text=current_step.short_label(),
                    final_text=final_text,
                )
                action = str(progress.get("next_action", "advance")).lower()
                reason = str(progress.get("reason", "")).strip()
                confidence = float(progress.get("confidence", 0.5) or 0.5)
                issues = progress.get("issues", [])
                if not isinstance(issues, list):
                    issues = []
                validation_issues = validation.get("issues", [])
                if isinstance(validation_issues, list):
                    issues.extend(validation_issues[:3])
                if model_done and action not in {"retry", "replan"}:
                    action = "done"
                if model_bored and action not in {"retry", "replan"}:
                    action = "bored"
                if validation_score < self.execution_validation_threshold and action in {
                    "advance",
                    "done",
                }:
                    action = "retry" if validation_status == "partial" else "replan"
                    validator_reason = (
                        f"validator gate: status={validation_status}, score={validation_score:.2f} "
                        f"(threshold={self.execution_validation_threshold:.2f})"
                    )
                    reason = (
                        f"{reason} | {validator_reason}".strip(" |")
                        if reason
                        else validator_reason
                    )
                state.history.append(
                    {
                        "loop_step": i,
                        "plan_index": state.current_step,
                        "plan_step": current_step.short_label(),
                        "action": action,
                        "reason": reason,
                        "confidence": round(confidence, 3),
                        "issues": issues[:5],
                        "validator": validation,
                        "model_done_hint": model_done,
                        "model_bored_hint": model_bored,
                        "tool_signature": tool_signature,
                        "final_text": final_text[:1200],
                        "metrics": dict(auto_metrics),
                    }
                )
                self._log_interaction_trajectory(
                    source="autonomous_step",
                    goal=objective,
                    prompt=prompt,
                    final_text=final_text,
                    tool_payloads=tool_payloads,
                    plan=state.steps,
                    assistant_action="",
                    success_hint=validation_status == "success",
                    score_hint=validation_score,
                    validator=validation,
                    metrics=dict(auto_metrics),
                    extra={
                        "current_step": current_step.short_label(),
                        "decision": action,
                        "reason": reason,
                        "repair_history": repair_history,
                    },
                )

                if action == "done":
                    current_step.status = "done"
                    state.completed_step_ids.add(current_step.step_id)
                    learned = self._maybe_learn_skill_from_step(
                        objective=objective,
                        step=current_step,
                        payloads=tool_payloads,
                        validation=validation,
                        learned_signatures=learned_signatures,
                    )
                    if isinstance(learned, dict) and learned.get("ok", False):
                        print(f"[auto] learned skill: {learned.get('name')}")
                    self._record_strategy_outcome(
                        objective,
                        state.steps,
                        success=True,
                        notes=reason or "autonomous objective completed",
                        context=project_context,
                    )
                    print("[auto] done")
                    self._print_auto_metrics(auto_metrics)
                    return
                if action == "bored":
                    self._record_strategy_outcome(
                        objective,
                        state.steps,
                        success=False,
                        notes=reason or "reflection reported no useful next action",
                        context=project_context,
                    )
                    print("[auto] stopped by reflection (no useful next action)")
                    self._print_auto_metrics(auto_metrics)
                    return
                if action == "replan":
                    auto_metrics["replans"] += 1
                    new_steps = progress.get("new_steps", [])
                    if isinstance(new_steps, list) and new_steps:
                        state.steps = self._validate_plan_steps(
                            [s for s in new_steps if isinstance(s, PlanStep)],
                            fallback_goal=objective,
                        )
                    else:
                        state.steps = self._plan_objective_steps(objective, step_cap=plan_cap)
                    state.refresh_dependency_order()
                    state.current_step = 0
                    state.completed_step_ids = set()
                    if reason:
                        print(f"[auto] replanned: {reason}")
                    else:
                        print("[auto] replanned")
                elif action == "retry":
                    auto_metrics["retries"] += 1
                    current_step.status = "pending"
                    if finite_steps:
                        i -= 1
                    if reason:
                        print(f"[auto] retrying current step: {reason}")
                    else:
                        print("[auto] retrying current step")
                    continue
                else:
                    current_step.status = "done"
                    state.completed_step_ids.add(current_step.step_id)
                    learned = self._maybe_learn_skill_from_step(
                        objective=objective,
                        step=current_step,
                        payloads=tool_payloads,
                        validation=validation,
                        learned_signatures=learned_signatures,
                    )
                    if reason:
                        print(f"[auto] advance: {reason}")
                    if issues:
                        print(f"[auto] issues: {', '.join(str(x) for x in issues[:3])}")
                    if isinstance(learned, dict) and learned.get("ok", False):
                        print(f"[auto] learned skill: {learned.get('name')}")

                if (
                    self.autonomous_token_budget > 0
                    and auto_metrics["est_tokens_out"] >= self.autonomous_token_budget
                ):
                    print(
                        "[auto] stopped: token budget reached "
                        f"({auto_metrics['est_tokens_out']}/{self.autonomous_token_budget})"
                    )
                    self._print_auto_metrics(auto_metrics)
                    return
                if self.auto_stop_enabled and repeated_signature_count >= 5:
                    print("[auto] stopped: repeated tool loop detected")
                    self._print_auto_metrics(auto_metrics)
                    return
                if self.auto_stop_enabled and stale_count >= 5:
                    print("[auto] stopped: repeated unchanged output")
                    self._print_auto_metrics(auto_metrics)
                    return
        except KeyboardInterrupt:
            print("\n[auto] interrupted (^C)")
            self._print_auto_metrics(auto_metrics)
            return

    def _recent_tool_payloads(self, limit: int = 8) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for msg in reversed(self.history):
            if msg.get("role") != "tool":
                if payloads:
                    break
                continue
            try:
                payload = json.loads(msg.get("content", ""))
            except Exception:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
            if len(payloads) >= max(1, limit):
                break
        payloads.reverse()
        return payloads

    def _autonomous_tool_signature(self) -> str:
        parts: list[str] = []
        for payload in self._recent_tool_payloads(limit=3):
            name = str(payload.get("tool", "")).strip()
            args = payload.get("args", {})
            if not name:
                continue
            parts.append(f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}")
        return "|".join(sorted(parts))

    def _collect_validation_signals(
        self,
        step: PlanStep,
        tool_payloads: list[dict[str, Any]],
        workspace_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        signals = {
            "tool_successes": 0,
            "tool_failures": 0,
            "command_exit_codes": [],
            "test_runs": [],
            "diff_observed": False,
            "changed_file_count": 0,
            "workspace_validation_ok": None,
        }
        for payload in tool_payloads:
            result = payload.get("result", {})
            tool_name = str(payload.get("tool", "")).strip()
            if isinstance(result, dict) and result.get("ok", False):
                signals["tool_successes"] += 1
            else:
                signals["tool_failures"] += 1

            if tool_name == "execute_command" and isinstance(result, dict):
                exit_code = result.get("exit_code")
                if isinstance(exit_code, int):
                    signals["command_exit_codes"].append(exit_code)

            if tool_name == "run_tests" and isinstance(result, dict):
                signals["test_runs"].append(
                    {
                        "ok": bool(result.get("ok", False)),
                        "exit_code": result.get("exit_code"),
                        "failed": int(result.get("failed", 0) or 0),
                        "errors": int(result.get("errors", 0) or 0),
                        "passed": int(result.get("passed", 0) or 0),
                    }
                )

            if tool_name == "get_git_diff" and isinstance(result, dict):
                diff_text = str(result.get("diff", ""))
                if diff_text.strip():
                    signals["diff_observed"] = True
                    signals["changed_file_count"] = max(
                        int(signals["changed_file_count"] or 0),
                        diff_text.count("\n+++ b/"),
                    )

        if isinstance(workspace_validation, dict):
            signals["workspace_validation_ok"] = bool(
                workspace_validation.get("ok", False)
            )
            validation_signals = workspace_validation.get("validation_signals", {})
            if isinstance(validation_signals, dict):
                signals["diff_observed"] = bool(
                    signals["diff_observed"] or validation_signals.get("has_diff", False)
                )
                signals["changed_file_count"] = max(
                    int(signals["changed_file_count"] or 0),
                    int(validation_signals.get("changed_file_count", 0) or 0),
                )
                exit_code = validation_signals.get("test_exit_code")
                if isinstance(exit_code, int):
                    signals["command_exit_codes"].append(exit_code)
                signals["test_runs"].append(
                    {
                        "ok": bool(validation_signals.get("tests_passed", False)),
                        "exit_code": exit_code,
                        "failed": int(validation_signals.get("failed_tests", 0) or 0),
                        "errors": int(validation_signals.get("test_errors", 0) or 0),
                        "passed": int(
                            workspace_validation.get("tests", {}).get("passed", 0) or 0
                        )
                        if isinstance(workspace_validation.get("tests"), dict)
                        else 0,
                    }
                )

        action = step.action.lower()
        signals["expects_workspace_change"] = any(
            marker in action
            for marker in ("edit", "write", "fix", "implement", "refactor", "create")
        ) or self._should_run_validation_tests(step, tool_payloads)
        return signals

    def _validate_autonomous_step_execution(
        self,
        step: PlanStep,
        final_text: str,
        tool_payloads: list[dict[str, Any]],
        workspace_validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        signals = self._collect_validation_signals(
            step=step,
            tool_payloads=tool_payloads,
            workspace_validation=workspace_validation,
        )
        total_tools = len(tool_payloads)
        ok_tools = int(signals.get("tool_successes", 0) or 0)

        expected_basis = f"{step.action} {step.expected_output}".strip()
        expected_terms = self._extract_keywords(expected_basis, limit=6)
        evidence_parts = [final_text.lower()]
        for payload in tool_payloads[:6]:
            result = payload.get("result", {})
            try:
                evidence_parts.append(
                    json.dumps(result, ensure_ascii=False)[:600].lower()
                )
            except Exception:
                evidence_parts.append(str(result).lower()[:600])
        evidence_text = "\n".join(evidence_parts)
        expected_matches = sum(1 for t in expected_terms if t and t.lower() in evidence_text)
        expected_ratio = (
            expected_matches / len(expected_terms)
            if expected_terms
            else 0.0
        )

        score = 0.0
        if total_tools > 0:
            score += 0.15
            score += 0.15 * (ok_tools / max(1, total_tools))
        else:
            issues.append("no tool evidence for current step")

        test_runs = signals.get("test_runs", [])
        if isinstance(test_runs, list) and test_runs:
            successful_runs = 0
            for run in test_runs:
                if not isinstance(run, dict):
                    continue
                exit_code = run.get("exit_code")
                failed = int(run.get("failed", 0) or 0)
                errors = int(run.get("errors", 0) or 0)
                run_ok = bool(run.get("ok", False)) and failed == 0 and errors == 0
                if run_ok:
                    successful_runs += 1
                else:
                    issues.append(
                        f"tests failed (exit_code={exit_code}, failed={failed}, errors={errors})"
                    )
            score += 0.35 * (successful_runs / max(1, len(test_runs)))
        elif signals.get("expects_workspace_change", False):
            issues.append("no test or validation evidence captured for code-changing step")

        exit_codes = [
            int(code)
            for code in signals.get("command_exit_codes", [])
            if isinstance(code, int)
        ]
        non_zero_exit_codes = [code for code in exit_codes if code != 0]
        if non_zero_exit_codes:
            issues.append(
                "non-zero exit codes observed: "
                + ", ".join(str(code) for code in non_zero_exit_codes[:3])
            )
        else:
            if exit_codes:
                score += 0.15

        if signals.get("expects_workspace_change", False):
            if signals.get("diff_observed", False) or int(
                signals.get("changed_file_count", 0) or 0
            ) > 0:
                score += 0.15
            else:
                issues.append("expected workspace diff was not observed")

        workspace_ok = signals.get("workspace_validation_ok")
        if workspace_ok is True:
            score += 0.1
        elif workspace_ok is False:
            issues.append("workspace validation failed")

        failure_markers = (
            "error",
            "failed",
            "exception",
            "traceback",
            "invalid",
            "unknown tool",
        )
        if final_text.strip() and not any(m in final_text.lower() for m in failure_markers):
            score += 0.1
        else:
            issues.append("assistant output indicates unresolved failure")

        score += 0.1 * expected_ratio
        if expected_terms and expected_matches == 0:
            issues.append("expected output terms missing from evidence")

        score = max(0.0, min(score, 1.0))
        if score >= 0.67:
            status = "success"
        elif score >= 0.4:
            status = "partial"
        else:
            status = "failed"

        return {
            "status": status,
            "score": round(score, 3),
            "issues": issues[:5],
            "ok_tools": ok_tools,
            "total_tools": total_tools,
            "expected_matches": expected_matches,
            "expected_terms": expected_terms,
            "signals": signals,
        }

    @staticmethod
    def _tool_names_from_payloads(payloads: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for payload in payloads:
            name = str(payload.get("tool", "")).strip()
            if name:
                names.append(name)
        return names

    def _should_run_validation_tests(
        self, step: PlanStep, payloads: list[dict[str, Any]]
    ) -> bool:
        edit_markers = {
            "create_file",
            "edit_file",
            "write_file",
            "delete_file",
            "execute_command",
        }
        tool_names = set(self._tool_names_from_payloads(payloads))
        action = step.action.lower()
        return (
            bool(tool_names.intersection(edit_markers))
            or "edit" in action
            or "refactor" in action
            or "fix" in action
            or "implement" in action
        )

    def _maybe_validate_workspace_after_step(
        self,
        step: PlanStep,
        project_context: dict[str, Any],
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.autonomous_validate_changes:
            return None
        if not self._should_run_validation_tests(step, payloads):
            return None

        runner = "auto"
        if isinstance(project_context, dict):
            detected_runner = str(project_context.get("test_runner", "")).strip()
            if detected_runner and detected_runner.lower() != "unknown":
                runner = detected_runner

        args = {"path": ".", "test_runner": runner, "test_args": "", "timeout": 180}
        print_tool_start("validate_workspace_changes", args)
        result = self.tools.execute("validate_workspace_changes", args)
        print_tool_event("validate_workspace_changes", args, result)
        return result

    @staticmethod
    def _workspace_validation_signals(
        workspace_validation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(workspace_validation, dict):
            return {}
        signals = workspace_validation.get("validation_signals", {})
        return signals if isinstance(signals, dict) else {}

    def _should_attempt_test_driven_repair(
        self,
        step: PlanStep,
        workspace_validation: dict[str, Any] | None,
        attempt_no: int,
    ) -> bool:
        if attempt_no >= self.autonomous_test_repair_attempts:
            return False
        if not self.autonomous_validate_changes:
            return False
        signals = self._workspace_validation_signals(workspace_validation)
        failed_tests = int(signals.get("failed_tests", 0) or 0)
        test_errors = int(signals.get("test_errors", 0) or 0)
        if failed_tests <= 0 and test_errors <= 0:
            return False
        action = step.action.lower()
        return any(
            marker in action
            for marker in ("fix", "edit", "write", "implement", "refactor", "patch")
        ) or self._should_run_validation_tests(step, [])

    @staticmethod
    def _render_test_failure_context(
        workspace_validation: dict[str, Any] | None,
        max_items: int = 4,
    ) -> str:
        signals = {}
        if isinstance(workspace_validation, dict):
            raw = workspace_validation.get("validation_signals", {})
            if isinstance(raw, dict):
                signals = raw
        failures = signals.get("test_failures", [])
        lines: list[str] = []
        if isinstance(failures, list):
            for item in failures[: max(1, max_items)]:
                if not isinstance(item, dict):
                    continue
                nodeid = str(item.get("nodeid", "")).strip()
                message = str(item.get("message", "")).strip()
                summary = str(item.get("summary", "")).strip()
                if nodeid and message:
                    lines.append(f"- {nodeid}: {message}")
                elif summary:
                    lines.append(f"- {summary}")
        if not lines and isinstance(workspace_validation, dict):
            tests = workspace_validation.get("tests", {})
            if isinstance(tests, dict):
                stderr = str(tests.get("stderr", "")).strip()
                stdout = str(tests.get("stdout", "")).strip()
                excerpt = stderr or stdout
                if excerpt:
                    lines.append(f"- {excerpt[:320]}")
        return "\n".join(lines)

    def _test_failure_snapshot(
        self, workspace_validation: dict[str, Any] | None
    ) -> dict[str, Any]:
        signals = self._workspace_validation_signals(workspace_validation)
        failures = signals.get("test_failures", [])
        signature_parts: list[str] = []
        if isinstance(failures, list):
            for item in failures[:8]:
                if not isinstance(item, dict):
                    continue
                nodeid = str(item.get("nodeid", "")).strip()
                message = str(item.get("message", "")).strip()
                summary = str(item.get("summary", "")).strip()
                part = nodeid or summary or message
                if part:
                    signature_parts.append(part[:180])
        return {
            "tests_passed": bool(signals.get("tests_passed", False)),
            "failed_tests": int(signals.get("failed_tests", 0) or 0),
            "test_errors": int(signals.get("test_errors", 0) or 0),
            "signature": "|".join(signature_parts),
            "summary": self._render_test_failure_context(workspace_validation),
        }

    @staticmethod
    def _repair_outcome_label(
        before: dict[str, Any], after: dict[str, Any]
    ) -> str:
        before_total = int(before.get("failed_tests", 0) or 0) + int(
            before.get("test_errors", 0) or 0
        )
        after_total = int(after.get("failed_tests", 0) or 0) + int(
            after.get("test_errors", 0) or 0
        )
        if bool(after.get("tests_passed", False)):
            return "resolved"
        if after_total < before_total:
            return "improved"
        if after_total > before_total:
            return "regressed"
        if str(after.get("signature", "")) != str(before.get("signature", "")):
            return "changed"
        return "unchanged"

    @staticmethod
    def _repair_hypothesis_key(hypothesis: dict[str, Any]) -> str:
        if not isinstance(hypothesis, dict):
            return ""
        core = str(hypothesis.get("hypothesis", "")).strip().lower()
        files = hypothesis.get("suspected_files", [])
        if not isinstance(files, list):
            files = []
        normalized_files = "|".join(sorted(str(p).strip().lower() for p in files if str(p).strip()))
        return f"{core}|{normalized_files}".strip("|")

    @staticmethod
    def _repair_fix_signature(payloads: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            tool = str(payload.get("tool", payload.get("name", ""))).strip().lower()
            args = payload.get("args", {})
            if not tool:
                continue
            try:
                args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
            except Exception:
                args_text = str(args)
            parts.append(f"{tool}:{args_text[:320]}")
        joined = "|".join(parts)
        return joined[:1200]

    @staticmethod
    def _build_root_cause_context(
        step: PlanStep,
        project_context: dict[str, Any],
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        framework = str(project_context.get("framework", "")).strip().lower()
        if framework:
            context["framework"] = framework
        test_runner = str(project_context.get("test_runner", "")).strip().lower()
        if test_runner:
            context["test_runner"] = test_runner
        action = step.action.lower()
        if "pytest" in action or "python" in action or framework in {"pytest", "django", "flask"}:
            context["language"] = "python"
        elif "node" in action or "npm" in action or framework in {"node", "nextjs", "react"}:
            context["language"] = "node"
        return context

    @staticmethod
    def _extract_root_cause_error_text(workspace_validation: dict[str, Any] | None) -> str:
        if not isinstance(workspace_validation, dict):
            return ""
        lines: list[str] = []
        signals = workspace_validation.get("validation_signals", {})
        if isinstance(signals, dict):
            failures = signals.get("test_failures", [])
            if isinstance(failures, list):
                for item in failures[:6]:
                    if not isinstance(item, dict):
                        continue
                    summary = str(item.get("summary", "")).strip()
                    message = str(item.get("message", "")).strip()
                    nodeid = str(item.get("nodeid", "")).strip()
                    if summary:
                        lines.append(summary)
                    elif nodeid and message:
                        lines.append(f"{nodeid}: {message}")
                    elif message:
                        lines.append(message)
        tests = workspace_validation.get("tests", {})
        if isinstance(tests, dict):
            stderr = str(tests.get("stderr", "")).strip()
            stdout = str(tests.get("stdout", "")).strip()
            if stderr:
                lines.append(stderr[:900])
            elif stdout:
                lines.append(stdout[:600])
        return "\n".join(line for line in lines if line).strip()[:1800]

    @staticmethod
    def _normalize_root_cause_fix_call(
        tool_name: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        name = str(tool_name or "").strip()
        normalized_args = dict(args or {})
        if name == "run_terminal":
            cmd = str(normalized_args.get("cmd", "")).strip()
            if cmd and "action" not in normalized_args:
                # One-shot command path is safer and deterministic for template fixes.
                name = "execute_command"
                normalized_args = {"cmd": cmd, "path": ".", "timeout": 180}
        return name, normalized_args

    def _maybe_apply_root_cause_fix(
        self,
        step: PlanStep,
        workspace_validation: dict[str, Any] | None,
        project_context: dict[str, Any],
        auto_metrics: dict[str, Any],
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
        store = self._strategy_memory_store()
        if store is None or not hasattr(store, "find_root_causes"):
            return False, [], workspace_validation, None
        error_text = self._extract_root_cause_error_text(workspace_validation)
        if not error_text:
            return False, [], workspace_validation, None
        context = self._build_root_cause_context(step=step, project_context=project_context)
        try:
            matches = store.find_root_causes(error_text=error_text, context=context, limit=1)
        except Exception:
            return False, [], workspace_validation, None
        if not isinstance(matches, list) or not matches:
            return False, [], workspace_validation, None
        selected = matches[0]
        fix_template = selected.get("fix_template", [])
        if not isinstance(fix_template, list) or not fix_template:
            return False, [], workspace_validation, None

        root_payloads: list[dict[str, Any]] = []
        all_ok = True
        for item in fix_template:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", "")).strip()
            args = item.get("args", {})
            if not tool_name or not isinstance(args, dict):
                continue
            tool_name, args = self._normalize_root_cause_fix_call(tool_name, args)
            result = self.tools.execute(tool_name, args)
            payload = {"tool": tool_name, "args": args, "result": result, "source": "root_cause"}
            root_payloads.append(payload)
            auto_metrics["tool_calls"] = int(auto_metrics.get("tool_calls", 0) or 0) + 1
            if not (isinstance(result, dict) and result.get("ok", False)):
                auto_metrics["failed_tool_calls"] = int(
                    auto_metrics.get("failed_tool_calls", 0) or 0
                ) + 1
                all_ok = False
                break

        updated_validation = self._maybe_validate_workspace_after_step(
            step=step,
            project_context=project_context,
            payloads=root_payloads,
        )
        signals = self._workspace_validation_signals(updated_validation)
        solved = bool(signals.get("tests_passed", False))
        confidence = float(selected.get("confidence", 0.5) or 0.5)
        if hasattr(store, "record_root_cause_feedback"):
            try:
                store.record_root_cause_feedback(
                    root_cause_id=str(selected.get("id", "")),
                    success=bool(all_ok and solved),
                    confidence=1.0 if solved else max(0.0, confidence * 0.6),
                )
            except Exception:
                pass
        history_item = {
            "attempt": 0,
            "kind": "root_cause",
            "pattern": str(selected.get("pattern", "")).strip(),
            "root_cause_id": str(selected.get("id", "")).strip(),
            "hypothesis": f"root-cause match: {str(selected.get('pattern', '')).strip()}",
            "confidence": confidence,
            "result": "success" if solved else "failed",
            "impact": "resolved" if solved else "no_change",
            "outcome": "resolved" if solved else "unchanged",
        }
        return solved, root_payloads, updated_validation, history_item

    def _propose_test_failure_hypothesis(
        self,
        objective: str,
        step: PlanStep,
        workspace_validation: dict[str, Any],
        repair_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        changed_files = workspace_validation.get("changed_files", [])
        if not isinstance(changed_files, list):
            changed_files = []
        failure_context = self._render_test_failure_context(workspace_validation)
        history_lines: list[str] = []
        for item in repair_history[-3:]:
            if not isinstance(item, dict):
                continue
            history_lines.append(
                f"- Attempt {item.get('attempt', '?')}: "
                f"{str(item.get('hypothesis', '')).strip()} "
                f"=> {str(item.get('outcome', '')).strip()}"
            )
        history_text = "\n".join(history_lines) or "- none"
        prompt = (
            "You are debugging a failing autonomous code-edit step.\n"
            "Return strict JSON only with keys:\n"
            "hypothesis (string), suspected_files (array), rationale (string), next_check (string), confidence (0..1).\n"
            f"Objective: {objective}\n"
            f"Current step: {step.short_label()}\n"
            f"Changed files: {changed_files[:10]}\n"
            f"Current failing tests:\n{failure_context or '- No structured failure summary available'}\n"
            f"Prior repair attempts:\n{history_text}"
        )
        fallback_message = failure_context.splitlines()[0].strip() if failure_context else ""
        fallback = {
            "hypothesis": (
                f"Fix the code path exercised by the current failing test. {fallback_message}".strip()
            ),
            "suspected_files": changed_files[:5],
            "rationale": "Use the parsed failing test summary and the recently changed files as the first debugging anchor.",
            "next_check": "Inspect the failing test target and the changed implementation before patching.",
            "confidence": 0.45,
        }
        try:
            raw = self.model.generate(
                [
                    {
                        "role": "system",
                        "content": "You are a debugging supervisor. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            payload = parse_json_payload(self._strip_thinking(raw))
            if not isinstance(payload, dict):
                return fallback
            hypothesis = str(payload.get("hypothesis", "")).strip()
            rationale = str(payload.get("rationale", "")).strip()
            next_check = str(payload.get("next_check", "")).strip()
            suspected_files = payload.get("suspected_files", [])
            if not isinstance(suspected_files, list):
                suspected_files = changed_files[:5]
            normalized_files = [
                str(item).strip()
                for item in suspected_files
                if str(item).strip()
            ][:6]
            try:
                confidence = float(payload.get("confidence", fallback["confidence"]))
            except Exception:
                confidence = float(fallback["confidence"])
            return {
                "hypothesis": hypothesis or fallback["hypothesis"],
                "suspected_files": normalized_files or changed_files[:5],
                "rationale": rationale or fallback["rationale"],
                "next_check": next_check or fallback["next_check"],
                "confidence": max(0.0, min(confidence, 1.0)),
            }
        except Exception:
            return fallback

    @staticmethod
    def _render_repair_history(repair_history: list[dict[str, Any]]) -> str:
        if not repair_history:
            return "- none"
        lines: list[str] = []
        for item in repair_history[-4:]:
            if not isinstance(item, dict):
                continue
            before_total = int(item.get("before_total", 0) or 0)
            after_total = int(item.get("after_total", 0) or 0)
            lines.append(
                f"- Attempt {item.get('attempt', '?')}: "
                f"hypothesis={str(item.get('hypothesis', '')).strip()} | "
                f"outcome={str(item.get('outcome', '')).strip()} | "
                f"failures {before_total}->{after_total}"
            )
        return "\n".join(lines) or "- none"

    def _build_test_repair_prompt(
        self,
        objective: str,
        step: PlanStep,
        workspace_validation: dict[str, Any],
        attempt_no: int,
        hypothesis: dict[str, Any] | None = None,
        repair_history: list[dict[str, Any]] | None = None,
    ) -> str:
        signals = self._workspace_validation_signals(workspace_validation)
        failed_tests = int(signals.get("failed_tests", 0) or 0)
        test_errors = int(signals.get("test_errors", 0) or 0)
        changed_files = workspace_validation.get("changed_files", [])
        if not isinstance(changed_files, list):
            changed_files = []
        failure_context = self._render_test_failure_context(workspace_validation)
        diff_excerpt = str(workspace_validation.get("diff_excerpt", "")).strip()
        diff_excerpt = diff_excerpt[:1200] if diff_excerpt else ""
        hypothesis = hypothesis or {}
        history_text = self._render_repair_history(repair_history or [])
        return (
            "Autonomous repair mode enabled.\n"
            f"Objective: {objective}\n"
            f"Current step: {step.short_label()}\n"
            f"Repair attempt: {attempt_no + 1}/{self.autonomous_test_repair_attempts}\n"
            "The previous code-editing step left the workspace failing tests.\n"
            f"Failed tests: {failed_tests}, test errors: {test_errors}\n"
            f"Changed files: {changed_files[:10]}\n"
            f"Current debugging hypothesis: {str(hypothesis.get('hypothesis', '')).strip() or 'Investigate the current failing path.'}\n"
            f"Suspected files: {hypothesis.get('suspected_files', []) if isinstance(hypothesis.get('suspected_files', []), list) else []}\n"
            f"Why this hypothesis: {str(hypothesis.get('rationale', '')).strip() or 'Use the failing test output as the main signal.'}\n"
            f"Next check before patching: {str(hypothesis.get('next_check', '')).strip() or 'Read the implicated code and test around the failure.'}\n"
            "Previous repair attempts:\n"
            f"{history_text}\n"
            "Focus only on fixing the concrete failing tests using the workspace state.\n"
            "Use tools to inspect files, test your hypothesis, patch code, and rerun tests. Avoid unrelated refactors.\n"
            "Failure summaries:\n"
            f"{failure_context or '- No structured failure summary available'}\n"
            + (
                f"Diff excerpt:\n{diff_excerpt}\n"
                if diff_excerpt
                else ""
            )
            + "After the fix attempt, include a short status sentence."
        )

    def _maybe_run_test_driven_repair(
        self,
        objective: str,
        step: PlanStep,
        workspace_validation: dict[str, Any] | None,
        current_tool_payloads: list[dict[str, Any]],
        project_context: dict[str, Any],
        auto_metrics: dict[str, Any],
    ) -> tuple[str | None, list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
        validation = workspace_validation
        combined_payloads = list(current_tool_payloads)
        latest_text: str | None = None
        repair_history: list[dict[str, Any]] = []
        repair_state = RepairState()

        # Root-cause first-pass: deterministic fix templates before model-driven repair.
        solved_by_root_cause, root_payloads, validation, root_history = self._maybe_apply_root_cause_fix(
            step=step,
            workspace_validation=validation,
            project_context=project_context,
            auto_metrics=auto_metrics,
        )
        if root_payloads:
            combined_payloads.extend(root_payloads)
        if root_history:
            repair_history.append(root_history)
        if solved_by_root_cause:
            latest_text = "Applied root-cause fix template and tests are now passing."
            return latest_text, combined_payloads, validation, repair_history

        for attempt_no in range(self.autonomous_test_repair_attempts):
            if not self._should_attempt_test_driven_repair(step, validation, attempt_no):
                break
            repair_state.attempts += 1
            auto_metrics["test_repair_attempts"] = int(
                auto_metrics.get("test_repair_attempts", 0) or 0
            ) + 1
            before_snapshot = self._test_failure_snapshot(validation)
            hypothesis = self._propose_test_failure_hypothesis(
                objective=objective,
                step=step,
                workspace_validation=validation or {},
                repair_history=repair_history,
            )
            hypothesis_key = self._repair_hypothesis_key(hypothesis)
            if hypothesis_key and hypothesis_key in repair_state.tried_hypotheses:
                repair_history.append(
                    {
                        "attempt": attempt_no + 1,
                        "hypothesis": str(hypothesis.get("hypothesis", "")).strip(),
                        "confidence": float(hypothesis.get("confidence", 0.0) or 0.0),
                        "result": "failed",
                        "impact": "no change",
                        "outcome": "unchanged",
                        "before_total": int(before_snapshot.get("failed_tests", 0) or 0)
                        + int(before_snapshot.get("test_errors", 0) or 0),
                        "after_total": int(before_snapshot.get("failed_tests", 0) or 0)
                        + int(before_snapshot.get("test_errors", 0) or 0),
                        "skipped": "duplicate_hypothesis",
                    }
                )
                continue
            if hypothesis_key:
                repair_state.tried_hypotheses.append(hypothesis_key)
            repair_state.hypotheses.append(hypothesis)
            repair_prompt = self._build_test_repair_prompt(
                objective=objective,
                step=step,
                workspace_validation=validation or {},
                attempt_no=attempt_no,
                hypothesis=hypothesis,
                repair_history=repair_history,
            )
            print(f"[auto] test-driven repair attempt {attempt_no + 1}")
            renderer = StreamRenderer()
            response = self.handle_turn_stream(
                repair_prompt,
                renderer.feed,
                print_tool_event,
                print_tool_start,
                renderer.prepare_tool_output,
                enforce_presearch=False,
                log_interaction=False,
            )
            renderer.finish()
            latest_text = extract_answer_text(response).strip()
            repair_payloads = self._recent_tool_payloads(limit=8)
            combined_payloads.extend(repair_payloads)
            auto_metrics["est_tokens_out"] += max(1, len(response) // 4)
            auto_metrics["tool_calls"] += len(repair_payloads)
            auto_metrics["failed_tool_calls"] += sum(
                1
                for p in repair_payloads
                if not (
                    isinstance(p.get("result"), dict)
                    and p.get("result", {}).get("ok", False)
                )
            )
            fix_signature = self._repair_fix_signature(repair_payloads)
            duplicate_fix = bool(fix_signature) and fix_signature in repair_state.tried_fixes
            if fix_signature:
                repair_state.tried_fixes.append(fix_signature)
            validation = self._maybe_validate_workspace_after_step(
                step=step,
                project_context=project_context,
                payloads=repair_payloads,
            )
            after_snapshot = self._test_failure_snapshot(validation)
            outcome = self._repair_outcome_label(before_snapshot, after_snapshot)
            result_label = (
                "success"
                if bool(after_snapshot.get("tests_passed", False))
                else "failed"
            )
            impact_label = (
                "resolved"
                if outcome == "resolved"
                else "improved"
                if outcome == "improved"
                else "regressed"
                if outcome == "regressed"
                else "no change"
            )
            repair_history.append(
                {
                    "attempt": attempt_no + 1,
                    "hypothesis": str(hypothesis.get("hypothesis", "")).strip(),
                    "suspected_files": hypothesis.get("suspected_files", []),
                    "confidence": float(hypothesis.get("confidence", 0.0) or 0.0),
                    "result": result_label,
                    "impact": impact_label,
                    "before_total": int(before_snapshot.get("failed_tests", 0) or 0)
                    + int(before_snapshot.get("test_errors", 0) or 0),
                    "after_total": int(after_snapshot.get("failed_tests", 0) or 0)
                    + int(after_snapshot.get("test_errors", 0) or 0),
                    "outcome": outcome,
                    "duplicate_fix": duplicate_fix,
                }
            )
            signals = self._workspace_validation_signals(validation)
            if bool(signals.get("tests_passed", False)):
                break
        return latest_text, combined_payloads, validation, repair_history

    @staticmethod
    def _infer_skill_input_name(
        key: str, value: Any, seen: set[str]
    ) -> str | None:
        lowered = key.lower()
        candidate = ""
        if "path" in lowered or "file" in lowered:
            candidate = "file_path"
        elif "query" in lowered or "pattern" in lowered or "symbol" in lowered:
            candidate = "query"
        elif lowered in {"runner", "test_runner"}:
            candidate = "test_runner"
        elif lowered in {"cmd", "command"}:
            candidate = "command"
        elif isinstance(value, str) and ("/" in value or value.endswith(".py")):
            candidate = key
        if not candidate:
            return None
        candidate = re.sub(r"[^a-z0-9_]+", "_", candidate.lower()).strip("_")
        if not candidate:
            return None
        if candidate in seen:
            return candidate
        seen.add(candidate)
        return candidate

    def _abstract_skill_template(
        self, tool_calls: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        inputs: list[str] = []
        seen_inputs: set[str] = set()
        templates: list[dict[str, Any]] = []
        placeholder_by_signature: dict[tuple[str, str], str] = {}

        for call in tool_calls:
            name = str(call.get("tool", "")).strip()
            raw_args = call.get("args", {})
            if not name or not isinstance(raw_args, dict):
                continue
            templated_args: dict[str, Any] = {}
            for key, value in raw_args.items():
                signature = (name, str(key))
                placeholder = placeholder_by_signature.get(signature)
                if placeholder is None:
                    placeholder = self._infer_skill_input_name(str(key), value, seen_inputs)
                    if placeholder:
                        placeholder_by_signature[signature] = placeholder
                        if placeholder not in inputs:
                            inputs.append(placeholder)
                if placeholder and isinstance(value, (str, int, float, bool)):
                    templated_args[str(key)] = f"${{{placeholder}}}"
                else:
                    templated_args[str(key)] = value
            templates.append({"tool": name, "args": templated_args})
        return inputs, templates

    def _maybe_learn_skill_from_step(
        self,
        objective: str,
        step: PlanStep,
        payloads: list[dict[str, Any]],
        validation: dict[str, Any],
        learned_signatures: set[str],
    ) -> dict[str, Any] | None:
        if not self.autonomous_skill_learning_enabled:
            return None
        score = float(validation.get("score", 0.0) or 0.0)
        if score < max(0.65, self.execution_validation_threshold):
            return None

        blocked = {"record_memory_feedback", "record_skill_outcome", "create_skill"}
        tool_calls: list[dict[str, Any]] = []
        for payload in payloads[:8]:
            name = str(payload.get("tool", "")).strip()
            if not name or name in blocked:
                continue
            args = payload.get("args", {})
            result = payload.get("result", {})
            if isinstance(result, dict) and not result.get("ok", False):
                return None
            if not isinstance(args, dict):
                args = {}
            tool_calls.append({"tool": name, "args": args})
        if not tool_calls:
            return None

        signature = "|".join(
            f"{c['tool']}:{json.dumps(c.get('args', {}), sort_keys=True, ensure_ascii=False)}"
            for c in tool_calls
        )
        if signature in learned_signatures:
            return None
        learned_signatures.add(signature)

        action_slug = re.sub(r"[^a-z0-9_]+", "_", step.action.lower()).strip("_")
        if not action_slug:
            action_slug = "step"
        skill_hash = abs(hash(signature)) % 100000
        skill_name = f"auto_{action_slug[:28]}_{skill_hash}"
        skill_key = action_slug[:40] or "step"
        description = (
            f"Auto-learned successful workflow for objective '{objective[:80]}' "
            f"at step '{step.action}'."
        )
        keywords = self._extract_keywords(
            f"{objective} {step.action} {step.expected_output}", limit=8
        )
        skill_inputs, steps_template = self._abstract_skill_template(tool_calls)
        created = self.tools.execute(
            "create_skill",
            {
                "name": skill_name,
                "description": description,
                "keywords": keywords,
                "skill": skill_key,
                "inputs": skill_inputs,
                "steps_template": steps_template or tool_calls,
                "match_conditions": keywords[:4],
                "tool_calls": tool_calls,
            },
        )
        if not created.get("ok", False):
            return {"ok": False, "name": skill_name, "error": created.get("error", "")}

        self.tools.execute(
            "record_skill_outcome",
            {
                "name": skill_name,
                "success": True,
                "confidence": score,
                "notes": "auto-learned from autonomous run",
            },
        )
        return {"ok": True, "name": skill_name, "tool_calls": len(tool_calls)}
