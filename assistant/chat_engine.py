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
from .utils import get_env_bool, get_env_int, parse_json_payload


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

    def current_step_text(self) -> str:
        if not self.steps:
            return "No plan available."
        idx = max(0, min(self.current_step, len(self.steps) - 1))
        return self.steps[idx].short_label()

    def next_runnable_index(self) -> int | None:
        if not self.steps:
            return None
        # Prefer the current pointer if it is runnable.
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            if step.status == "pending" and all(
                dep in self.completed_step_ids for dep in step.depends_on
            ):
                return self.current_step
        for idx, step in enumerate(self.steps):
            if step.status != "pending":
                continue
            if all(dep in self.completed_step_ids for dep in step.depends_on):
                return idx
        return None

    def remaining_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "pending")


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

            raw_step_id = item.get("step_id", idx)
            try:
                step_id = int(raw_step_id)
            except Exception:
                step_id = idx
            if step_id <= 0 or step_id in seen_ids:
                step_id = idx if idx not in seen_ids else max(seen_ids or {0}) + 1
            seen_ids.add(step_id)

            action = str(item.get("action", "") or item.get("step", "")).strip()
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
            expected_output = str(item.get("expected_output", "")).strip()

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

        ids = {s.step_id for s in steps}
        normalized: list[PlanStep] = []
        for idx, s in enumerate(sorted(steps, key=lambda x: x.step_id), start=1):
            step_id = int(s.step_id) if int(s.step_id) > 0 else idx
            if step_id in {x.step_id for x in normalized}:
                step_id = idx
            deps = [d for d in s.depends_on if d in ids and d != step_id]
            normalized.append(
                PlanStep(
                    step_id=step_id,
                    action=s.action.strip() or f"step_{idx}",
                    args=s.args if isinstance(s.args, dict) else {},
                    depends_on=deps,
                    expected_output=s.expected_output.strip(),
                    status=s.status if s.status in {"pending", "in_progress", "done"} else "pending",
                )
            )

        # Break cycles/invalid ordering by forcing dependencies to prior step ids.
        seen: set[int] = set()
        for s in normalized:
            s.depends_on = [d for d in s.depends_on if d in seen]
            seen.add(s.step_id)

        if not normalized:
            return ChatEngine._fallback_plan_steps(fallback_goal)
        return normalized

    def _plan_objective_steps(
        self, objective: str, step_cap: int | None = None
    ) -> list[PlanStep]:
        self._status("planning task steps…")
        cap = (
            max(3, min(int(step_cap), 20))
            if step_cap is not None
            else max(3, min(self.autonomous_plan_step_cap, 20))
        )
        planning_prompt = (
            "Break this objective into actionable execution steps with strict structure.\n"
            f"Objective: {objective}\n"
            "Return strict JSON only in this shape:\n"
            "{\n"
            '  "steps":[\n'
            "    {\n"
            '      "step_id": 1,\n'
            '      "action": "search_project",\n'
            '      "args": {"query":"..."},\n'
            '      "depends_on": [],\n'
            '      "expected_output": "what this step should produce"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Rules: 1..{cap} steps, unique step_id, depends_on references earlier step ids only."
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
            if isinstance(retry_call, dict):
                retry_name = str(retry_call.get("name", "")).strip() or initial_name
                retry_args = retry_call.get("args", {})
                if not isinstance(retry_args, dict):
                    retry_args = initial_args
            else:
                retry_name = initial_name
                retry_args = initial_args
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
        # Capture lightweight file-like mentions such as src/app.py or README.md
        pattern = re.compile(r"([A-Za-z0-9_\-./]+\.[A-Za-z0-9_]{1,8})")
        out: list[str] = []
        seen = set()
        for match in pattern.findall(text):
            candidate = match.strip().strip(".,;:()[]{}\"'")
            if not candidate or "/" in candidate and candidate.startswith("http"):
                continue
            if candidate not in seen:
                out.append(candidate)
                seen.add(candidate)
            if len(out) >= limit:
                break
        return out

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
            calls.append({"name": "read_file", "args": {"path": p, "max_chars": 6000}})
        return calls

    def _ensure_web_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        web_search_executed: bool,
    ) -> list[dict[str, Any]]:
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
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False
        web_search_executed_this_turn = False
        datetime_executed_this_turn = False

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
                        return assistant_text

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history[-1]["content"] = self._strip_thinking(
                self.history[-1]["content"]
            )
            self._log_tool_training_sample(
                user_message=user_message,
                assistant_text=assistant_text,
                tool_calls=tool_calls,
            )
            if on_tool_phase:
                on_tool_phase()

            for call in tool_calls:
                if on_tool_start:
                    on_tool_start(call["name"], call.get("args", {}))
                executed = self._run_tool_call_with_reflection(user_message, call)
                result = executed.get("result", {})
                tool_name = str(executed.get("name", call.get("name", "")))
                tool_args = executed.get("args", call.get("args", {}))
                if on_tool:
                    on_tool(tool_name, tool_args, result)
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
                tool_payload = {
                    "tool": tool_name,
                    "args": tool_args,
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
            tools_executed_this_turn = True
            continue_after_tools = True

        final_text = (
            self._fallback_answer_from_tools()
            or "Tool-call loop limit reached. Return direct answer."
        )
        self.history.append({"role": "assistant", "content": final_text})
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
                        return
                    state = self._create_task_state(objective, step_cap=None)
                    print("[auto] regenerated plan:")
                    for idx, step in enumerate(state.steps, start=1):
                        print(f"  {idx}. {step.short_label()}")
                    next_idx = state.next_runnable_index()
                    if next_idx is None:
                        print("[auto] stopped: no runnable steps after replan")
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
                prompt = (
                    "Autonomous mode enabled.\n"
                    f"Objective: {objective}\n"
                    f"Loop step: {step_label}\n"
                    "Planned task list:\n"
                    + "\n".join(plan_lines)
                    + "\n"
                    f"Current plan step ({state.current_step + 1}/{len(state.steps)}): {current_step.short_label()}\n"
                    f"Current step args: {json.dumps(current_step.args, ensure_ascii=False)}\n"
                    "Execute this current plan step now. Use tools as needed and stay in workspace root only.\n"
                    "After execution, include a short status sentence.\n"
                    "If finished, output a final line that starts with: AUTONOMOUS_DONE\n"
                    "If no useful next action remains, output a final line that starts with: AUTONOMOUS_BORED"
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
                )
                renderer.finish()
                # Save after every step so messages are on disk before the next
                # step's _messages() call can compact them away.
                self._flush_to_session_log()
                final_text = extract_answer_text(response).strip()
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
                # Check model-signalled completion FIRST, before any error patterns,
                # so a response that contains "402" or other substrings but also
                # signals DONE/BORED is not misclassified as an error.
                if re.search(r"(?mi)^\s*AUTONOMOUS_DONE\b", final_text):
                    print("[auto] done")
                    return
                if re.search(r"(?mi)^\s*AUTONOMOUS_BORED\b", final_text):
                    print("[auto] stopped by model (bored/no useful next action)")
                    return
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
                    return
                if "payment required" in lower_text or "402" in final_text:
                    print(
                        "[auto] stopped: model requires payment (402). Add credits or set OPENROUTER_FALLBACK_MODEL to a free model."
                    )
                    return

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
                state.history.append(
                    {
                        "loop_step": i,
                        "plan_index": state.current_step,
                        "plan_step": current_step.short_label(),
                        "action": action,
                        "reason": reason,
                        "confidence": round(confidence, 3),
                        "issues": issues[:5],
                        "tool_signature": tool_signature,
                        "final_text": final_text[:1200],
                    }
                )

                if action == "done":
                    print("[auto] done")
                    return
                if action == "bored":
                    print("[auto] stopped by reflection (no useful next action)")
                    return
                if action == "replan":
                    new_steps = progress.get("new_steps", [])
                    if isinstance(new_steps, list) and new_steps:
                        state.steps = self._validate_plan_steps(
                            [s for s in new_steps if isinstance(s, PlanStep)],
                            fallback_goal=objective,
                        )
                    else:
                        state.steps = self._plan_objective_steps(objective, step_cap=plan_cap)
                    state.current_step = 0
                    state.completed_step_ids = set()
                    if reason:
                        print(f"[auto] replanned: {reason}")
                    else:
                        print("[auto] replanned")
                elif action == "retry":
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
                    if reason:
                        print(f"[auto] advance: {reason}")
                    if issues:
                        print(f"[auto] issues: {', '.join(str(x) for x in issues[:3])}")

                if self.auto_stop_enabled and repeated_signature_count >= 5:
                    print("[auto] stopped: repeated tool loop detected")
                    return
                if self.auto_stop_enabled and stale_count >= 5:
                    print("[auto] stopped: repeated unchanged output")
                    return
        except KeyboardInterrupt:
            print("\n[auto] interrupted (^C)")
            return

    def _autonomous_tool_signature(self) -> str:
        parts: list[str] = []
        for msg in reversed(self.history):
            if msg.get("role") != "tool":
                if parts:
                    break
                continue
            try:
                payload = json.loads(msg.get("content", ""))
            except Exception:
                continue
            name = str(payload.get("tool", "")).strip()
            args = payload.get("args", {})
            if not name:
                continue
            parts.append(
                f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            )
            if len(parts) >= 3:
                break
        return "|".join(sorted(parts))
