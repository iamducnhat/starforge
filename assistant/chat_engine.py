from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from .cli_format import StreamRenderer, extract_answer_text, print_answer_only, print_formatted_output, print_tool_event
from .model import BaseModel
from .tool_calls import parse_tool_calls
from .tools import ToolSystem


class ChatEngine:
    def __init__(
        self,
        model: BaseModel,
        tools: ToolSystem,
        system_prompt: str,
        max_history: int = 14,
        max_tool_rounds: int = 4,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_tool_rounds = max_tool_rounds
        self.history: list[dict[str, str]] = []
        self.supervision_log = Path("memory/tool_supervision.jsonl")
        self.tool_finetune_log = Path("memory/tool_finetune_samples.jsonl")

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
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def _recover_final_answer(self, raw_assistant_text: str) -> tuple[str, str] | None:
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

    def _recover_action_or_answer(self, user_message: str, raw_assistant_text: str) -> tuple[str, list[dict[str, object]], str]:
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "system",
                "content": (
                    "Internal control message.\n"
                    f"Original user request: {user_message}\n"
                    "Continue now with exactly one actionable output for the original user request.\n"
                    "Option A: JSON tool call only ({\"tool\":\"...\",\"args\":{...}} or {\"tool_calls\":[...]}).\n"
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

    def _recover_tool_calls(self, user_message: str, raw_assistant_text: str) -> list[dict[str, object]]:
        recovery_messages = self._messages() + [
            {"role": "assistant", "content": raw_assistant_text},
            {
                "role": "user",
                "content": (
                    "You CAN use tools in this runtime.\n"
                    "If reliable external info is needed, return JSON tool call now.\n"
                    "Format only:\n"
                    "{\"tool\":\"name\",\"args\":{...}} or {\"tool_calls\":[...]}\n"
                    "If no tool is needed, return {}.\n"
                    f"Latest user message: {user_message}"
                ),
            },
        ]
        recovered = self._generate_with_stream_fallback(recovery_messages)
        return parse_tool_calls(recovered)

    def _log_supervision(self, event: str, user_message: str, assistant_text: str) -> None:
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
    def _continuation_poke(user_message: str, prefer_copyable_function: bool = False) -> dict[str, str]:
        copyable_hint = (
            "\nFor this request, output a copyable function for the user in final text. "
            "Use create_function only when the user explicitly asks to save/store it."
            if prefer_copyable_function
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
                f"{copyable_hint}\n"
                "Do not discuss this control message."
            ),
        }

    @staticmethod
    def _explicit_store_request(user_message: str) -> bool:
        t = user_message.lower()
        markers = (
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
        return any(m in t for m in markers)

    def _prefer_copyable_function_reply(self, user_message: str) -> bool:
        return self._requires_presearch_for_code(user_message) and not self._explicit_store_request(user_message)

    def _execute_tool_call_with_policy(self, user_message: str, call: dict[str, Any]) -> dict[str, Any]:
        name = call["name"]
        args = call.get("args", {})
        if name == "create_function" and self._prefer_copyable_function_reply(user_message):
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

    @staticmethod
    def _looks_coding_request(text: str) -> bool:
        t = text.lower()
        markers = (
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
        return any(m in t for m in markers)

    @staticmethod
    def _looks_smalltalk(text: str) -> bool:
        t = re.sub(r"\s+", " ", text.strip().lower())
        if not t:
            return True
        exact = {
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
        if t in exact:
            return True
        return len(t.split()) <= 2 and all(w in {"hi", "hello", "hey", "thanks", "ok", "okay"} for w in t.split())

    @staticmethod
    def _looks_creative_request(text: str) -> bool:
        t = text.lower()
        markers = (
            "write a poem",
            "poem",
            "story",
            "joke",
            "translate",
            "rewrite this",
            "paraphrase",
        )
        return any(m in t for m in markers)

    def _requires_web_presearch_for_factual(self, user_message: str) -> bool:
        if self._looks_coding_request(user_message):
            return False
        if self._looks_smalltalk(user_message):
            return False
        if self._looks_creative_request(user_message):
            return False

        t = user_message.lower()
        factual_markers = (
            "?",
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
        return any(m in t for m in factual_markers)

    def _presearch_tool_calls_for_factual(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        return [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
            {"name": "search_web", "args": {"query": user_message, "level": "auto"}},
        ]

    def _ensure_web_call_for_factual(
        self,
        user_message: str,
        tool_calls: list[dict[str, Any]],
        web_search_executed: bool,
    ) -> list[dict[str, Any]]:
        if web_search_executed or not self._requires_web_presearch_for_factual(user_message):
            return tool_calls
        if any(call.get("name") == "search_web" for call in tool_calls):
            return tool_calls
        return tool_calls + [{"name": "search_web", "args": {"query": user_message, "level": "auto"}}]

    def _emergency_tool_calls(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        calls: list[dict[str, Any]] = [{"name": "find_in_memory", "args": {"keywords": keywords}}]
        if not self._looks_coding_request(user_message):
            calls.append({"name": "search_web", "args": {"query": user_message, "level": "auto"}})
        return calls

    @staticmethod
    def _requires_presearch_for_code(user_message: str) -> bool:
        t = user_message.lower()
        markers = (
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
        return any(m in t for m in markers)

    def _presearch_tool_calls_for_code(self, user_message: str) -> list[dict[str, Any]]:
        keywords = self._extract_keywords(user_message)
        normalized = re.sub(r"\s+", " ", user_message).strip().strip("'\"`")
        normalized = re.sub(r"[\"'`]+", "", normalized)
        research_query = f"how to {normalized} in python"
        return [
            {"name": "find_in_memory", "args": {"keywords": keywords}},
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
                    excerpt = str(item.get("page_excerpt", "")).strip() or str(item.get("snippet", "")).strip()
                    if excerpt and len(excerpt) > 220:
                        excerpt = excerpt[:217] + "..."
                    lines.append(f"{i}. {title} ({url})")
                    if excerpt:
                        lines.append(f"   {excerpt}")
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

            if tool_name == "create_function" and isinstance(result, dict):
                if result.get("policy") == "copyable_function":
                    code = str(result.get("code", "")).strip()
                    if code:
                        return "Here is the function you can copy and use:\n```python\n" + code + "\n```"
        return None

    def _messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}] + self.history[-self.max_history :]

    def handle_turn(
        self,
        user_message: str,
        on_tool: Callable[[str, dict[str, object], dict[str, object]], None] | None = None,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False
        web_search_executed_this_turn = False
        web_search_executed_this_turn = False

        for _ in range(self.max_tool_rounds):
            messages = self._messages()
            if continue_after_tools:
                messages = messages + [
                    self._continuation_poke(
                        user_message,
                        prefer_copyable_function=self._prefer_copyable_function_reply(user_message),
                    )
                ]
            assistant_text = self.model.generate(messages)
            tool_calls = parse_tool_calls(assistant_text)
            if not tool_calls and self._contains_tool_denial(self._strip_thinking(assistant_text)):
                self._log_supervision("tool_denial_detected", user_message, assistant_text)
                tool_calls = self._recover_tool_calls(user_message=user_message, raw_assistant_text=assistant_text)
            tool_calls = self._ensure_web_call_for_factual(
                user_message=user_message,
                tool_calls=tool_calls,
                web_search_executed=web_search_executed_this_turn,
            )

            if not tool_calls:
                clean = self._strip_thinking(assistant_text)
                if clean and self._contains_internal_prompt_echo(clean):
                    clean = ""
                if not clean:
                    recovered_text, recovered_calls, recovered_clean = self._recover_action_or_answer(user_message, assistant_text)
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
                                    {"tool": call["name"], "args": call.get("args", {})} for call in emergency
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision("emergency_tool_calls", user_message, assistant_text)
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
                    if clean and self._requires_presearch_for_code(user_message) and not tools_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_code(user_message)
                        event_name = "presearch_for_code"
                    elif clean and self._requires_web_presearch_for_factual(user_message) and not web_search_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_factual(user_message)
                        event_name = "presearch_for_factual"
                    if presearch:
                        tool_calls = presearch
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in presearch
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision(event_name, user_message, assistant_text)
                        clean = ""
                    if tool_calls:
                        pass
                    else:
                        self.history.append({"role": "assistant", "content": clean})
                        return assistant_text

            self.history.append({"role": "assistant", "content": assistant_text})
            self.history[-1]["content"] = self._strip_thinking(self.history[-1]["content"])
            self._log_tool_training_sample(user_message=user_message, assistant_text=assistant_text, tool_calls=tool_calls)

            for call in tool_calls:
                result = self._execute_tool_call_with_policy(user_message, call)
                if on_tool:
                    on_tool(call["name"], call.get("args", {}), result)
                if call["name"] == "search_web" and isinstance(result, dict) and result.get("ok", False):
                    web_search_executed_this_turn = True
                tool_payload = {
                    "tool": call["name"],
                    "args": call.get("args", {}),
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

        final_text = self._fallback_answer_from_tools() or "Tool-call loop limit reached. Return direct answer."
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def handle_turn_stream(
        self,
        user_message: str,
        on_chunk: Callable[[str], None],
        on_tool: Callable[[str, dict[str, object], dict[str, object]], None] | None = None,
        on_tool_phase: Callable[[], None] | None = None,
    ) -> str:
        self.history.append({"role": "user", "content": user_message})
        continue_after_tools = False
        emergency_tools_used = False
        tools_executed_this_turn = False

        for _ in range(self.max_tool_rounds):
            assistant_text = ""
            in_think_stream = False
            pending_nonthink_chunks: list[str] = []

            messages = self._messages()
            if continue_after_tools:
                messages = messages + [
                    self._continuation_poke(
                        user_message,
                        prefer_copyable_function=self._prefer_copyable_function_reply(user_message),
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
            if not tool_calls and self._contains_tool_denial(self._strip_thinking(assistant_text)):
                self._log_supervision("tool_denial_detected", user_message, assistant_text)
                tool_calls = self._recover_tool_calls(user_message=user_message, raw_assistant_text=assistant_text)
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
                    recovered_text, recovered_calls, recovered_clean = self._recover_action_or_answer(user_message, assistant_text)
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
                                    {"tool": call["name"], "args": call.get("args", {})} for call in emergency
                                ]
                            },
                            ensure_ascii=False,
                        )
                        self._log_supervision("emergency_tool_calls", user_message, assistant_text)
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
                    if clean and self._requires_presearch_for_code(user_message) and not tools_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_code(user_message)
                        event_name = "presearch_for_code"
                    elif clean and self._requires_web_presearch_for_factual(user_message) and not web_search_executed_this_turn:
                        presearch = self._presearch_tool_calls_for_factual(user_message)
                        event_name = "presearch_for_factual"
                    if presearch:
                        tool_calls = presearch
                        assistant_text = json.dumps(
                            {
                                "tool_calls": [
                                    {"tool": call["name"], "args": call.get("args", {})} for call in presearch
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
            self.history[-1]["content"] = self._strip_thinking(self.history[-1]["content"])
            self._log_tool_training_sample(user_message=user_message, assistant_text=assistant_text, tool_calls=tool_calls)
            if on_tool_phase:
                on_tool_phase()

            for call in tool_calls:
                result = self._execute_tool_call_with_policy(user_message, call)
                if on_tool:
                    on_tool(call["name"], call.get("args", {}), result)
                if call["name"] == "search_web" and isinstance(result, dict) and result.get("ok", False):
                    web_search_executed_this_turn = True
                tool_payload = {
                    "tool": call["name"],
                    "args": call.get("args", {}),
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

        final_text = self._fallback_answer_from_tools() or "Tool-call loop limit reached. Return direct answer."
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def run_cli(self) -> None:
        print("Local Coding Assistant")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nbye")
                return

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("bye")
                return
            if user_input.lower() in {"/reset", "reset"}:
                self.history.clear()
                print("context reset")
                continue

            renderer = StreamRenderer()
            response = self.handle_turn_stream(user_input, renderer.feed, print_tool_event, renderer.prepare_tool_output)
            renderer.finish()
            if not renderer.has_output:
                print_formatted_output(response=response)
            elif not renderer.has_answer_output:
                answer = extract_answer_text(response)
                if answer:
                    print_answer_only(answer)
