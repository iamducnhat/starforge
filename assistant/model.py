from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BaseModel:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self.generate(messages)


class OllamaModel(BaseModel):
    def __init__(self, model_name: str, base_url: str = "http://127.0.0.1:11434", timeout: int = 60) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        num_ctx = self._env_int("ASSISTANT_NUM_CTX", 8192)
        num_predict = self._env_int("ASSISTANT_NUM_PREDICT", 2048)
        temperature = self._env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - localhost endpoint
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310 - localhost endpoint
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n\n".join(lines)

    @staticmethod
    def _compose_chat_text(data: dict[str, Any]) -> str:
        message = data.get("message", {})
        thinking = message.get("thinking", "")
        content = message.get("content", "")
        out = ""
        if thinking:
            out += f"<think>{thinking}</think>"
        if content:
            out += content
        return out

    @staticmethod
    def _compose_generate_text(data: dict[str, Any]) -> str:
        thinking = data.get("thinking", "")
        response = data.get("response", "")
        out = ""
        if thinking:
            out += f"<think>{thinking}</think>"
        if response:
            out += response
        return out

    @staticmethod
    def _compose_chat_stream_text(data: dict[str, Any]) -> tuple[str, str]:
        message = data.get("message", {})
        return message.get("thinking", ""), message.get("content", "")

    @staticmethod
    def _compose_generate_stream_text(data: dict[str, Any]) -> tuple[str, str]:
        return data.get("thinking", ""), data.get("response", "")

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.base_url}/api/tags", method="GET")
            with urlopen(req, timeout=5):  # nosec B310 - localhost endpoint
                return True
        except URLError:
            return False
        except Exception:
            return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }
        try:
            data = self._post_json("/api/chat", chat_payload)
            text = self._compose_chat_text(data)
            if text:
                return text
            return data.get("message", {}).get("content", "")
        except HTTPError as e:
            if e.code != 404:
                raise
        except Exception:
            pass

        # Fallback for older/local variants exposing generate endpoint only.
        generate_payload = {
            "model": self.model_name,
            "prompt": self._messages_to_prompt(messages),
            "stream": False,
            "options": self.options,
        }
        try:
            data = self._post_json("/api/generate", generate_payload)
            text = self._compose_generate_text(data)
            if text:
                return text
            return data.get("response", "")
        except Exception:
            return (
                "Local model endpoint available but incompatible. "
                "Set --ollama-url to a valid Ollama server."
            )

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        chat_payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": self.options,
        }
        in_think = False
        try:
            for data in self._stream_post_json("/api/chat", chat_payload):
                thinking, content = self._compose_chat_stream_text(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
            if in_think:
                yield "</think>"
            return
        except HTTPError as e:
            if e.code != 404:
                raise
        except Exception:
            pass

        generate_payload = {
            "model": self.model_name,
            "prompt": self._messages_to_prompt(messages),
            "stream": True,
            "options": self.options,
        }
        in_think = False
        try:
            for data in self._stream_post_json("/api/generate", generate_payload):
                thinking, response = self._compose_generate_stream_text(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                if response:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield response
            if in_think:
                yield "</think>"
            return
        except Exception:
            yield (
                "Local model endpoint available but incompatible. "
                "Set --ollama-url to a valid Ollama server."
            )


class FallbackModel(BaseModel):
    CONTINUATION_PREFIX = "Tool execution is complete."

    def generate(self, messages: list[dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if content.startswith(self.CONTINUATION_PREFIX):
                    continue
                last_user = content
                break
        return (
            "Local model backend not available. Start Ollama and retry. "
            "No tool calls were executed. "
            f"Last user message: {last_user[:300]}"
        )

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield self.generate(messages)


def build_model(model_name: str, base_url: str = "http://127.0.0.1:11434") -> BaseModel:
    ollama = OllamaModel(model_name=model_name, base_url=base_url)
    if ollama.is_available():
        return ollama
    return FallbackModel()
