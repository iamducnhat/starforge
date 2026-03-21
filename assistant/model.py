from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .utils import get_env_bool as _env_bool
from .utils import get_env_float as _env_float
from .utils import get_env_int as _env_int

# Fix for macOS DNS resolution issue - use nslookup as fallback
_dns_cache: dict[str, str] = {}


def _resolve_hostname(hostname: str) -> str:
    """Resolve hostname with fallback to nslookup."""
    if hostname in _dns_cache:
        return _dns_cache[hostname]

    # Try standard resolution first
    try:
        ip = socket.gethostbyname(hostname)
        _dns_cache[hostname] = ip
        return ip
    except socket.gaierror:
        pass

    # Fallback to nslookup
    try:
        result = subprocess.run(
            ["nslookup", hostname], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "Address:" in line and not line.startswith("Server:"):
                parts = line.split()
                if parts:
                    ip = parts[-1]
                    if "." in ip and all(p.isdigit() for p in ip.split(".")):
                        _dns_cache[hostname] = ip
                        return ip
    except Exception:
        pass

    # If all fails, return original hostname
    return hostname


# Patch socket.getaddrinfo to use our DNS resolver
_original_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Patch getaddrinfo to handle DNS resolution failures."""
    try:
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        # Try resolving with our fallback
        resolved_host = _resolve_hostname(host)
        if resolved_host != host:
            return _original_getaddrinfo(
                resolved_host, port, family, type, proto, flags
            )
        raise


socket.getaddrinfo = _patched_getaddrinfo


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _detect_total_ram_gb() -> float:
    # POSIX path (macOS/Linux)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if (
            isinstance(pages, int)
            and isinstance(page_size, int)
            and pages > 0
            and page_size > 0
        ):
            return (pages * page_size) / (1024**3)
    except Exception:
        pass
    return 16.0


def _estimate_model_params_b(model_name: str) -> float:
    text = (model_name or "").lower()
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b", text)
    if not matches:
        return 0.0
    try:
        values = [float(x) for x in matches]
        return max(values) if values else 0.0
    except Exception:
        return 0.0


def _auto_local_limits(model_name: str) -> tuple[int, int, float, float]:
    ram_gb = _detect_total_ram_gb()
    params_b = _estimate_model_params_b(model_name)

    if ram_gb <= 12:
        base_ctx, base_pred = 4096, 1024
    elif ram_gb <= 16:
        base_ctx, base_pred = 8192, 2048
    elif ram_gb <= 24:
        base_ctx, base_pred = 16384, 4096
    elif ram_gb <= 32:
        base_ctx, base_pred = 32768, 8192
    elif ram_gb <= 48:
        base_ctx, base_pred = 65536, 12288
    elif ram_gb <= 64:
        base_ctx, base_pred = 98304, 16384
    else:
        base_ctx, base_pred = 131072, 24576

    if params_b >= 70:
        scale = 0.25
    elif params_b >= 34:
        scale = 0.4
    elif params_b >= 20:
        scale = 0.5
    elif params_b >= 14:
        scale = 0.65
    elif params_b >= 8:
        scale = 0.8
    elif 0 < params_b <= 4:
        scale = 1.4
    else:
        scale = 1.0

    ctx = int(base_ctx * scale)
    pred = int(base_pred * scale)
    ctx = max(4096, min(ctx, 131072))
    pred = max(1024, min(pred, 32768))
    pred = min(pred, max(1024, ctx // 2))
    return ctx, pred, ram_gb, params_b


def _pick_openrouter_model(
    available: list[str],
    requested: str,
    env_fallback: str = "",
) -> tuple[str, str]:
    if not available:
        return requested, "model list unavailable, keep requested model"
    requested_clean = (requested or "").strip()
    if requested_clean in available:
        return requested_clean, "requested model is available"

    candidates = [
        env_fallback.strip(),
        "arcee-ai/trinity-large-preview:free",
        "openrouter/auto",
    ]
    for cand in candidates:
        if cand and cand in available:
            return cand, f"requested model not found, fallback to {cand}"

    free_models = [m for m in available if m.endswith(":free")]
    if free_models:
        return (
            free_models[0],
            f"requested model not found, fallback to {free_models[0]}",
        )

    return available[0], f"requested model not found, fallback to {available[0]}"


class BaseModel:
    provider = "unknown"
    model_name = ""
    endpoint = ""
    native_streaming = False
    connect_log: list[str] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        return False, "model does not support dynamic max output tokens"

    def get_max_output_tokens(self) -> int | None:
        return None

    def set_context_window(self, value: int) -> tuple[bool, str]:
        return False, "model does not support dynamic context window changes"

    def get_context_window(self) -> int | None:
        return None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        return False, "model does not support stream mode changes"

    def get_stream_mode(self) -> str:
        return "chunk"

    def apply_auto_limits(self) -> tuple[bool, str]:
        return False, "model does not support auto limit tuning"

    def get_auto_limits(self) -> dict[str, Any] | None:
        return None

    def set_temperature(self, value: float) -> tuple[bool, str]:
        return False, "model does not support dynamic temperature changes"

    def get_temperature(self) -> float | None:
        return None

    def set_top_p(self, value: float) -> tuple[bool, str]:
        return False, "model does not support dynamic top_p changes"

    def get_top_p(self) -> float | None:
        return None

    @staticmethod
    def _stream_text(text: str, chunk_size: int = 28) -> Iterator[str]:
        if not text:
            return
        i = 0
        n = len(text)
        while i < n:
            j = min(n, i + chunk_size)
            # Prefer splitting on whitespace for readability.
            if j < n:
                pivot = text.rfind(" ", i, j)
                if pivot > i:
                    j = pivot + 1
            yield text[i:j]
            i = j

    def info(self) -> dict[str, Any]:
        details: dict[str, Any] = {}
        for key in (
            "temperature",
            "max_tokens",
            "timeout",
            "stream_mode",
            "stream_timeout",
            "context_window",
            "total_ram_gb",
            "model_params_b",
            "auto_limits",
            "auto_limits_snapshot",
            "provider_order",
            "provider_only",
        ):
            if hasattr(self, key):
                details[key] = getattr(self, key)
        if hasattr(self, "options") and isinstance(getattr(self, "options"), dict):
            details["options"] = dict(getattr(self, "options"))
        return {
            "provider": getattr(self, "provider", "unknown"),
            "model": getattr(self, "model_name", ""),
            "endpoint": getattr(self, "endpoint", ""),
            "native_streaming": bool(getattr(self, "native_streaming", False)),
            "connect_log": list(getattr(self, "connect_log", []) or []),
            "details": details,
        }

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        text = self.generate(messages)
        yield from self._stream_text(text)


class OllamaModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "ollama"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        auto_limits = _env_bool("ASSISTANT_AUTO_LIMITS", True)
        user_ctx_raw = os.getenv("ASSISTANT_NUM_CTX", "").strip()
        user_pred_raw = os.getenv("ASSISTANT_NUM_PREDICT", "").strip()

        self.total_ram_gb = round(_detect_total_ram_gb(), 2)
        self.model_params_b = _estimate_model_params_b(model_name)
        self.auto_limits = bool(auto_limits)
        recommended_ctx, recommended_pred, _ram, _params = _auto_local_limits(
            model_name
        )
        self.auto_limits_snapshot = {
            "recommended_ctx": recommended_ctx,
            "recommended_maxout": recommended_pred,
            "ram_gb": round(_ram, 2),
            "model_params_b": _params,
            "auto_enabled": bool(self.auto_limits),
        }

        if auto_limits and not user_ctx_raw and not user_pred_raw:
            num_ctx = recommended_ctx
            num_predict = recommended_pred
        else:
            num_ctx = _env_int("ASSISTANT_NUM_CTX", 262144)
            num_predict = _env_int("ASSISTANT_NUM_PREDICT", 32768)
        num_predict = min(num_predict, max(1024, num_ctx // 2))
        temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.context_window = num_ctx
        self.options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        value = int(value)
        ctx = self.get_context_window() or int(self.options.get("num_ctx", 8192))
        capped = min(value, max(1024, ctx // 2))
        self.options["num_predict"] = capped
        if capped != value:
            return (
                True,
                f"ollama num_predict set to {capped} (capped by context window)",
            )
        return True, f"ollama num_predict set to {capped}"

    def get_max_output_tokens(self) -> int | None:
        raw = self.options.get("num_predict")
        return int(raw) if isinstance(raw, int) else None

    def set_context_window(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "context window must be > 0"
        value = max(1024, min(int(value), 262144))
        self.options["num_ctx"] = value
        current_pred = int(self.options.get("num_predict", 1024))
        self.options["num_predict"] = min(current_pred, max(1024, value // 2))
        self.context_window = value
        return True, f"ollama num_ctx set to {value}"

    def get_context_window(self) -> int | None:
        raw = self.options.get("num_ctx")
        return int(raw) if isinstance(raw, int) else None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"ollama stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def apply_auto_limits(self) -> tuple[bool, str]:
        ctx, pred, ram_gb, params_b = _auto_local_limits(self.model_name)
        self.options["num_ctx"] = ctx
        self.options["num_predict"] = pred
        self.context_window = ctx
        self.total_ram_gb = round(ram_gb, 2)
        self.model_params_b = params_b
        self.auto_limits_snapshot = {
            "recommended_ctx": ctx,
            "recommended_maxout": pred,
            "ram_gb": round(ram_gb, 2),
            "model_params_b": params_b,
            "auto_enabled": bool(self.auto_limits),
        }
        return (
            True,
            f"applied auto limits: ctx={ctx}, maxout={pred} (ram≈{round(ram_gb, 2)}GB, params≈{params_b}B)",
        )

    def get_auto_limits(self) -> dict[str, Any] | None:
        snapshot = getattr(self, "auto_limits_snapshot", None)
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return None

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(
            req, timeout=self.timeout
        ) as resp:  # nosec B310 - localhost endpoint
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(
        self, path: str, payload: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(
            req, timeout=self.timeout
        ) as resp:  # nosec B310 - localhost endpoint
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
        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

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
            fallback_text = self.generate(messages)
            yield from self._stream_text(fallback_text)


class OpenRouterModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 120,
        context_window: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "openrouter"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        self.stream_timeout = _env_int("ASSISTANT_STREAM_TIMEOUT", 35)
        self.temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.top_p = _env_float("ASSISTANT_TOP_P", 0.95)
        self.context_window = (
            int(context_window)
            if isinstance(context_window, int) and context_window > 0
            else None
        )
        self.auto_limits = os.getenv(
            "ASSISTANT_AUTO_LIMITS", "1"
        ).strip().lower() not in {"0", "false", "off"}
        user_pred_raw = os.getenv("ASSISTANT_NUM_PREDICT", "").strip()
        if user_pred_raw:
            self.max_tokens = _env_int("ASSISTANT_NUM_PREDICT", 32768)
        else:
            default_max = 32768
            if isinstance(self.context_window, int) and self.context_window > 0:
                default_max = min(default_max, max(1024, self.context_window // 2))
            self.max_tokens = default_max
        if isinstance(self.context_window, int) and self.context_window > 0:
            self.max_tokens = min(self.max_tokens, max(1024, self.context_window // 2))
        self.auto_limits_snapshot = {
            "recommended_maxout": self._recommended_max_tokens(),
            "context_window": self.context_window,
            "auto_enabled": bool(self.auto_limits),
        }
        self.http_referer = os.getenv(
            "OPENROUTER_HTTP_REFERER", "http://localhost"
        ).strip()
        self.app_name = os.getenv(
            "OPENROUTER_APP_NAME", "Local Coding Assistant"
        ).strip()
        self.fallback_model = os.getenv(
            "OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free"
        ).strip()
        raw_provider = os.getenv("OPENROUTER_PROVIDER", "").strip()
        self.provider_order: list[str] = (
            [p.strip() for p in raw_provider.split(",") if p.strip()]
            if raw_provider
            else []
        )
        self.provider_only: bool = os.getenv(
            "OPENROUTER_PROVIDER_ONLY", "0"
        ).strip().lower() not in {"0", "false", "off", ""}

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        value = int(value)
        if isinstance(self.context_window, int) and self.context_window > 0:
            capped = min(value, max(1024, self.context_window // 2))
        else:
            capped = value
        self.max_tokens = capped
        if capped != value:
            return (
                True,
                f"openrouter max_tokens set to {capped} (capped by context window)",
            )
        return True, f"openrouter max_tokens set to {capped}"

    def get_max_output_tokens(self) -> int | None:
        return int(self.max_tokens) if isinstance(self.max_tokens, int) else None

    def get_context_window(self) -> int | None:
        return (
            int(self.context_window)
            if isinstance(self.context_window, int) and self.context_window > 0
            else None
        )

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"openrouter stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def _recommended_max_tokens(self) -> int:
        base = 32768
        if isinstance(self.context_window, int) and self.context_window > 0:
            return min(base, max(1024, self.context_window // 2))
        return base

    def apply_auto_limits(self) -> tuple[bool, str]:
        recommended = self._recommended_max_tokens()
        self.max_tokens = recommended
        self.auto_limits_snapshot = {
            "recommended_maxout": recommended,
            "context_window": self.context_window,
            "auto_enabled": bool(self.auto_limits),
        }
        if self.context_window:
            return (
                True,
                f"applied auto limits: maxout={recommended} (context={self.context_window})",
            )
        return True, f"applied auto limits: maxout={recommended}"

    def get_auto_limits(self) -> dict[str, Any] | None:
        snapshot = getattr(self, "auto_limits_snapshot", None)
        if isinstance(snapshot, dict):
            return dict(snapshot)
        return None

    def set_temperature(self, value: float) -> tuple[bool, str]:
        try:
            temp = float(value)
            if temp < 0.0 or temp > 2.0:
                return False, "temperature must be between 0.0 and 2.0"
            self.temperature = temp
            return True, f"openrouter temperature set to {temp}"
        except (ValueError, TypeError):
            return False, "invalid temperature value (must be a number)"

    def get_temperature(self) -> float | None:
        return self.temperature

    def set_top_p(self, value: float) -> tuple[bool, str]:
        try:
            top_p = float(value)
            if top_p < 0.0 or top_p > 1.0:
                return False, "top_p must be between 0.0 and 1.0"
            self.top_p = top_p
            return True, f"openrouter top_p set to {top_p}"
        except (ValueError, TypeError):
            return False, "invalid top_p value (must be a number)"

    def get_top_p(self) -> float | None:
        return self.top_p

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_name:
            headers["X-Title"] = self.app_name
        return headers

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(
            req, timeout=self.timeout
        ) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(
        self, path: str, payload: dict[str, Any], timeout: int | None = None
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        effective_timeout = (
            timeout if isinstance(timeout, int) and timeout > 0 else self.timeout
        )
        with urlopen(
            req, timeout=effective_timeout
        ) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_chunk = line[5:].strip()
                    if data_chunk == "[DONE]":
                        break
                else:
                    # Some proxies strip `data:` prefix, keep parser permissive.
                    data_chunk = line
                try:
                    yield json.loads(data_chunk)
                except json.JSONDecodeError:
                    continue

    def _chat_payload(
        self, messages: list[dict[str, str]], stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.provider_order:
            if self.provider_only:
                payload["provider"] = {"only": self.provider_order}
            else:
                payload["provider"] = {
                    "order": self.provider_order,
                    "allow_fallbacks": True,
                }
        return payload

    @staticmethod
    def _extract_final_text(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message", {})
        reasoning = _message_content_to_text(message.get("reasoning", ""))
        content = _message_content_to_text(message.get("content", ""))
        out = ""
        if reasoning:
            out += f"<think>{reasoning}</think>"
        if content:
            out += content
        return out

    @staticmethod
    def _extract_stream_delta(data: dict[str, Any]) -> tuple[str, str]:
        choices = data.get("choices", [])
        if not choices:
            return "", ""
        delta = choices[0].get("delta", {})
        reasoning = _message_content_to_text(
            delta.get("reasoning") or delta.get("reasoning_content") or ""
        )
        content = _message_content_to_text(delta.get("content", ""))
        return reasoning, content

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            req = Request(
                f"{self.base_url}/models",
                headers=self._headers(),
                method="GET",
            )
            with urlopen(req, timeout=8):  # nosec B310 - HTTPS OpenRouter endpoint
                return True
        except Exception:
            return False

    def _switch_model_if_missing(self) -> bool:
        available = list_openrouter_models(self.api_key, self.base_url, timeout=12)
        next_model, reason = _pick_openrouter_model(
            available=available,
            requested=self.model_name,
            env_fallback=self.fallback_model,
        )
        if next_model and next_model != self.model_name:
            old = self.model_name
            self.model_name = next_model
            self.connect_log.append(f"[warn] model '{old}' unavailable")
            self.connect_log.append(f"[ok] switched model to '{next_model}' ({reason})")
            return True
        return False

    def _switch_to_free_fallback(self) -> bool:
        """Switch to the free fallback model when the current model requires payment."""
        fallback = self.fallback_model.strip()
        if fallback and fallback != self.model_name:
            old = self.model_name
            self.model_name = fallback
            # Clear provider routing so the free model isn't restricted to a paid provider
            self.provider_order = []
            self.provider_only = False
            self.connect_log.append(f"[warn] model '{old}' requires payment (402)")
            self.connect_log.append(
                f"[ok] switched to free fallback model '{fallback}' (provider routing cleared)"
            )
            print(
                f"[402] model '{old}' requires payment, switching to free fallback '{fallback}'"
            )
            return True
        return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return "OpenRouter API key missing. Set OPENROUTER_API_KEY or use --provider ollama."
        payload = self._chat_payload(messages, stream=False)
        for attempt in range(4):
            try:
                data = self._post_json("/chat/completions", payload)
                text = self._extract_final_text(data)
                return text or "OpenRouter returned an empty response."
            except HTTPError as e:
                if e.code == 429:
                    wait = min(60, 10 * (2**attempt))
                    print(
                        f"[rate limit] OpenRouter 429, waiting {wait}s (attempt {attempt + 1}/4)..."
                    )
                    time.sleep(wait)
                    continue
                if e.code == 402:
                    if self._switch_to_free_fallback():
                        payload = self._chat_payload(messages, stream=False)
                        continue
                    return "OpenRouter request failed: HTTP Error 402: Payment Required. Add credits or use a free model."
                if e.code == 404 and self._switch_model_if_missing():
                    try:
                        retry = self._post_json(
                            "/chat/completions",
                            self._chat_payload(messages, stream=False),
                        )
                        retry_text = self._extract_final_text(retry)
                        return retry_text or "OpenRouter returned an empty response."
                    except Exception as e2:
                        return f"OpenRouter request failed after model fallback: {e2}"
                return f"OpenRouter request failed: {e}"
            except Exception as e:
                return f"OpenRouter request failed: {e}"
        return "OpenRouter request failed: HTTP Error 429: Too Many Requests"

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.api_key:
            yield "OpenRouter API key missing. Set OPENROUTER_API_KEY or use --provider ollama."
            return

        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

        payload = self._chat_payload(messages, stream=True)
        in_think = False
        emitted = False
        try:
            for data in self._stream_post_json(
                "/chat/completions", payload, timeout=self.stream_timeout
            ):
                thinking, content = self._extract_stream_delta(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                    emitted = True
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
                    emitted = True
            if in_think:
                yield "</think>"
            if not emitted:
                yield from self._stream_text(self.generate(messages))
        except HTTPError as e:
            if in_think:
                yield "</think>"
            if e.code == 429:
                wait = 15
                print(f"[rate limit] OpenRouter 429 (stream), waiting {wait}s...")
                time.sleep(wait)
                yield from self._stream_text(f"OpenRouter stream failed: {e}")
                return
            if e.code == 402:
                if self._switch_to_free_fallback():
                    yield from self._stream_text(self.generate(messages))
                    return
                yield from self._stream_text(
                    "OpenRouter stream failed: HTTP Error 402: Payment Required. "
                    "Add credits or set OPENROUTER_FALLBACK_MODEL to a free model."
                )
                return
            if e.code == 404 and self._switch_model_if_missing():
                yield from self._stream_text(self.generate(messages))
                return
            yield from self._stream_text(f"OpenRouter stream failed: {e}")
        except Exception as e:
            if in_think:
                yield "</think>"
            # Stream fallback for models/providers that fail SSE.
            fallback_text = self.generate(messages)
            if "OpenRouter request failed" in fallback_text:
                fallback_text = f"OpenRouter stream failed: {e}. {fallback_text}"
            yield from self._stream_text(fallback_text)


class GoogleModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/models",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "google"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        self.stream_timeout = _env_int("ASSISTANT_STREAM_TIMEOUT", 35)
        self.temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.max_tokens = _env_int("ASSISTANT_NUM_PREDICT", 16384)

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        self.max_tokens = int(value)
        return True, f"google max_tokens set to {self.max_tokens}"

    def get_max_output_tokens(self) -> int | None:
        return int(self.max_tokens) if isinstance(self.max_tokens, int) else None

    def get_context_window(self) -> int | None:
        return None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"google stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def apply_auto_limits(self) -> tuple[bool, str]:
        return True, "google auto limits ignored"

    def get_auto_limits(self) -> dict[str, Any] | None:
        return None

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(
        self, path: str, payload: dict[str, Any], timeout: int | None = None
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        effective_timeout = (
            timeout if isinstance(timeout, int) and timeout > 0 else self.timeout
        )
        with urlopen(req, timeout=effective_timeout) as resp:  # nosec B310
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_chunk = line[5:].strip()
                    if data_chunk == "[DONE]":
                        break
                else:
                    data_chunk = line
                try:
                    yield json.loads(data_chunk)
                except json.JSONDecodeError:
                    continue

    def _chat_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        # Gemini API strict role alternation requirement.
        # Roles must be 'user' or 'model'.
        alt_messages: list[dict[str, str]] = []
        for m in messages:
            orig_role = m.get("role", "user")
            role = "model" if orig_role == "assistant" else "user"
            content = m.get("content", "")
            if not content:
                continue
            if alt_messages and alt_messages[-1].get("role") == role:
                alt_messages[-1]["parts"][0]["text"] += f"\n\n{content}"
            else:
                alt_messages.append({"role": role, "parts": [{"text": content}]})

        # Ensure the first message is a user message
        if alt_messages and alt_messages[0].get("role") == "model":
            alt_messages.insert(
                0, {"role": "user", "parts": [{"text": "(conversation begin)"}]}
            )

        payload: dict[str, Any] = {
            "contents": alt_messages,
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        if self.max_tokens > 0:
            payload["generationConfig"]["maxOutputTokens"] = self.max_tokens

        # Enable Gemini thinking output for all Google models.
        # The Gemini API silently ignores thinkingConfig on models that don't
        # support it, so this is safe.  Thinking output surfaces as <think>…</think>
        # in the stream and is rendered in the CLI by StreamRenderer.
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": -1,  # let the model decide how much to think
            "includeThoughts": True,
        }

        return payload

    @staticmethod
    def _extract_final_text(data: dict[str, Any]) -> str:
        """
        Extract final text from a Gemini response.

        For thinking-enabled models, reasoning summaries are returned in parts
        with thought=true. We wrap those in <think>...</think> and append the
        normal answer content after.
        """
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        first = candidates[0]
        content = first.get("content", {}) or {}
        parts = content.get("parts", []) or []
        if not isinstance(parts, list) or not parts:
            return ""

        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = _message_content_to_text(part.get("text", ""))
            if not text:
                continue
            if part.get("thought") is True:
                reasoning_chunks.append(text)
            else:
                answer_chunks.append(text)

        reasoning = "".join(reasoning_chunks)
        answer = "".join(answer_chunks)
        out = ""
        if reasoning:
            out += f"<think>{reasoning}</think>"
        if answer:
            out += answer
        return out

    @staticmethod
    def _extract_stream_delta(data: dict[str, Any]) -> tuple[str, str]:
        """
        Extract incremental reasoning/content chunks from a streaming response.

        Returns (thinking, content) where each is a string chunk that may be
        empty. For non-thinking models, thinking will always be empty.
        """
        candidates = data.get("candidates", [])
        if not candidates:
            return "", ""
        first = candidates[0]
        content = first.get("content", {}) or {}
        parts = content.get("parts", []) or []
        if not isinstance(parts, list) or not parts:
            return "", ""

        reasoning_chunks: list[str] = []
        answer_chunks: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = _message_content_to_text(part.get("text", ""))
            if not text:
                continue
            if part.get("thought") is True:
                reasoning_chunks.append(text)
            else:
                answer_chunks.append(text)

        return "".join(reasoning_chunks), "".join(answer_chunks)

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            req = Request(
                f"{self.base_url}/{self.model_name}?key={self.api_key}",
                headers=self._headers(),
                method="GET",
            )
            with urlopen(req, timeout=8):  # nosec B310
                return True
        except Exception:
            return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return "Google API key missing. Set GOOGLE_API_KEY."
        payload = self._chat_payload(messages)
        path = f"/{self.model_name}:generateContent?key={self.api_key}"
        try:
            data = self._post_json(path, payload)
            text = self._extract_final_text(data)
            return text or "Google returned an empty response."
        except HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            return f"Google request failed: HTTP Error {e.code}: {err_msg}"
        except Exception as e:
            return f"Google request failed: {e}"

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.api_key:
            yield "Google API key missing. Set GOOGLE_API_KEY."
            return

        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

        payload = self._chat_payload(messages)
        path = f"/{self.model_name}:streamGenerateContent?alt=sse&key={self.api_key}"
        emitted = False
        in_think = False
        try:
            for data in self._stream_post_json(
                path, payload, timeout=self.stream_timeout
            ):
                thinking, content = self._extract_stream_delta(data)
                if thinking:
                    if not in_think:
                        yield "<think>"
                        in_think = True
                    yield thinking
                    emitted = True
                if content:
                    if in_think:
                        yield "</think>"
                        in_think = False
                    yield content
                    emitted = True
            if in_think:
                yield "</think>"
            if not emitted:
                yield from self._stream_text(self.generate(messages))
        except HTTPError as e:
            if in_think:
                yield "</think>"
            err_msg = e.read().decode("utf-8", errors="ignore")
            yield from self._stream_text(
                f"Google stream failed: HTTP Error {e.code} {err_msg}"
            )
        except Exception as e:
            if in_think:
                yield "</think>"
            fallback_text = self.generate(messages)
            if "Google request failed" in fallback_text:
                fallback_text = f"Google stream failed: {e}. {fallback_text}"
            yield from self._stream_text(fallback_text)


class NvidiaModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url
        self.provider = "nvidia"
        self.timeout = timeout
        self.native_streaming = True
        self.connect_log = []
        self.stream_mode = os.getenv("ASSISTANT_STREAM_MODE", "auto").strip().lower()
        if self.stream_mode not in {"auto", "native", "chunk"}:
            self.stream_mode = "auto"
        self.stream_timeout = _env_int("ASSISTANT_STREAM_TIMEOUT", 35)
        self.temperature = _env_float("ASSISTANT_TEMPERATURE", 0.2)
        self.top_p = _env_float("ASSISTANT_TOP_P", 0.95)
        self.max_tokens = _env_int("ASSISTANT_NUM_PREDICT", 16384)
        self.enable_reasoning = _env_bool("NVIDIA_ENABLE_REASONING", True)

    def set_max_output_tokens(self, value: int) -> tuple[bool, str]:
        if value <= 0:
            return False, "max output tokens must be > 0"
        self.max_tokens = int(value)
        return True, f"nvidia max_tokens set to {self.max_tokens}"

    def get_max_output_tokens(self) -> int | None:
        return int(self.max_tokens) if isinstance(self.max_tokens, int) else None

    def get_context_window(self) -> int | None:
        return None

    def set_stream_mode(self, mode: str) -> tuple[bool, str]:
        m = (mode or "").strip().lower()
        if m not in {"auto", "native", "chunk"}:
            return False, "stream mode must be one of: auto, native, chunk"
        self.stream_mode = m
        return True, f"nvidia stream mode set to {m}"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def apply_auto_limits(self) -> tuple[bool, str]:
        return True, "nvidia auto limits ignored"

    def get_auto_limits(self) -> dict[str, Any] | None:
        return None

    def set_temperature(self, value: float) -> tuple[bool, str]:
        try:
            temp = float(value)
            if temp < 0.0 or temp > 2.0:
                return False, "temperature must be between 0.0 and 2.0"
            self.temperature = temp
            return True, f"nvidia temperature set to {temp}"
        except (ValueError, TypeError):
            return False, "invalid temperature value (must be a number)"

    def get_temperature(self) -> float | None:
        return self.temperature

    def set_top_p(self, value: float) -> tuple[bool, str]:
        try:
            top_p = float(value)
            if top_p < 0.0 or top_p > 1.0:
                return False, "top_p must be between 0.0 and 1.0"
            self.top_p = top_p
            return True, f"nvidia top_p set to {top_p}"
        except (ValueError, TypeError):
            return False, "invalid top_p value (must be a number)"

    def get_top_p(self) -> float | None:
        return self.top_p

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        with urlopen(req, timeout=self.timeout) as resp:  # nosec B310
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _stream_post_json(
        self, path: str, payload: dict[str, Any], timeout: int | None = None
    ) -> Iterator[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        req.add_header("Accept", "text/event-stream")
        effective_timeout = (
            timeout if isinstance(timeout, int) and timeout > 0 else self.timeout
        )
        with urlopen(req, timeout=effective_timeout) as resp:  # nosec B310
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    data_chunk = line[5:].strip()
                    if data_chunk == "[DONE]":
                        break
                else:
                    data_chunk = line
                try:
                    yield json.loads(data_chunk)
                except json.JSONDecodeError:
                    continue

    def _chat_payload(
        self, messages: list[dict[str, str]], stream: bool
    ) -> dict[str, Any]:
        compatible_messages = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "tool":
                role = "user"
                content = f"[System: Tool Execution Result]\n{content}"
            compatible_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": compatible_messages,
            "stream": stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.enable_reasoning:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        return payload

    @staticmethod
    def _extract_final_text(data: dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        first = choices[0]
        message = first.get("message", {})
        content = _message_content_to_text(message.get("content", ""))
        return content

    @staticmethod
    def _extract_stream_delta(data: dict[str, Any]) -> tuple[str, str]:
        choices = data.get("choices", [])
        if not choices:
            return "", ""
        delta = choices[0].get("delta", {})
        content = _message_content_to_text(delta.get("content", ""))
        return "", content

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            req = Request(
                f"{self.base_url}/models",
                headers=self._headers(),
                method="GET",
            )
            with urlopen(req, timeout=8):  # nosec B310
                return True
        except Exception:
            return False

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return "Nvidia API key missing. Set NVIDIA_API_KEY."
        payload = self._chat_payload(messages, stream=False)
        try:
            data = self._post_json("/chat/completions", payload)
            text = self._extract_final_text(data)
            return text or "Nvidia returned an empty response."
        except HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            return f"Nvidia request failed: HTTP Error {e.code}: {err_msg}"
        except Exception as e:
            return f"Nvidia request failed: {e}"

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.api_key:
            yield "Nvidia API key missing. Set NVIDIA_API_KEY."
            return

        if self.stream_mode == "chunk":
            yield from self._stream_text(self.generate(messages))
            return

        payload = self._chat_payload(messages, stream=True)
        emitted = False
        try:
            for data in self._stream_post_json(
                "/chat/completions", payload, timeout=self.stream_timeout
            ):
                _, content = self._extract_stream_delta(data)
                if content:
                    yield content
                    emitted = True
            if not emitted:
                yield from self._stream_text(self.generate(messages))
        except HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            yield from self._stream_text(
                f"Nvidia stream failed: HTTP Error {e.code} {err_msg}"
            )
        except Exception as e:
            fallback_text = self.generate(messages)
            if "Nvidia request failed" in fallback_text:
                fallback_text = f"Nvidia stream failed: {e}. {fallback_text}"
            yield from self._stream_text(fallback_text)


class FallbackModel(BaseModel):
    CONTINUATION_PREFIX = "Tool execution is complete."

    def __init__(self, reason: str = "") -> None:
        self.reason = reason.strip()
        self.provider = "fallback"
        self.model_name = "unavailable"
        self.endpoint = ""
        self.native_streaming = False
        self.connect_log = []
        self.stream_mode = "chunk"

    def get_stream_mode(self) -> str:
        return self.stream_mode

    def generate(self, messages: list[dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if content.startswith(self.CONTINUATION_PREFIX):
                    continue
                last_user = content
                break
        reason_text = f" Reason: {self.reason}." if self.reason else ""
        return (
            "Model backend not available."
            f"{reason_text} "
            "Start Ollama or configure OpenRouter and retry. "
            "No tool calls were executed. "
            f"Last user message: {last_user[:300]}"
        )

    def stream_generate(self, messages: list[dict[str, str]]) -> Iterator[str]:
        yield from self._stream_text(self.generate(messages))


def _fetch_openrouter_models(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], bool]:
    """Returns (model_list, reachable). reachable=True means the key/endpoint is valid."""
    key = api_key.strip()
    if not key:
        return [], False
    req = Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urlopen(
            req, timeout=timeout
        ) as resp:  # nosec B310 - HTTPS OpenRouter endpoint
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        data = payload.get("data", [])
        return (data if isinstance(data, list) else []), True
    except Exception:
        return [], False


def _extract_openrouter_context_window(model_payload: dict[str, Any]) -> int | None:
    if not isinstance(model_payload, dict):
        return None
    keys = (
        "context_length",
        "max_context_length",
        "max_input_tokens",
        "max_tokens",
        "token_limit",
        "max_completion_tokens",
        "max_output_tokens",
    )

    def _to_positive_int(value: Any) -> int | None:
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, float) and value > 0:
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                n = int(raw)
                return n if n > 0 else None
        return None

    for key in keys:
        parsed = _to_positive_int(model_payload.get(key))
        if parsed:
            return parsed
    top = model_payload.get("top_provider")
    if isinstance(top, dict):
        for key in keys:
            parsed = _to_positive_int(top.get(key))
            if parsed:
                return parsed
    return None


def list_openrouter_models(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: int = 20,
) -> list[str]:
    data, _ = _fetch_openrouter_models(
        api_key=api_key, base_url=base_url, timeout=timeout
    )
    names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            names.append(model_id.strip())
    return sorted(set(names))


def build_model(
    model_name: str,
    provider: str = "auto",
    ollama_url: str = "http://127.0.0.1:11434",
    openrouter_url: str = "https://openrouter.ai/api/v1",
    openrouter_api_key: str | None = None,
    google_api_key: str | None = None,
    nvidia_api_key: str | None = None,
) -> BaseModel:
    provider_key = (provider or "auto").strip().lower()
    if provider_key not in {"auto", "ollama", "openrouter", "google", "nvidia"}:
        provider_key = "auto"

    api_key = (openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
    g_api_key = (
        google_api_key or os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    ).strip()
    n_api_key = (nvidia_api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    connect_log: list[str] = []

    if provider_key == "nvidia":
        if not n_api_key:
            connect_log.append("[fail] nvidia api key missing")
            fb = FallbackModel(reason="NVIDIA_API_KEY missing")
            fb.connect_log = connect_log
            return fb
        connect_log.append("[try] nvidia")
        nvidia_model = NvidiaModel(model_name=model_name, api_key=n_api_key)
        connect_log.append(f"[ok] nvidia connected, model={model_name}")
        nvidia_model.connect_log = connect_log
        return nvidia_model

    if provider_key == "google":
        if not g_api_key:
            connect_log.append("[fail] google api key missing")
            fb = FallbackModel(reason="GOOGLE_API_KEY missing")
            fb.connect_log = connect_log
            return fb
        connect_log.append("[try] google ai studio")
        google_model = GoogleModel(model_name=model_name, api_key=g_api_key)
        connect_log.append(f"[ok] google connected, model={model_name}")
        google_model.connect_log = connect_log
        return google_model

    if provider_key == "ollama":
        connect_log.append(f"[try] ollama @ {ollama_url}")
        ollama = OllamaModel(model_name=model_name, base_url=ollama_url)
        if ollama.is_available():
            connect_log.append(f"[ok] ollama connected, model={model_name}")
            connect_log.append(
                f"[ok] local limits ctx={ollama.get_context_window()} maxout={ollama.get_max_output_tokens()} "
                f"(ram≈{getattr(ollama, 'total_ram_gb', '?')}GB, params≈{getattr(ollama, 'model_params_b', '?')}B)"
            )
            ollama.connect_log = connect_log
            return ollama
        connect_log.append("[fail] ollama unavailable")
        fb = FallbackModel(reason="Ollama unavailable")
        fb.connect_log = connect_log
        return fb

    if provider_key == "openrouter":
        if not api_key:
            connect_log.append("[fail] openrouter api key missing")
            fb = FallbackModel(reason="OPENROUTER_API_KEY missing")
            fb.connect_log = connect_log
            return fb
        connect_log.append(f"[try] openrouter @ {openrouter_url}")
        model_payloads, reachable = _fetch_openrouter_models(
            api_key=api_key, base_url=openrouter_url, timeout=12
        )
        if not reachable:
            connect_log.append("[fail] openrouter unavailable or unauthorized")
            fb = FallbackModel(reason="OpenRouter unavailable or unauthorized")
            fb.connect_log = connect_log
            return fb
        available_models = [
            str(item.get("id", "")).strip()
            for item in model_payloads
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and str(item.get("id", "")).strip()
        ]
        resolved_model, model_note = _pick_openrouter_model(
            available=available_models,
            requested=model_name,
            env_fallback=os.getenv(
                "OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free"
            ),
        )
        if resolved_model != model_name:
            connect_log.append(f"[warn] requested model '{model_name}' unavailable")
            connect_log.append(f"[ok] using '{resolved_model}' ({model_note})")
        else:
            connect_log.append(f"[ok] using requested model '{resolved_model}'")
        context_window = None
        for item in model_payloads:
            if (
                isinstance(item, dict)
                and str(item.get("id", "")).strip() == resolved_model
            ):
                context_window = _extract_openrouter_context_window(item)
                break
        if context_window:
            connect_log.append(f"[ok] model context window={context_window}")
        openrouter = OpenRouterModel(
            model_name=resolved_model,
            api_key=api_key,
            base_url=openrouter_url,
            context_window=context_window,
        )
        # reachable already confirmed above; skip redundant is_available() call
        connect_log.append(f"[ok] openrouter connected, model={resolved_model}")
        connect_log.append(
            f"[ok] online limits maxout={openrouter.get_max_output_tokens()} "
            f"(context={openrouter.get_context_window() or 'unknown'})"
        )
        if openrouter.provider_order:
            mode = "only" if openrouter.provider_only else "order"
            connect_log.append(
                f"[ok] provider routing: {mode}={openrouter.provider_order}"
            )
        openrouter.connect_log = connect_log
        return openrouter

    # auto: prefer local first, then OpenRouter.
    connect_log.append(f"[try] auto->ollama @ {ollama_url}")
    ollama = OllamaModel(model_name=model_name, base_url=ollama_url)
    if ollama.is_available():
        connect_log.append(f"[ok] auto selected ollama, model={model_name}")
        connect_log.append(
            f"[ok] local limits ctx={ollama.get_context_window()} maxout={ollama.get_max_output_tokens()} "
            f"(ram≈{getattr(ollama, 'total_ram_gb', '?')}GB, params≈{getattr(ollama, 'model_params_b', '?')}B)"
        )
        ollama.connect_log = connect_log
        return ollama
    connect_log.append("[fail] auto ollama unavailable")
    if api_key:
        connect_log.append(f"[try] auto->openrouter @ {openrouter_url}")
        model_payloads, reachable = _fetch_openrouter_models(
            api_key=api_key, base_url=openrouter_url, timeout=12
        )
        if not reachable:
            connect_log.append("[fail] auto openrouter unavailable or unauthorized")
        else:
            available_models = [
                str(item.get("id", "")).strip()
                for item in model_payloads
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and str(item.get("id", "")).strip()
            ]
            resolved_model, model_note = _pick_openrouter_model(
                available=available_models,
                requested=model_name,
                env_fallback=os.getenv(
                    "OPENROUTER_FALLBACK_MODEL", "arcee-ai/trinity-large-preview:free"
                ),
            )
            if resolved_model != model_name:
                connect_log.append(f"[warn] requested model '{model_name}' unavailable")
                connect_log.append(f"[ok] using '{resolved_model}' ({model_note})")
            else:
                connect_log.append(f"[ok] using requested model '{resolved_model}'")
            context_window = None
            for item in model_payloads:
                if (
                    isinstance(item, dict)
                    and str(item.get("id", "")).strip() == resolved_model
                ):
                    context_window = _extract_openrouter_context_window(item)
                    break
            if context_window:
                connect_log.append(f"[ok] model context window={context_window}")
            openrouter = OpenRouterModel(
                model_name=resolved_model,
                api_key=api_key,
                base_url=openrouter_url,
                context_window=context_window,
            )
            # reachable already confirmed above; skip redundant is_available() call
            connect_log.append(f"[ok] auto selected openrouter, model={resolved_model}")
            connect_log.append(
                f"[ok] online limits maxout={openrouter.get_max_output_tokens()} "
                f"(context={openrouter.get_context_window() or 'unknown'})"
            )
            if openrouter.provider_order:
                mode = "only" if openrouter.provider_only else "order"
                connect_log.append(
                    f"[ok] provider routing: {mode}={openrouter.provider_order}"
                )
            openrouter.connect_log = connect_log
            return openrouter
    else:
        connect_log.append("[skip] auto openrouter (api key missing)")
    if g_api_key:
        connect_log.append("[try] auto->google")
        google_model = GoogleModel(model_name=model_name, api_key=g_api_key)
        connect_log.append(f"[ok] auto selected google, model={model_name}")
        google_model.connect_log = connect_log
        return google_model
    else:
        connect_log.append("[skip] auto google (api key missing)")
    if n_api_key:
        connect_log.append("[try] auto->nvidia")
        nvidia_model = NvidiaModel(model_name=model_name, api_key=n_api_key)
        connect_log.append(f"[ok] auto selected nvidia, model={model_name}")
        nvidia_model.connect_log = connect_log
        return nvidia_model
    else:
        connect_log.append("[skip] auto nvidia (api key missing)")
    fb = FallbackModel(reason="No available backend (Ollama/OpenRouter/Google/Nvidia)")
    fb.connect_log = connect_log
    return fb
