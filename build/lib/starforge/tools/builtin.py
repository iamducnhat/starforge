from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from ..context import RuntimeContext
from ..observations import Observation


@dataclass(slots=True)
class RunCommandTool:
    name: str = "run_command"
    description: str = "Run a shell command inside the working directory."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        command = str(arguments.get("command") or arguments.get("cmd") or "").strip()
        if not command:
            raise ValueError("run_command requires 'command'")
        timeout = max(1, int(arguments.get("timeout", 120)))
        completed = subprocess.run(
            command,
            cwd=context.working_dir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return Observation(
            type="command_result",
            content={
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
            },
            metadata={
                "working_dir": str(context.working_dir),
                "timeout": timeout,
                "success": completed.returncode == 0,
            },
        )


@dataclass(slots=True)
class ReadFileTool:
    name: str = "read_file"
    description: str = "Read a text file from the working directory."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        path = _resolve_path(arguments, context, key="path")
        content = path.read_text(encoding="utf-8")
        return Observation(
            type="file_read",
            content=content,
            metadata={
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
            },
        )


@dataclass(slots=True)
class WriteFileTool:
    name: str = "write_file"
    description: str = "Write text content to a file in the working directory."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        path = _resolve_path(arguments, context, key="path")
        content = str(arguments.get("content", ""))
        append = bool(arguments.get("append", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            existing = path.read_text(encoding="utf-8") if path.exists() else ""
            content_to_write = existing + content
        else:
            content_to_write = content
        path.write_text(content_to_write, encoding="utf-8")
        return Observation(
            type="file_write",
            content={
                "path": str(path),
                "bytes": len(content_to_write.encode("utf-8")),
            },
            metadata={
                "path": str(path),
                "append": append,
            },
        )


@dataclass(slots=True)
class ListFilesTool:
    name: str = "list_files"
    description: str = "List files below the working directory."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        relative = str(arguments.get("path", "."))
        base = _resolve_path({"path": relative}, context, key="path", must_exist=True)
        limit = max(1, min(int(arguments.get("limit", 100)), 500))
        files: list[str] = []
        for candidate in sorted(base.rglob("*")):
            if candidate.is_file():
                files.append(str(candidate.relative_to(context.working_dir)))
            if len(files) >= limit:
                break
        return Observation(
            type="filesystem_snapshot",
            content=files,
            metadata={
                "root": str(base),
                "count": len(files),
            },
        )


@dataclass(slots=True)
class HttpRequestTool:
    name: str = "http_request"
    description: str = "Call an HTTP API and normalize the response."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        del context
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise ValueError("http_request requires 'url'")
        method = str(arguments.get("method", "GET")).upper()
        timeout = max(1, int(arguments.get("timeout", 30)))
        headers = dict(arguments.get("headers") or {})
        params = dict(arguments.get("params") or {})
        json_payload = arguments.get("json")
        data_payload = arguments.get("data")
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_payload,
            data=data_payload,
            timeout=timeout,
        )
        content_type = response.headers.get("content-type", "")
        parsed: Any
        if "application/json" in content_type:
            parsed = response.json()
        else:
            parsed = response.text
        return Observation(
            type="api_response",
            content=parsed,
            metadata={
                "url": response.url,
                "status_code": response.status_code,
                "content_type": content_type,
                "ok": response.ok,
            },
        )


@dataclass(slots=True)
class WebSearchTool:
    name: str = "web_search"
    description: str = "Search the web and return structured search results."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        del context
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("web_search requires 'query'")
        limit = max(1, min(int(arguments.get("limit", 5)), 10))
        timeout = max(1, int(arguments.get("timeout", 20)))

        instant = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1,
            },
            timeout=timeout,
        )
        payload = instant.json()
        items = _extract_duckduckgo_items(payload)
        if len(items) < limit:
            html_results = _search_duckduckgo_html(query=query, limit=limit, timeout=timeout)
            for item in html_results:
                if item not in items:
                    items.append(item)
                if len(items) >= limit:
                    break
        return Observation(
            type="search_results",
            content=items[:limit],
            metadata={
                "query": query,
                "count": min(len(items), limit),
                "source": "duckduckgo",
            },
        )


@dataclass(slots=True)
class SearchTool(WebSearchTool):
    name: str = "search"
    description: str = "Alias for web_search."


@dataclass(slots=True)
class ReadWebpageTool:
    name: str = "read_webpage"
    description: str = "Fetch a webpage and normalize its readable content."

    def run(self, arguments: dict[str, Any], context: RuntimeContext) -> Observation:
        del context
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise ValueError("read_webpage requires 'url'")
        timeout = max(1, int(arguments.get("timeout", 20)))
        response = requests.get(
            url,
            timeout=timeout,
            headers={"user-agent": "starforge/1.0"},
        )
        html = response.text
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = _strip_html(title_match.group(1)) if title_match else ""
        readable_text = _strip_html(html)
        max_chars = max(200, min(int(arguments.get("max_chars", 6000)), 20000))
        excerpt = readable_text[:max_chars]
        return Observation(
            type="webpage_read",
            content={
                "title": title,
                "text": excerpt,
                "status_code": response.status_code,
            },
            metadata={
                "url": response.url,
                "title": title,
                "status_code": response.status_code,
                "chars": len(excerpt),
            },
        )


def _resolve_path(
    arguments: dict[str, Any],
    context: RuntimeContext,
    *,
    key: str,
    must_exist: bool = False,
) -> Path:
    raw_path = str(arguments.get(key) or "").strip()
    if not raw_path:
        raise ValueError(f"missing path: {key}")
    path = Path(raw_path)
    resolved = path if path.is_absolute() else context.working_dir / path
    resolved = resolved.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def _extract_duckduckgo_items(payload: dict[str, Any]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    abstract = str(payload.get("AbstractText") or "").strip()
    abstract_url = str(payload.get("AbstractURL") or "").strip()
    heading = str(payload.get("Heading") or "").strip() or "DuckDuckGo"
    if abstract:
        items.append({"title": heading, "url": abstract_url, "snippet": abstract})

    def visit(topic: Any) -> None:
        if not isinstance(topic, dict):
            return
        text = str(topic.get("Text") or "").strip()
        url = str(topic.get("FirstURL") or "").strip()
        if text:
            title = text.split(" - ", 1)[0]
            items.append({"title": title, "url": url, "snippet": text})
        for nested in topic.get("Topics", []) or []:
            visit(nested)

    for topic in payload.get("RelatedTopics", []) or []:
        visit(topic)
    return items


def _search_duckduckgo_html(query: str, limit: int, timeout: int) -> list[dict[str, str]]:
    response = requests.get(
        f"https://duckduckgo.com/html/?{urlencode({'q': query})}",
        timeout=timeout,
        headers={"user-agent": "starforge/1.0"},
    )
    html = response.text
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    results: list[dict[str, str]] = []
    for match in pattern.finditer(html):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        url = unescape(match.group("url"))
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", unescape(text)).strip()
