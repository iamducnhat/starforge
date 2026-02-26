from __future__ import annotations

import re
from typing import Any, Callable

from .functions_registry import FunctionRegistry
from .memory import MemoryStore
from .utils import normalize_keywords
from .web import extract_code_snippets, search_web


class ToolSystem:
    def __init__(self, memory_store: MemoryStore, function_registry: FunctionRegistry) -> None:
        self.memory_store = memory_store
        self.function_registry = function_registry
        self._tools: dict[str, Callable[..., Any]] = {
            "find_in_memory": self.find_in_memory,
            "create_block": self.create_block,
            "create_function": self.create_function,
            "search_web": self.search_web,
            "extract_code_snippets": self.extract_code_snippets,
        }

    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"unknown tool: {name}"}

        args = args or {}
        try:
            result = self._tools[name](**args)
            if isinstance(result, dict) and "ok" in result:
                return result
            return {"ok": True, "result": result}
        except TypeError as e:
            return {"ok": False, "error": f"invalid args for {name}: {e}"}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": f"tool execution failed: {e}"}

    def find_in_memory(self, keywords: list[str]) -> dict[str, Any]:
        matches = self.memory_store.find_in_memory(keywords)
        return {"ok": True, "matches": matches}

    def create_block(
        self,
        name: str,
        topic: str,
        keywords: list[str],
        knowledge: str,
        source: list[str],
    ) -> dict[str, Any]:
        return self.memory_store.create_block(
            name=name,
            topic=topic,
            keywords=keywords,
            knowledge=knowledge,
            source=source,
        )

    def create_function(
        self,
        name: str,
        description: str,
        code: str,
        keywords: list[str],
    ) -> dict[str, Any]:
        return self.function_registry.create_function(
            name=name,
            description=description,
            code=code,
            keywords=keywords,
        )

    @staticmethod
    def _web_keywords(query: str, results: list[dict[str, Any]], limit: int = 10) -> list[str]:
        stop = {
            "to",
            "in",
            "of",
            "on",
            "at",
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "what",
            "when",
            "where",
            "which",
            "want",
            "learn",
            "more",
            "about",
            "give",
            "some",
            "example",
            "examples",
            "please",
        }
        words = re.findall(r"[a-zA-Z0-9_]{2,}", query.lower())
        for item in results[:5]:
            words.extend(re.findall(r"[a-zA-Z0-9_]{2,}", str(item.get("title", "")).lower()))

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
        return normalize_keywords(out) or ["web", "search"]

    @staticmethod
    def _web_knowledge(query: str, results: list[dict[str, Any]], search_meta: dict[str, Any] | None = None) -> str:
        search_meta = search_meta or {}
        lines = [
            "## Compressed Summary",
            f"- Query: {query}",
            f"- Search engine: {search_meta.get('engine', 'unknown')}",
            f"- Search level: {search_meta.get('level_used', 'unknown')}",
            f"- Pages scanned: {search_meta.get('pages_scanned', 0)}",
            "- Top findings:",
        ]
        for idx, item in enumerate(results[:5], start=1):
            title = str(item.get("title", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            url = str(item.get("url", "")).strip()
            page_excerpt = str(item.get("page_excerpt", "")).strip()
            page_fetched = bool(item.get("page_fetched", False))
            lines.append(f"- [{idx}] {title} | {url}")
            lines.append(f"- Page fetched {idx}: {page_fetched}")
            if page_excerpt:
                lines.append(f"- Page excerpt {idx}: {page_excerpt}")
            if snippet:
                lines.append(f"- Snippet {idx}: {snippet}")
            code_snippets = item.get("code_snippets", [])
            if isinstance(code_snippets, list) and code_snippets:
                lines.append(f"- Code snippet {idx}: {str(code_snippets[0]).strip()}")

        lines.extend(
            [
                "",
                "## Reusable Patterns",
                "- Use official or primary sources first when available.",
                "- Cross-check at least two sources before treating facts as stable.",
                "",
                "## Minimal Snippets",
                "```text",
            ]
        )
        for item in results[:3]:
            page_excerpt = str(item.get("page_excerpt", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if page_excerpt:
                lines.append(page_excerpt)
            elif snippet:
                lines.append(snippet)
        lines.append("```")
        return "\n".join(lines).strip() + "\n"

    def search_web(
        self,
        query: str,
        max_results: int | None = None,
        fetch_top_pages: int | None = None,
        page_timeout: int | None = None,
        level: str = "auto",
    ) -> dict[str, Any]:
        search = search_web(
            query,
            max_results=max_results,
            fetch_top_pages=fetch_top_pages,
            page_timeout=page_timeout,
            level=level,
        )
        results = search.get("results", [])
        saved_block: dict[str, Any] | None = None

        if results:
            sources = [str(item.get("url", "")).strip() for item in results if str(item.get("url", "")).strip()]
            try:
                saved_block = self.memory_store.create_block(
                    name=f"web_{query[:64]}",
                    topic=f"web_search:{query}",
                    keywords=self._web_keywords(query=query, results=results),
                    knowledge=self._web_knowledge(
                        query=query,
                        results=results,
                        search_meta=search.get("meta", {}),
                    ),
                    source=sources,
                )
            except Exception as e:  # pragma: no cover
                saved_block = {"ok": False, "error": f"failed to save web block: {e}"}

        return {"ok": True, "search": search, "saved_block": saved_block}

    def extract_code_snippets(self, html: str) -> dict[str, Any]:
        return {"ok": True, "snippets": extract_code_snippets(html)}
