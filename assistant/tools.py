from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .functions_registry import FunctionRegistry
from .logging_config import get_logger
from .memory import MemoryStore
from .utils import get_env_float, get_env_int, normalize_keywords
from .web import extract_code_snippets, read_web, scrape_web, search_web
from .workspace_tools import WorkspaceTools

logger = get_logger(__name__)

_TOOL_NAME_ALIASES: dict[str, str] = {
    "google_search": "search_web",
    "web_search": "search_web",
    "search_google": "search_web",
    "internet_search": "search_web",
    "websearch": "search_web",
    "search_manga": "search_web",
    "manga_search": "search_web",
    "read_url": "read_web",
    "crawl_web": "scrape_web",
}


class ToolSystem:
    def __init__(
        self,
        memory_store: MemoryStore,
        function_registry: FunctionRegistry,
        workspace_tools: WorkspaceTools,
    ) -> None:
        self.memory_store = memory_store
        self.function_registry = function_registry
        self.workspace_tools = workspace_tools
        self.tool_max_retries = max(1, min(get_env_int("ASSISTANT_TOOL_RETRIES", 2), 5))
        self.tool_retry_backoff = max(
            0.05, min(get_env_float("ASSISTANT_TOOL_RETRY_BACKOFF", 0.35), 3.0)
        )
        self._tools: dict[str, Callable[..., Any]] = {
            "find_in_memory": self.find_in_memory,
            "search_memory": self.search_memory,
            "record_memory_feedback": self.record_memory_feedback,
            "find_strategies": self.find_strategies,
            "record_strategy": self.record_strategy,
            "create_block": self.create_block,
            "create_function": self.create_function,
            "create_skill": self.create_skill,
            "list_skills": self.list_skills,
            "find_skills": self.find_skills,
            "record_skill_outcome": self.record_skill_outcome,
            "search_web": self.search_web,
            "read_web": self.read_web,
            "scrape_web": self.scrape_web,
            "extract_code_snippets": self.extract_code_snippets,
            "list_files": self.list_files,
            "read_file": self.read_file,
            "create_file": self.create_file,
            "create_folder": self.create_folder,
            "delete_file": self.delete_file,
            "run_terminal": self.run_terminal,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "search_project": self.search_project,
            "index_symbols": self.index_symbols,
            "lookup_symbol": self.lookup_symbol,
            "summarize_file": self.summarize_file,
            "detect_project_context": self.detect_project_context,
            "execute_command": self.execute_command,
            "run_tests": self.run_tests,
            "get_git_diff": self.get_git_diff,
            "validate_workspace_changes": self.validate_workspace_changes,
            "create_plan": self.create_plan,
            "list_plans": self.list_plans,
            "get_plan": self.get_plan,
            "add_todo": self.add_todo,
            "update_todo": self.update_todo,
            "get_current_datetime": self.get_current_datetime,
            "run_function": self.run_function,
        }

    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    @staticmethod
    def _canonical_tool_name(name: str) -> str:
        key = str(name or "").strip().lower()
        if not key:
            return ""
        return _TOOL_NAME_ALIASES.get(key, key)

    @staticmethod
    def _normalize_tool_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(args or {})

        if name == "get_current_datetime":
            return {}

        if name == "search_web":
            query = normalized.get("query")
            if not isinstance(query, str) or not query.strip():
                queries = normalized.get("queries")
                if isinstance(queries, list):
                    for item in queries:
                        s = str(item).strip()
                        if s:
                            normalized["query"] = s
                            break
                elif isinstance(queries, str) and queries.strip():
                    normalized["query"] = queries.strip()

            if "max_results" not in normalized and "limit" in normalized:
                normalized["max_results"] = normalized.get("limit")

            allowed = {"query", "max_results", "fetch_top_pages", "page_timeout", "level"}
            normalized = {
                k: v for k, v in normalized.items() if k in allowed and v is not None
            }
            return normalized

        if name == "find_in_memory":
            keywords = normalized.get("keywords")
            if isinstance(keywords, str):
                parts = [p.strip() for p in re.split(r"[,;\n]+", keywords) if p.strip()]
                normalized["keywords"] = parts if parts else [keywords.strip()]
            elif not isinstance(keywords, list):
                query = normalized.get("query")
                if isinstance(query, str) and query.strip():
                    normalized["keywords"] = [query.strip()]
            return {"keywords": normalized.get("keywords", [])}

        return normalized

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        requested_name = str(name or "").strip()
        canonical_name = self._canonical_tool_name(requested_name)
        if canonical_name not in self._tools:
            logger.error(f"Unknown tool requested: {requested_name}")
            return {"ok": False, "error": f"unknown tool: {requested_name}"}

        if canonical_name != requested_name:
            logger.info(
                f"Aliasing tool name '{requested_name}' -> '{canonical_name}'"
            )

        args = args or {}
        normalized_args = self._normalize_tool_args(canonical_name, args)
        logger.debug(f"Executing tool: {canonical_name} with args: {normalized_args}")
        result = self.safe_tool_call(name=canonical_name, args=normalized_args)
        result.setdefault("name", canonical_name)
        if canonical_name != requested_name:
            result.setdefault("requested_name", requested_name)
            result.setdefault("aliased", True)
        if normalized_args != args:
            result.setdefault("normalized_args", normalized_args)
        return result

    @staticmethod
    def _normalize_tool_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict) and "ok" in result:
            return result
        return {"ok": True, "result": result}

    @staticmethod
    def _is_retryable_error(error: str) -> bool:
        t = (error or "").lower()
        retry_markers = (
            "timeout",
            "timed out",
            "temporarily",
            "temporary",
            "rate limit",
            "429",
            "connection reset",
            "connection aborted",
            "connection refused",
            "network",
            "dns",
            "unavailable",
            "too many requests",
        )
        return any(m in t for m in retry_markers)

    def _fallback_args(
        self, tool_name: str, args: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any] | None:
        def _int_or(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        if result.get("ok", True):
            return None
        fallback = dict(args)
        if tool_name == "search_web":
            fallback["level"] = "quick"
            fallback["max_results"] = min(_int_or(fallback.get("max_results", 5), 5), 5)
            fallback["fetch_top_pages"] = min(
                _int_or(fallback.get("fetch_top_pages", 1), 1), 1
            )
            fallback["page_timeout"] = min(
                _int_or(fallback.get("page_timeout", 8), 8), 8
            )
            return fallback if fallback != args else None
        if tool_name == "read_web":
            fallback["timeout"] = min(_int_or(fallback.get("timeout", 8), 8), 8)
            fallback["max_chars"] = min(
                _int_or(fallback.get("max_chars", 6000), 6000), 6000
            )
            return fallback if fallback != args else None
        if tool_name == "scrape_web":
            fallback["max_pages"] = min(_int_or(fallback.get("max_pages", 8), 8), 8)
            fallback["max_depth"] = min(_int_or(fallback.get("max_depth", 1), 1), 1)
            fallback["timeout"] = min(_int_or(fallback.get("timeout", 8), 8), 8)
            return fallback if fallback != args else None
        return None

    def safe_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        attempts = 0
        used_fallback = False
        current_args = dict(args)

        while True:
            attempts += 1
            try:
                raw = self._tools[name](**current_args)
                result = self._normalize_tool_result(raw)
            except TypeError as e:
                logger.error(f"Invalid arguments for tool {name}: {e}")
                return {
                    "ok": False,
                    "error": f"invalid args for {name}: {e}",
                    "attempts": attempts,
                }
            except Exception as e:  # pragma: no cover
                logger.exception(f"Tool execution failed: {name}: {e}")
                result = {"ok": False, "error": f"tool execution failed: {e}"}

            result["attempts"] = attempts
            if result.get("ok", False):
                if used_fallback:
                    result["fallback_used"] = True
                return result

            error_text = str(result.get("error", ""))
            if attempts <= self.tool_max_retries and self._is_retryable_error(error_text):
                delay = min(5.0, self.tool_retry_backoff * attempts)
                logger.warning(
                    f"Tool {name} failed with retryable error (attempt {attempts}/{self.tool_max_retries}): {error_text}"
                )
                time.sleep(delay)
                continue

            if not used_fallback:
                fallback_args = self._fallback_args(name, current_args, result)
                if fallback_args is not None:
                    logger.warning(f"Tool {name} failed, trying fallback arguments")
                    used_fallback = True
                    current_args = fallback_args
                    continue

            logger.warning(f"Tool {name} returned error: {result.get('error')}")
            return result

    def find_in_memory(self, keywords: list[str]) -> dict[str, Any]:
        matches = self.memory_store.find_in_memory(keywords)
        return {"ok": True, "matches": matches}

    def search_memory(self, query: str, limit: int = 5) -> dict[str, Any]:
        matches = self.memory_store.semantic_search(query=query, limit=limit)
        return {"ok": True, "query": query, "matches": matches}

    def record_memory_feedback(
        self, block_name: str, success: bool, confidence: float = 1.0, source: str = "runtime"
    ) -> dict[str, Any]:
        return self.memory_store.record_feedback(
            block_name=block_name,
            success=success,
            confidence=confidence,
            source=source,
        )

    def find_strategies(self, query: str, limit: int = 5) -> dict[str, Any]:
        matches = self.memory_store.find_strategies(query=query, limit=limit)
        return {"ok": True, "query": query, "matches": matches}

    def record_strategy(
        self,
        goal: str,
        strategy: list[dict[str, Any]],
        success: bool,
        source: str = "runtime",
        notes: str = "",
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.memory_store.record_strategy(
            goal=goal,
            strategy=strategy,
            success=success,
            source=source,
            notes=notes,
            context=context,
        )

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
        keywords: list[str],
        code: str = "",
        tool_name: str = "",
        tool_args: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.function_registry.create_function(
            name=name,
            description=description,
            keywords=keywords,
            code=code,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_calls=tool_calls,
        )

    def create_skill(
        self,
        name: str,
        description: str,
        keywords: list[str],
        code: str = "",
        tool_name: str = "",
        tool_args: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        skill: str = "",
        inputs: list[str] | None = None,
        steps_template: list[dict[str, Any]] | None = None,
        match_conditions: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.function_registry.create_skill(
            name=name,
            description=description,
            keywords=keywords,
            code=code,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_calls=tool_calls,
            skill=skill,
            inputs=inputs,
            steps_template=steps_template,
            match_conditions=match_conditions,
        )

    def list_skills(self, limit: int = 50, query: str = "", min_score: float = 0.0) -> dict[str, Any]:
        return self.function_registry.list_skills(
            limit=limit, query=query, min_score=min_score
        )

    def find_skills(self, query: str, limit: int = 5) -> dict[str, Any]:
        return self.function_registry.find_skills(query=query, limit=limit)

    def record_skill_outcome(
        self, name: str, success: bool, confidence: float = 1.0, notes: str = ""
    ) -> dict[str, Any]:
        return self.function_registry.record_skill_outcome(
            name=name, success=success, confidence=confidence, notes=notes
        )

    def run_function(
        self, name: str, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a previously registered function.

        For 'code' functions, args are passed as keyword arguments to the
        underlying Python function whose name matches the slugified function
        name. For 'tool_macro' functions, the saved tool_calls are replayed
        using this ToolSystem's execute method.
        """
        return self.function_registry.execute_function(
            name=name,
            args=args or {},
            execute_tool=self.execute,
        )

    @staticmethod
    def get_current_datetime() -> dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone()
        return {
            "ok": True,
            "datetime": {
                "utc_iso": now_utc.isoformat(),
                "local_iso": now_local.isoformat(),
                "date": now_local.strftime("%Y-%m-%d"),
                "time": now_local.strftime("%H:%M:%S"),
                "timezone": str(now_local.tzinfo or "local"),
                "year": int(now_local.strftime("%Y")),
                "month": int(now_local.strftime("%m")),
                "day": int(now_local.strftime("%d")),
                "weekday": now_local.strftime("%A"),
                "unix": int(now_utc.timestamp()),
            },
        }

    @staticmethod
    def _web_keywords(
        query: str, results: list[dict[str, Any]], limit: int = 10
    ) -> list[str]:
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
            words.extend(
                re.findall(r"[a-zA-Z0-9_]{2,}", str(item.get("title", "")).lower())
            )

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
    def _web_knowledge(
        query: str,
        results: list[dict[str, Any]],
        search_meta: dict[str, Any] | None = None,
    ) -> str:
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
        meta = search.get("meta", {}) if isinstance(search, dict) else {}
        saved_block: dict[str, Any] | None = None

        if results:
            sources = [
                str(item.get("url", "")).strip()
                for item in results
                if str(item.get("url", "")).strip()
            ]
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

        had_search_error = bool(meta.get("had_search_error", False))
        if had_search_error and not results:
            engine = str(meta.get("engine", "search")).strip() or "search"
            return {
                "ok": False,
                "error": f"{engine} search failed or returned no results",
                "search": search,
                "saved_block": saved_block,
            }

        return {"ok": True, "search": search, "saved_block": saved_block}

    def extract_code_snippets(self, html: str) -> dict[str, Any]:
        return {"ok": True, "snippets": extract_code_snippets(html)}

    def read_web(
        self,
        url: str,
        timeout: int = 12,
        max_chars: int = 12000,
        include_links: bool = True,
        max_links: int = 120,
    ) -> dict[str, Any]:
        return read_web(
            url=url,
            timeout=timeout,
            max_chars=max_chars,
            include_links=include_links,
            max_links=max_links,
        )

    def scrape_web(
        self,
        start_url: str,
        max_pages: int = 20,
        max_depth: int = 2,
        same_domain_only: bool = True,
        include_external: bool = False,
        timeout: int = 10,
        max_links_per_page: int = 120,
    ) -> dict[str, Any]:
        return scrape_web(
            start_url=start_url,
            max_pages=max_pages,
            max_depth=max_depth,
            same_domain_only=same_domain_only,
            include_external=include_external,
            timeout=timeout,
            max_links_per_page=max_links_per_page,
        )

    def list_files(
        self,
        path: str = ".",
        glob: str = "**/*",
        include_hidden: bool = False,
        max_entries: int = 200,
    ) -> dict[str, Any]:
        return self.workspace_tools.list_files(
            path=path,
            glob=glob,
            include_hidden=include_hidden,
            max_entries=max_entries,
        )

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 12000,
    ) -> dict[str, Any]:
        return self.workspace_tools.read_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
        )

    def create_file(
        self, path: str, content: str, overwrite: bool = False
    ) -> dict[str, Any]:
        return self.workspace_tools.create_file(
            path=path, content=content, overwrite=overwrite
        )

    def create_folder(self, path: str) -> dict[str, Any]:
        return self.workspace_tools.create_folder(path=path)

    def delete_file(self, path: str) -> dict[str, Any]:
        return self.workspace_tools.delete_file(path=path)

    def run_terminal(
        self, action: str, cmd: str | None = None, session_id: str = "default"
    ) -> dict[str, Any]:
        return self.workspace_tools.run_terminal(
            action=action, cmd=cmd, session_id=session_id
        )

    def write_file(
        self, path: str, content: str, append: bool = False
    ) -> dict[str, Any]:
        # Backward-compatible alias.
        return self.workspace_tools.write_file(
            path=path, content=content, append=append
        )

    def edit_file(
        self,
        path: str,
        find_text: str,
        replace_text: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        return self.workspace_tools.edit_file(
            path=path,
            find_text=find_text,
            replace_text=replace_text,
            replace_all=replace_all,
        )

    def search_project(
        self,
        query: str,
        path: str = ".",
        glob: str = "**/*",
        case_sensitive: bool = False,
        regex: bool = False,
        max_matches: int = 200,
    ) -> dict[str, Any]:
        return self.workspace_tools.search_project(
            query=query,
            path=path,
            glob=glob,
            case_sensitive=case_sensitive,
            regex=regex,
            max_matches=max_matches,
        )

    def index_symbols(
        self,
        path: str = ".",
        glob: str = "**/*",
        max_files: int = 300,
        max_symbols: int = 5000,
    ) -> dict[str, Any]:
        return self.workspace_tools.index_symbols(
            path=path,
            glob=glob,
            max_files=max_files,
            max_symbols=max_symbols,
        )

    def lookup_symbol(
        self,
        symbol: str,
        path: str = ".",
        glob: str = "**/*",
        exact: bool = False,
        max_results: int = 30,
    ) -> dict[str, Any]:
        return self.workspace_tools.lookup_symbol(
            symbol=symbol,
            path=path,
            glob=glob,
            exact=exact,
            max_results=max_results,
        )

    def summarize_file(self, path: str, max_symbols: int = 20) -> dict[str, Any]:
        return self.workspace_tools.summarize_file(
            path=path,
            max_symbols=max_symbols,
        )

    def detect_project_context(
        self, path: str = ".", include_runtime: bool = True
    ) -> dict[str, Any]:
        return self.workspace_tools.detect_project_context(
            path=path, include_runtime=include_runtime
        )

    def execute_command(
        self,
        cmd: str,
        path: str = ".",
        timeout: int = 120,
        max_output_chars: int = 12000,
    ) -> dict[str, Any]:
        return self.workspace_tools.execute_command(
            cmd=cmd,
            path=path,
            timeout=timeout,
            max_output_chars=max_output_chars,
        )

    def run_tests(
        self,
        path: str = ".",
        runner: str = "auto",
        args: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        return self.workspace_tools.run_tests(
            path=path,
            runner=runner,
            args=args,
            timeout=timeout,
        )

    def get_git_diff(
        self, path: str = ".", staged: bool = False, max_chars: int = 12000
    ) -> dict[str, Any]:
        return self.workspace_tools.get_git_diff(
            path=path, staged=staged, max_chars=max_chars
        )

    def validate_workspace_changes(
        self,
        path: str = ".",
        test_runner: str = "auto",
        test_args: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        return self.workspace_tools.validate_workspace_changes(
            path=path,
            test_runner=test_runner,
            test_args=test_args,
            timeout=timeout,
        )

    def create_plan(self, title: str, goal: str, steps: list[str]) -> dict[str, Any]:
        return self.workspace_tools.create_plan(title=title, goal=goal, steps=steps)

    def list_plans(self) -> dict[str, Any]:
        return self.workspace_tools.list_plans()

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        return self.workspace_tools.get_plan(plan_id=plan_id)

    def add_todo(self, plan_id: str, text: str) -> dict[str, Any]:
        return self.workspace_tools.add_todo(plan_id=plan_id, text=text)

    def update_todo(self, plan_id: str, todo_id: int, status: str) -> dict[str, Any]:
        return self.workspace_tools.update_todo(
            plan_id=plan_id, todo_id=todo_id, status=status
        )
