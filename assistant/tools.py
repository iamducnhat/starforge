from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

from .functions_registry import FunctionRegistry
from .logging_config import get_logger
from .memory import MemoryStore
from .utils import get_env_float, get_env_int, normalize_keywords
from .web import extract_code_snippets, read_web, scrape_web, search_web
from .workspace_tools import WorkspaceTools


logger = get_logger(__name__)

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
        self._tools: dict[str, Callable[..., Any]] = {
            "find_in_memory": self.find_in_memory,
            "create_block": self.create_block,
            "create_function": self.create_function,
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

    def execute(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            logger.error(f"Unknown tool requested: {name}")
            return {"ok": False, "error": f"unknown tool: {name}"}

        args = args or {}
        logger.debug(f"Executing tool: {name} with args: {args}")
        try:
            result = self._tools[name](**args)
            if isinstance(result, dict) and "ok" in result:
                if not result.get("ok"):
                    logger.warning(f"Tool {name} returned error: {result.get('error')}")
                return result
            return {"ok": True, "result": result}
        except TypeError as e:
            logger.error(f"Invalid arguments for tool {name}: {e}")
            return {"ok": False, "error": f"invalid args for {name}: {e}"}
        except Exception as e:  # pragma: no cover
            logger.exception(f"Tool execution failed: {name}: {e}")
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

    def run_function(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def create_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        return self.workspace_tools.create_file(path=path, content=content, overwrite=overwrite)

    def create_folder(self, path: str) -> dict[str, Any]:
        return self.workspace_tools.create_folder(path=path)

    def delete_file(self, path: str) -> dict[str, Any]:
        return self.workspace_tools.delete_file(path=path)

    def run_terminal(self, action: str, cmd: str | None = None, session_id: str = "default") -> dict[str, Any]:
        return self.workspace_tools.run_terminal(action=action, cmd=cmd, session_id=session_id)

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        # Backward-compatible alias.
        return self.workspace_tools.write_file(path=path, content=content, append=append)

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

    def create_plan(self, title: str, goal: str, steps: list[str]) -> dict[str, Any]:
        return self.workspace_tools.create_plan(title=title, goal=goal, steps=steps)

    def list_plans(self) -> dict[str, Any]:
        return self.workspace_tools.list_plans()

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        return self.workspace_tools.get_plan(plan_id=plan_id)

    def add_todo(self, plan_id: str, text: str) -> dict[str, Any]:
        return self.workspace_tools.add_todo(plan_id=plan_id, text=text)

    def update_todo(self, plan_id: str, todo_id: int, status: str) -> dict[str, Any]:
        return self.workspace_tools.update_todo(plan_id=plan_id, todo_id=todo_id, status=status)
