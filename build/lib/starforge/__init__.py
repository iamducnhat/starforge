from __future__ import annotations

from typing import Any

from .adapters import get_adapter
from .engine import StarforgeRuntime
from .memory import MemoryStore
from .tools import ToolRegistry

__all__ = ["MemoryStore", "StarforgeRuntime", "create_runtime", "run"]


def create_runtime(
    *,
    adapter: str | None = None,
    extra_tools: list[object] | None = None,
    memory_root: str | None = None,
) -> StarforgeRuntime:
    selected_adapter = get_adapter(adapter)
    registry = ToolRegistry()
    selected_adapter.configure(registry)
    for tool in extra_tools or []:
        registry.register(tool)
    memory_store = MemoryStore(root=memory_root)
    return StarforgeRuntime(registry=registry, memory_store=memory_store)


def run(
    objective: str,
    context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    selected_adapter = get_adapter(config.get("adapter"))
    runtime_context = selected_adapter.merge_context(dict(context or {}))
    runtime = create_runtime(adapter=selected_adapter.name, memory_root=config.get("memory_root"))
    return runtime.run(objective=objective, context=runtime_context, config=config)
